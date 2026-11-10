"""小红书 Web 自动化后端（T2）。

设计要点
--------
1. 持久化登录态：用 launch_persistent_context，首次人工扫码一次，之后免登录。
   Agent 不尝试绕过登录 —— 扫码/验证码是账号安全边界，必须由人完成。
2. 反自动化检测：真实浏览器内核（优先 msedge channel）+ 去 webdriver 指纹 +
   中文 locale + 人类化操作间隔，避免一上来就被风控拦。
3. 结构化抽取：用 page.evaluate 在页面内一次性取回数据，不依赖 hash class
   （小红书的 class 带 data-v 哈希，逐元素 locator 极易随发版失效）。
4. 只读与写入分离：search/read 只读；publish/like/comment 属于写操作，
   由上层注册表标记 require_confirm。
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

XHS_HOME = "https://www.xiaohongshu.com"
XHS_EXPLORE = "https://www.xiaohongshu.com/explore"
XHS_SEARCH_URL = (
    "https://www.xiaohongshu.com/search_result?keyword={kw}&source=web_search_result_notes"
)
XHS_CREATOR = "https://creator.xiaohongshu.com/publish/publish?source=official"

# 去掉最常见的自动化指纹
STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || { runtime: {} };
const origQuery = window.navigator.permissions && window.navigator.permissions.query;
if (origQuery) {
  window.navigator.permissions.query = (p) =>
    p && p.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : origQuery(p);
}
"""

# 从页面里一次性抽取笔记卡片
EXTRACT_NOTES_JS = r"""
(() => {
  const seen = new Set();
  const out = [];
  const cards = document.querySelectorAll(
    'section.note-item, div.note-item, section[class*="note-item"]'
  );
  const pick = (root, sels) => {
    for (const s of sels) {
      const el = root.querySelector(s);
      if (el) {
        const t = (el.innerText || el.textContent || '').trim();
        if (t) return t;
      }
    }
    return '';
  };
  cards.forEach((el) => {
    // 优先取带 xsec_token 的链接：不带 token 的 /explore/xxx 打开会 404(300031)
    const a =
      el.querySelector('a[href*="xsec_token"]') ||
      el.querySelector('a[href*="/explore/"], a[href*="/discovery/item/"]');
    if (!a) return;
    const href = a.getAttribute('href') || '';
    if (!href || seen.has(href)) return;
    seen.add(href);
    const abs = href.startsWith('http') ? href : location.origin + href;
    const noteId =
      (abs.match(/\/(?:explore|search_result|discovery\/item)\/([0-9a-zA-Z]+)/) || [])[1] || '';
    out.push({
      title: pick(el, ['a.title span', '.footer .title span', '.title', 'img[alt]']),
      author: pick(el, ['.author .name', '.author-wrapper .name', '.name', '.author']),
      likes: pick(el, ['.like-wrapper .count', '.count', '[class*="like"] .count']),
      cover: (el.querySelector('img') || {}).src || '',
      url: abs,
      note_id: noteId,
    });
  });
  return out;
})()
"""

EXTRACT_DETAIL_JS = r"""
(() => {
  const txt = (sels) => {
    for (const s of sels) {
      const el = document.querySelector(s);
      if (el) {
        const t = (el.innerText || el.textContent || '').trim();
        if (t) return t;
      }
    }
    return '';
  };
  const meta = (label) => {
    const nodes = document.querySelectorAll('span, div');
    for (const n of nodes) {
      const t = (n.innerText || '').trim();
      if (t === label && n.nextElementSibling) {
        return (n.nextElementSibling.innerText || '').trim();
      }
    }
    return '';
  };
  const comments = Array.from(
    document.querySelectorAll('.comment-item, div[class*="comment-item"]')
  ).slice(0, 20).map((el) => {
    const name = el.querySelector('.name, .author, [class*="name"]');
    const content = el.querySelector('.content, .note-text, [class*="content"]');
    return {
      author: name ? (name.innerText || '').trim() : '',
      content: content ? (content.innerText || '').trim() : '',
    };
  }).filter((c) => c.content);
  return {
    title: txt(['#detail-title', '.note-content .title', '[class*="note-title"]']),
    content: txt(['#detail-desc', '.note-content .desc', '.desc', '[class*="note-text"]']),
    author: txt(['.author-wrapper .username', '.author-container .name', '.username']),
    likes: meta('点赞') || txt(['.like-wrapper .count', '[class*="like"] .count']),
    collects: meta('收藏') || txt(['.collect-wrapper .count', '[class*="collect"] .count']),
    comments_count: meta('评论') || txt(['.chat-wrapper .count', '[class*="chat"] .count']),
    comments,
    url: location.href,
  };
})()
"""


class XhsError(RuntimeError):
    """小红书操作失败。"""


class NotLoggedIn(XhsError):
    """未登录或登录态失效。"""


@dataclass
class XhsNote:
    title: str = ""
    author: str = ""
    likes: str = ""
    url: str = ""
    note_id: str = ""
    cover: str = ""

    def brief(self) -> str:
        return f"{self.title[:40]} | {self.author} | 赞{self.likes or '0'}"


@dataclass
class XhsConfig:
    profile_dir: str = ".agent/browser/xhs"
    headless: bool = True
    channel: str = "msedge"  # 真实 Edge 内核，指纹更正常；失败回退 chromium
    locale: str = "zh-CN"
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1440, "height": 900})
    min_delay: float = 1.2
    max_delay: float = 3.0
    nav_timeout: int = 45000


class XhsSession:
    """一个持久化的小红书浏览器会话。线程内复用，进程退出时 close()。"""

    def __init__(
        self,
        config: Optional[XhsConfig] = None,
        audit_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config or XhsConfig()
        self.audit_fn = audit_fn
        self._pw = None
        self._ctx = None
        self._page = None
        self._login_ok: Optional[bool] = None
        self._login_checked_at: float = 0.0
        self.profile_dir = Path(self.config.profile_dir).expanduser().resolve()

    # ---------- 生命周期 ----------
    def _audit(self, event: Dict[str, Any]) -> None:
        if self.audit_fn is None:
            return
        try:
            self.audit_fn({"kind": "xhs", "ts": datetime.now().isoformat(timespec="seconds"), **event})
        except Exception:
            pass

    def _launch_args(self, offscreen: bool) -> List[str]:
        args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-features=IsolateOrigins,site-per-process",
            "--lang=zh-CN",
        ]
        if offscreen:
            # 有头但移出屏幕：保留真实窗口指纹，又不打扰用户
            args += ["--window-position=-3200,-3200"]
        return args

    def start(self, offscreen: bool = True) -> "XhsSession":
        if self._ctx is not None:
            return self
        from playwright.sync_api import sync_playwright

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()

        last_err: Optional[Exception] = None
        for channel in (self.config.channel, None):
            try:
                kwargs: Dict[str, Any] = dict(
                    headless=self.config.headless,
                    viewport=self.config.viewport,
                    locale=self.config.locale,
                    timezone_id="Asia/Shanghai",
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
                    ),
                    args=self._launch_args(offscreen),
                    ignore_default_args=["--enable-automation"],
                )
                if channel:
                    kwargs["channel"] = channel
                self._ctx = self._pw.chromium.launch_persistent_context(
                    str(self.profile_dir), **kwargs
                )
                self._ctx.add_init_script(STEALTH_JS)
                break
            except Exception as e:  # channel 不可用则回退
                last_err = e
                self._ctx = None
                continue

        if self._ctx is None:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None
            raise XhsError(f"浏览器启动失败：{last_err}")

        pages = self._ctx.pages
        self._page = pages[0] if pages else self._ctx.new_page()
        self._login_ok = None
        self._login_checked_at = 0.0
        return self

    def close(self) -> None:
        for obj, meth in ((self._ctx, "close"), (self._pw, "stop")):
            if obj is not None:
                try:
                    getattr(obj, meth)()
                except Exception:
                    pass
        self._ctx = None
        self._pw = None
        self._page = None
        self._login_ok = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    @property
    def page(self):
        if self._page is None:
            self.start()
        return self._page

    # ---------- 登录 ----------
    def is_logged_in(self, nav: bool = False) -> bool:
        """判断是否已登录。

        不能看 cookie：小红书给游客也会下发 web_session（同名同长度），
        唯一可靠信号是"访问受保护页面是否被踢回 /login"。
        结果缓存 120 秒，避免每次判定都导航。
        """
        now = time.time()
        if self._login_ok is not None and now - self._login_checked_at < 120 and not nav:
            return self._login_ok

        page = self.page
        need_nav = nav or "xiaohongshu.com" not in (page.url or "")
        if need_nav:
            try:
                page.goto(XHS_EXPLORE, wait_until="domcontentloaded", timeout=self.config.nav_timeout)
                page.wait_for_timeout(2500)
            except Exception:
                return False

        url = page.url or ""
        ok = "xiaohongshu.com" in url and "/login" not in url
        self._login_ok = ok
        self._login_checked_at = time.time()
        return ok

    def ensure_login(self, wait_seconds: int = 180) -> bool:
        """检测登录态；未登录时打开窗口等用户扫码，超时返回 False。"""
        if self.is_logged_in(nav=True):
            return True

        page = self.page
        try:
            page.goto(XHS_HOME, wait_until="domcontentloaded", timeout=self.config.nav_timeout)
        except Exception:
            pass

        deadline = time.time() + wait_seconds
        notified = False
        while time.time() < deadline:
            url = page.url or ""
            if "xiaohongshu.com" in url and "/login" not in url:
                # 离开登录页即视为登录成功，再确认一次
                page.wait_for_timeout(1500)
                if "/login" not in (page.url or ""):
                    self._login_ok = True
                    self._login_checked_at = time.time()
                    self._audit({"action": "login", "ok": True})
                    return True
            if not notified:
                print(
                    f"[xhs] 未登录，请在弹出的浏览器窗口中扫码登录（最多等待 {wait_seconds}s）……",
                    flush=True,
                )
                notified = True
            page.wait_for_timeout(2000)

        self._login_ok = False
        self._audit({"action": "login", "ok": False, "reason": "timeout"})
        return False

    def _require_login(self) -> None:
        if not self.is_logged_in():
            raise NotLoggedIn(
                "小红书未登录。请先运行 `python main.py --xhs-login` 扫码登录一次，"
                "登录态会持久化到 " + str(self.profile_dir)
            )

    # ---------- 人类化节奏 ----------
    def _delay(self) -> None:
        time.sleep(random.uniform(self.config.min_delay, self.config.max_delay))

    def _goto(self, url: str) -> None:
        self._require_login()
        self.page.goto(url, wait_until="domcontentloaded", timeout=self.config.nav_timeout)
        # 被踢回登录页说明登录态失效
        if "/login" in (self.page.url or ""):
            self._login_ok = False
            self._login_checked_at = time.time()
            raise NotLoggedIn("登录态已失效，请重新运行 --xhs-login")
        self._login_ok = True
        self._login_checked_at = time.time()
        self._delay()

    def _guard_risk(self) -> None:
        """检测滑块/验证码页，避免模型在风控页上瞎点。"""
        try:
            body = self.page.locator("body").inner_text(timeout=3000)
        except Exception:
            return
        for kw in ("请完成安全验证", "滑动验证", "拖动滑块", "验证码"):
            if kw in body[:800]:
                raise XhsError(f"触发风控验证（{kw}），请人工在浏览器里完成验证后重试")

    # ---------- 搜索 ----------
    def search(self, keyword: str, limit: int = 20) -> List[XhsNote]:
        from urllib.parse import quote

        url = XHS_SEARCH_URL.format(kw=quote(keyword))
        self._goto(url)
        self._guard_risk()

        notes: List[XhsNote] = []
        seen = set()
        for _ in range(3):
            try:
                self.page.wait_for_selector(
                    'section.note-item, div.note-item', timeout=15000
                )
            except Exception:
                pass
            for item in self.page.evaluate(EXTRACT_NOTES_JS) or []:
                key = item.get("note_id") or item.get("url")
                if not key or key in seen:
                    continue
                seen.add(key)
                notes.append(XhsNote(**{k: item.get(k, "") for k in XhsNote.__dataclass_fields__}))
                if len(notes) >= limit:
                    break
            if len(notes) >= limit:
                break
            self.page.mouse.wheel(0, 1200)
            self.page.wait_for_timeout(int(random.uniform(1200, 2200)))

        self._audit({"action": "search", "keyword": keyword, "count": len(notes)})
        return notes

    # ---------- 笔记详情 ----------
    def read_note(self, url: str) -> Dict[str, Any]:
        raw = (url or "").strip()
        if raw.startswith("/"):
            raw = XHS_HOME + raw
        if not raw.startswith("http"):
            raise XhsError(
                "需要完整笔记链接（应直接使用 xhs_search 返回的链接，形如 "
                "https://www.xiaohongshu.com/search_result/<id>?xsec_token=...）"
            )
        self._goto(raw)
        self._guard_risk()

        if "/404" in (self.page.url or "") or "error_code=300031" in (self.page.url or ""):
            raise XhsError(
                "笔记打开失败（300031 当前笔记暂时无法浏览）：常见原因是链接缺少 "
                "xsec_token 或笔记已被删除/设为私密。请使用 xhs_search 返回的完整链接，"
                "不要自己拼接 /explore/<id>。"
            )

        try:
            self.page.wait_for_selector(
                "#detail-title, #detail-desc, .note-content", timeout=20000
            )
        except Exception:
            pass
        data = self.page.evaluate(EXTRACT_DETAIL_JS) or {}
        self._audit({"action": "read_note", "url": raw, "ok": bool(data.get("title") or data.get("content"))})
        return data

    def read_note_by_index(self, index: int) -> Dict[str, Any]:
        """在当前搜索结果页点开第 index 张卡片（1 起）。

        比直接跳转 URL 更稳：xsec_token 由前端自己带，不依赖链接拼接。
        """
        self._require_login()
        page = self.page
        cards = page.locator("section.note-item, div.note-item")
        n = cards.count()
        if n == 0:
            raise XhsError("当前页没有笔记卡片，请先调用 xhs_search")
        idx = max(1, min(int(index), n)) - 1

        before = page.url
        card = cards.nth(idx)
        for sel in ("a[href*='xsec_token']", "a.cover", "a"):
            try:
                link = card.locator(sel).first
                if link.count():
                    link.click()
                    break
            except Exception:
                continue
        else:
            raise XhsError(f"第 {index} 张卡片无法点击")

        try:
            page.wait_for_selector("#detail-title, .note-content", timeout=15000)
        except Exception:
            page.wait_for_timeout(1500)
        self._guard_risk()

        if "/404" in (page.url or ""):
            raise XhsError(f"第 {index} 条笔记已不可见（可能被删除或私密）")

        data = page.evaluate(EXTRACT_DETAIL_JS) or {}
        self._audit({"action": "read_note_by_index", "index": index, "ok": bool(data.get("title") or data.get("content"))})
        data["opened_from"] = before
        return data

    # ---------- 滚动浏览 ----------
    def scroll(self, times: int = 3) -> int:
        self._require_login()
        for _ in range(max(1, min(times, 20))):
            self.page.mouse.wheel(0, random.randint(900, 1500))
            self.page.wait_for_timeout(int(random.uniform(1000, 2000)))
        return times

    # ---------- 当前页状态 ----------
    def state(self) -> Dict[str, Any]:
        page = self.page
        try:
            body = page.locator("body").inner_text(timeout=5000)
        except Exception:
            body = ""
        return {
            "url": page.url,
            "title": page.title(),
            "logged_in": self.is_logged_in(),
            "text_head": body[:1200],
            "text_len": len(body),
        }

    # ---------- 发布（默认只填不提交）----------
    def publish_draft(
        self,
        title: str,
        content: str,
        images: Optional[List[str]] = None,
        submit: bool = False,
    ) -> Dict[str, Any]:
        """打开发布页并填写图文笔记，可选提交。

        实测约束：小红书图文笔记必须**先上传图片**，标题/正文输入框才会渲染。
        没图片时无法填写，这里直接返回明确错误而不是假装成功。
        默认 submit=False：只填不提交，把最终发布权留给人。
        """
        paths = [str(Path(f).expanduser().resolve()) for f in (images or [])]
        paths = [p for p in paths if Path(p).is_file()]
        if not paths:
            return {
                "ok": False,
                "filled_title": False,
                "filled_body": False,
                "uploaded": [],
                "submitted": False,
                "url": "",
                "error": (
                    "图文笔记必须先上传至少一张本地图片（png/jpg/webp，≤32MB）。"
                    "请把图片路径放进 images 参数后重试。"
                ),
            }

        self._goto(XHS_CREATOR)
        self._guard_risk()
        page = self.page
        page.wait_for_timeout(4000)

        # 页面默认停在"上传视频"，先切到"上传图文"
        switched = page.evaluate(
            """() => {
              const els = Array.from(document.querySelectorAll('div,span,button,li,a'));
              for (const el of els) {
                if ((el.innerText || '').trim() === '上传图文') {
                  const r = el.getBoundingClientRect();
                  if (r.width > 0 && r.height > 0) { el.click(); return true; }
                }
              }
              return false;
            }"""
        )
        if switched:
            page.wait_for_timeout(3000)

        # 上传图片：上传前不能 wait_for_timeout 太久，需要等缩略图/输入框出现
        uploaded: List[str] = []
        file_input = page.locator('input[type="file"]').first
        for p in paths[:9]:
            try:
                file_input.set_input_files(p)
                uploaded.append(p)
                page.wait_for_timeout(3500)
            except Exception:
                continue

        if not uploaded:
            return {
                "ok": False,
                "filled_title": False,
                "filled_body": False,
                "uploaded": [],
                "submitted": False,
                "url": page.url,
                "error": "图片上传失败（未找到 input[type=file] 或格式不被接受）",
            }

        # 等编辑区渲染（标题框出现即表示图片已就绪）
        title_box = None
        for sel in ('input[placeholder*="标题"]', 'input[placeholder*="填写标题"]', ".d-input input"):
            try:
                for _ in range(10):
                    loc = page.locator(sel).first
                    if loc.count() and loc.is_visible():
                        title_box = loc
                        break
                    page.wait_for_timeout(1000)
                if title_box is not None:
                    break
            except Exception:
                continue

        filled_title = False
        if title_box is not None:
            try:
                title_box.click()
                title_box.fill(title[:20])
                filled_title = True
            except Exception:
                pass

        filled_body = False
        for sel in ('div[contenteditable="true"]', ".ql-editor", "#post-textarea"):
            try:
                box = page.locator(sel).first
                if box.count() and box.is_visible():
                    box.click()
                    page.keyboard.insert_text(content)
                    filled_body = True
                    break
            except Exception:
                continue

        submitted = False
        if submit and filled_body:
            for sel in ('button:has-text("发布")', ".publish-btn", 'button[class*="publish"]'):
                try:
                    btn = page.locator(sel).last
                    if btn.count() and btn.is_visible() and btn.is_enabled():
                        btn.click()
                        submitted = True
                        page.wait_for_timeout(5000)
                        break
                except Exception:
                    continue

        result = {
            "ok": filled_title and filled_body,
            "filled_title": filled_title,
            "filled_body": filled_body,
            "uploaded": uploaded,
            "submitted": submitted,
            "url": page.url,
            "error": None if (filled_title and filled_body) else "标题或正文未能填入（页面结构可能已改版，请跑 xhs_probe.py）",
            "note": "默认只填草稿不提交；需要自动发布请显式传 submit=True",
        }
        self._audit({"action": "publish_draft", **{k: v for k, v in result.items() if k not in ("note",)}})
        return result


# ---------- 进程级单例 + 线程隔离 ----------
#
# 关键约束：Playwright Sync API 在同一线程里只允许存在一个实例。
# 网页模型客户端（ZhipuWebClient 等）已经占用了主线程的 sync_playwright，
# 此时若在主线程再起第二个，会直接抛
#   "It looks like you are using Playwright Sync API inside the asyncio loop."
# 因此小红书会话必须跑在专属工作线程里，所有调用通过队列串行提交。

_WORKER: Optional["XhsWorker"] = None


class XhsWorker:
    """在独立线程中持有 XhsSession，对外提供同步调用。"""

    _METHODS = {
        "is_logged_in",
        "ensure_login",
        "search",
        "read_note",
        "read_note_by_index",
        "scroll",
        "state",
        "publish_draft",
    }

    def __init__(
        self,
        config: Optional[XhsConfig] = None,
        audit_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        offscreen: bool = True,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.config = config or XhsConfig()
        self.audit_fn = audit_fn
        self.offscreen = offscreen
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xhs-browser")
        self._session: Optional[XhsSession] = None

    # 只在工作线程内执行
    def _ensure(self) -> XhsSession:
        if self._session is None:
            session = XhsSession(config=self.config, audit_fn=self.audit_fn)
            session.start(offscreen=self.offscreen)
            self._session = session
        return self._session

    def _invoke(self, method: str, args: tuple, kwargs: Dict[str, Any]) -> Any:
        session = self._ensure()
        return getattr(session, method)(*args, **kwargs)

    def call(self, method: str, *args: Any, timeout: int = 240, **kwargs: Any) -> Any:
        if method not in self._METHODS:
            raise AttributeError(f"不支持的小红书操作：{method}")
        future = self._exec.submit(self._invoke, method, args, kwargs)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        if self._exec is None:
            return
        try:
            self._exec.submit(self._close).result(timeout=40)
        except Exception:
            pass
        self._exec.shutdown(wait=False)
        self._exec = None  # type: ignore[assignment]

    def _close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


class _SessionProxy:
    """把 XhsWorker 包装成看起来像 XhsSession 的对象。"""

    def __init__(self, worker: XhsWorker) -> None:
        object.__setattr__(self, "_worker", worker)

    def __getattr__(self, name: str) -> Any:
        if name not in XhsWorker._METHODS:
            raise AttributeError(name)

        def fn(*args: Any, **kwargs: Any) -> Any:
            return self._worker.call(name, *args, **kwargs)

        return fn

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("代理对象不支持赋值")

    @property
    def profile_dir(self) -> Path:
        return Path(self._worker.config.profile_dir).expanduser().resolve()


def get_session(
    config: Optional[XhsConfig] = None,
    audit_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    offscreen: bool = True,
) -> Any:
    global _WORKER
    if _WORKER is None:
        _WORKER = XhsWorker(config=config, audit_fn=audit_fn, offscreen=offscreen)
    return _SessionProxy(_WORKER)


def close_session() -> None:
    global _WORKER
    if _WORKER is not None:
        _WORKER.shutdown()
        _WORKER = None
