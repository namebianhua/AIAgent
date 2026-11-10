from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # 仅用于类型标注，避免 import 本模块时强依赖 playwright
    from playwright.sync_api import Page


@dataclass
class LocatorRule:
    """一条定位规则"""
    selector: str
    is_text: bool = False
    must_visible: bool = False


@dataclass
class PageLocatorConfig:
    """某个网页模型的定位配置"""
    name: str = ""
    url: str = ""
    input_selectors: List[LocatorRule] = field(default_factory=list)
    send_selectors: List[LocatorRule] = field(default_factory=list)
    answer_selectors: List[LocatorRule] = field(default_factory=list)
    # 命中这些祖先节点的块会被丢弃（例如思维链/"思考中"面板，不能当成最终回答）
    exclude_ancestors: List[str] = field(
        default_factory=lambda: ["[class*='thinking']", "[class*='reasoning']"]
    )
    min_answer_length: int = 2


# 默认通用配置，适合多数聊天页面
DEFAULT_LOCATOR = PageLocatorConfig(
    name="default",
    input_selectors=[
        LocatorRule("textarea"),
        LocatorRule("[contenteditable='true']"),
        LocatorRule("input[type='text']"),
    ],
    send_selectors=[
        LocatorRule("button:has-text('发送')", is_text=True),
        LocatorRule("button[aria-label*='发送']"),
        LocatorRule("button[class*='send']"),
    ],
    answer_selectors=[
        LocatorRule(".markdown-body"),
        LocatorRule("[class*='answer-content']"),
        LocatorRule("[class*='message-content']"),
        LocatorRule("[class*='bubble']"),
    ],
)

# 智谱清言 chatglm.cn（已用真实页面验证）
# 真实 DOM：回答在 .answer-content 里的 .markdown-body；思维链在同级 thinking 容器中
ZHIPU_LOCATOR = PageLocatorConfig(
    name="zhipu-web",
    url="https://chatglm.cn/",
    input_selectors=[
        LocatorRule("textarea"),
        LocatorRule("[contenteditable='true']"),
    ],
    send_selectors=[
        LocatorRule("button[aria-label*='发送']"),
        LocatorRule("button:has-text('发送')", is_text=True),
    ],
    answer_selectors=[
        LocatorRule(".answer-content .markdown-body"),
        LocatorRule(".markdown-body.md-body"),
        LocatorRule(".answer-content"),
    ],
)

DEEPSEEK_LOCATOR = PageLocatorConfig(
    name="deepseek-web",
    url="https://chat.deepseek.com/",
    input_selectors=[
        LocatorRule("textarea"),
        LocatorRule("[contenteditable='true']"),
    ],
    send_selectors=[
        LocatorRule("button[aria-label*='发送']"),
        LocatorRule("button:has-text('发送')", is_text=True),
    ],
    answer_selectors=[
        LocatorRule(".answer-content .markdown-body"),
        LocatorRule(".markdown-body.md-body"),
        LocatorRule(".answer-content"),
    ],
)

# 百度文心一言（yiyan.baidu.com 会跳到 wenxin.baidu.com）
WENXIN_LOCATOR = PageLocatorConfig(
    name="wenxin-web",
    url="https://yiyan.baidu.com/",
    input_selectors=[
        LocatorRule("textarea"),
        LocatorRule("[contenteditable='true']"),
    ],
    send_selectors=[
        LocatorRule("button[aria-label*='发送']"),
        LocatorRule("button:has-text('发送')", is_text=True),
    ],
    answer_selectors=[
        LocatorRule("[class*='markdown']"),
        LocatorRule("[class*='answer-content']"),
        LocatorRule("[class*='answer']"),
    ],
)

# 千问网页版 www.qianwen.com（已用真实页面验证：回答在 [class*=markdown] / [class*=answer]）
QWEN_LOCATOR = PageLocatorConfig(
    name="qwen-web",
    url="https://www.qianwen.com/",
    input_selectors=[
        LocatorRule("[contenteditable='true']"),
        LocatorRule("textarea"),
    ],
    send_selectors=[
        LocatorRule("button[aria-label*='发送']"),
        LocatorRule("button:has-text('发送')", is_text=True),
    ],
    answer_selectors=[
        LocatorRule("[class*='markdown']"),
        LocatorRule("[class*='answer']"),
    ],
)


class PageLocator:
    """页面元素定位器"""

    def __init__(self, config: Optional[PageLocatorConfig] = None):
        self.config = config or DEFAULT_LOCATOR

    def _first_usable(self, page: "Page", rules: List[LocatorRule], need_enabled: bool = False):
        """按配置顺序找第一个“可见（且可用）”的元素，找不到返回 None。

        很多聊天页会保留一个隐藏的 textarea 给输入法用，所以不能只看 attached。
        """
        for rule in rules:
            try:
                loc = page.locator(rule.selector)
                count = loc.count()
            except Exception:
                continue
            for i in range(min(count, 5)):
                cand = loc.nth(i)
                try:
                    if not cand.is_visible():
                        continue
                    if need_enabled and not cand.is_enabled():
                        continue
                    return cand
                except Exception:
                    continue
        return None

    def find_input(self, page: "Page", timeout: int = 10000):
        # 先等页面出现任意候选输入框，再挑可见的那个
        try:
            page.wait_for_selector(
                ", ".join(r.selector for r in self.config.input_selectors),
                state="visible",
                timeout=timeout,
            )
        except Exception:
            pass

        loc = self._first_usable(page, self.config.input_selectors)
        if loc is not None:
            return loc

        # 兜底：实在都不“可见”就直接用第一个 attached 的，至少不会直接崩
        for rule in self.config.input_selectors:
            try:
                cand = page.locator(rule.selector).first
                cand.wait_for(state="attached", timeout=2000)
                return cand
            except Exception:
                continue
        raise RuntimeError(f"{self.config.name} 未找到可用输入框")

    def find_send_button(self, page: "Page", timeout: int = 5000):
        """返回可点击的发送按钮；没有就返回 None（调用方回退到回车发送）。"""
        return self._first_usable(page, self.config.send_selectors, need_enabled=True)

    def _texts(self, page: "Page", selector: str) -> List[str]:
        """返回每个匹配块的文本；思维链/思考面板用空串占位（保持下标稳定）。"""
        try:
            blocks = page.locator(selector).all()
        except Exception:
            return []
        out = []
        for b in blocks:
            try:
                text = (b.inner_text(timeout=1000) or "").strip()
            except Exception:
                out.append("")
                continue
            if text and self.config.exclude_ancestors:
                try:
                    excluded = b.evaluate(
                        """(el, sels) => {
                            if (!el.closest) return false;
                            return sels.some(s => { try { return !!el.closest(s); }
                                                    catch (e) { return false; } });
                        }""",
                        self.config.exclude_ancestors,
                    )
                except Exception:
                    excluded = False
                if excluded:
                    out.append("")
                    continue
            out.append(text)
        return out

    def _pick(self, page: "Page", selector: str, baseline: dict, question: Optional[str]) -> Optional[str]:
        """在指定选择器下取“发送之后新增的最后一个有效回答块”。

        用“文本多重集去重”而不是下标跳过，这样即使页面重渲染导致块数量变化也不会错位。
        """
        q = (question or "").strip()
        texts = self._texts(page, selector)
        old = Counter(baseline.get(selector + "::texts", []))

        for i in range(len(texts) - 1, -1, -1):
            text = texts[i]
            if not text or len(text) < self.config.min_answer_length:
                continue
            if text == q:
                continue
            if old[text] > 0:
                old[text] -= 1  # 这是历史回答，跳过
                continue
            return text
        return None

    def snapshot(self, page: "Page") -> dict:
        """记录发送前的回答块基线，用于区分“本轮新回答”和“历史回答”。"""
        baseline = {}
        for rule in self.config.answer_selectors:
            texts = self._texts(page, rule.selector)
            baseline[rule.selector] = len(texts)
            baseline[rule.selector + "::texts"] = texts
        return baseline

    def find_answer(
        self,
        page: "Page",
        timeout: int = 120,
        question: Optional[str] = None,
        baseline: Optional[dict] = None,
        poll_ms: int = 800,
        stable_polls: int = 3,
        rule_grace: float = 6.0,
    ) -> str:
        """等待并返回本轮回答。

        - baseline 由调用方在“发送前”通过 snapshot() 采集；
        - 就绪的选择器会被锁定：回答区往往先渲染标题（如 "ChatGLM"）、正文稍后才出现，
          若一有东西就降级到宽松选择器，就会把标题当成答案；
        - 低优先级选择器需要等 rule_grace 秒的宽限期才参与竞争；
        - 要求文本连续 stable_polls 次采集不变，避免返回流式输出到一半的内容。
        """
        baseline = baseline if baseline is not None else {}

        start = time.time()
        deadline = start + timeout
        chosen: Optional[str] = None
        last_text: Optional[str] = None
        stable = 0
        misses = 0

        while time.time() < deadline:
            try:
                page.wait_for_timeout(poll_ms)
            except Exception:
                # 页面跳转/刷新时可能抛错，稍后重试
                pass

            if chosen is None:
                elapsed = time.time() - start
                for idx, rule in enumerate(self.config.answer_selectors):
                    if elapsed < idx * rule_grace:
                        break  # 还没轮到这条宽松规则
                    try:
                        count = page.locator(rule.selector).count()
                    except Exception:
                        count = 0
                    if count > 0:
                        chosen = rule.selector
                        break
                if chosen is None:
                    continue

            candidate = self._pick(page, chosen, baseline, question)
            if candidate:
                misses = 0
                if candidate == last_text:
                    stable += 1
                    if stable >= stable_polls:
                        return candidate
                else:
                    last_text, stable = candidate, 1
            else:
                # 锁定的选择器长时间没内容（页面重渲染/切到别的会话），允许重新选择
                misses += 1
                if misses * poll_ms >= 8000:
                    chosen, misses = None, 0

        if last_text:
            # 超时但已经拿到内容，宁可返回半截也不完全失败
            return last_text
        raise TimeoutError(f"{self.config.name} 未在超时时间内返回可见回答")