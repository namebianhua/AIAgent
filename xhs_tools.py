"""小红书工具集（T2）：把 XhsSession 暴露成模型可调用的工具。

安全约定
--------
- 只读工具（状态/搜索/读笔记）无需确认；
- 有副作用的工具（发布、上传）标记 require_confirm，必须人工放行；
- 不提供"执行任意 JS / 任意点击坐标"这类逃逸工具 —— 模型只能做业务动作，
  不能借浏览器执行任意代码或跳出小红书域名。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import xhs_web
from engine import ToolRegistry

MODULE_STATE: Dict[str, Any] = {
    "profile_dir": ".agent/browser/xhs",
    "headless": True,
    "offscreen": True,
    "audit": None,
}


def configure(
    profile_dir: Optional[str] = None,
    headless: Optional[bool] = None,
    offscreen: Optional[bool] = None,
    audit=None,
) -> None:
    if profile_dir:
        MODULE_STATE["profile_dir"] = profile_dir
    if headless is not None:
        MODULE_STATE["headless"] = headless
    if offscreen is not None:
        MODULE_STATE["offscreen"] = offscreen
    if audit is not None:
        MODULE_STATE["audit"] = audit


def _session() -> xhs_web.XhsSession:
    return xhs_web.get_session(
        config=xhs_web.XhsConfig(
            profile_dir=MODULE_STATE["profile_dir"],
            headless=bool(MODULE_STATE["headless"]),
        ),
        audit_fn=MODULE_STATE["audit"],
        offscreen=bool(MODULE_STATE["offscreen"]),
    )


def _fmt_notes(notes: List[xhs_web.XhsNote]) -> str:
    lines = []
    for i, n in enumerate(notes, 1):
        title = n.title.replace("\n", " ").strip() or "（无标题）"
        likes = n.likes or "0"
        lines.append(f"{i}. {title[:60]}")
        lines.append(f"   作者：{n.author or '-'}  赞：{likes}")
        lines.append(f"   链接：{n.url}")
    return "\n".join(lines) if lines else "（没有解析到笔记卡片）"


def register_xhs_tools(reg: ToolRegistry) -> ToolRegistry:
    """把小红书工具注册进已有注册表。"""

    @reg.register()
    def xhs_status() -> Dict[str, Any]:
        """查看小红书登录状态与当前页面内容。任务开始或不确定页面时先调用它。"""
        s = _session()
        logged = s.is_logged_in()
        if not logged:
            return {
                "ok": False,
                "output": (
                    "未登录小红书。需要用户先运行 `python main.py --xhs-login` "
                    "在弹出窗口里扫码登录一次（登录态会持久保存）。"
                ),
                "error": "not_logged_in",
            }
        st = s.state()
        return {
            "ok": True,
            "output": (
                f"已登录。\n页面标题：{st['title']}\nURL：{st['url']}\n"
                f"可见文本（前 800 字）：\n{st['text_head'][:800]}"
            ),
            "error": None,
            "state": st,
        }

    @reg.register()
    def xhs_search(keyword: str, limit: int = 10) -> Dict[str, Any]:
        """在小红书搜索笔记，返回标题/作者/点赞/链接列表。

        keyword: 搜索关键词
        limit: 最多返回条数（1-30）
        """
        notes = _session().search(keyword, limit=max(1, min(int(limit), 30)))
        return {
            "ok": bool(notes),
            "output": f"关键词「{keyword}」共取到 {len(notes)} 条笔记：\n{_fmt_notes(notes)}",
            "error": None if notes else "no_results",
            "notes": [n.__dict__ for n in notes],
        }

    @reg.register()
    def xhs_read_note(url: str) -> Dict[str, Any]:
        """打开一条笔记，返回标题、作者、正文和前若干条评论。

        url: 笔记完整链接（必须是 https://www.xiaohongshu.com/explore/... 形式，
             通常直接来自 xhs_search 的结果，务必原样传入，不要自己拼接）
        """
        data = _session().read_note(url)
        content = (data.get("content") or "").strip()
        if len(content) > 2500:
            content = content[:2500] + "\n...[正文已截断]"
        comments = data.get("comments") or []
        clines = [
            f"- {c.get('author', '-')}：{c.get('content', '')[:120]}" for c in comments[:10]
        ]
        out = (
            f"标题：{data.get('title') or '（无）'}\n"
            f"作者：{data.get('author') or '-'}\n"
            f"赞 {data.get('likes') or '-'} / 收藏 {data.get('collects') or '-'} / "
            f"评论 {data.get('comments_count') or '-'}\n"
            f"正文：\n{content or '（未解析到正文）'}"
        )
        if clines:
            out += "\n热评：\n" + "\n".join(clines)
        return {
            "ok": bool(data.get("title") or content),
            "output": out,
            "error": None if (data.get("title") or content) else "empty_note",
            "note": data,
        }

    @reg.register()
    def xhs_scroll(times: int = 3) -> Dict[str, Any]:
        """在当前页面向下滚动，用于加载更多笔记。times 为滚动次数（1-20）。"""
        n = _session().scroll(times=max(1, min(int(times), 20)))
        return {"ok": True, "output": f"已向下滚动 {n} 次", "error": None}

    @reg.register(dangerous=True, require_confirm=True)
    def xhs_publish(
        title: str,
        content: str,
        images: str = "",
        submit: bool = False,
    ) -> Dict[str, Any]:
        """在小红书创作中心填写图文笔记草稿（危险操作，需人工确认）。

        title: 标题（最多 20 字）
        content: 正文
        images: 本地图片路径，多个用英文逗号分隔；为空则不上传图片
        submit: 是否点击"发布"。默认 false 只填草稿，把最终发布权交给用户
        """
        paths = [p.strip() for p in images.split(",") if p.strip()] if images else []
        result = _session().publish_draft(
            title=title, content=content, images=paths, submit=bool(submit)
        )
        lines = [
            f"标题：{'已填' if result['filled_title'] else '未填'}",
            f"正文：{'已填' if result['filled_body'] else '未填'}",
            f"图片：{len(result['uploaded'])} 张",
            f"发布：{'已提交' if result['submitted'] else '未提交（只在页面上填好）'}",
        ]
        if result.get("error"):
            lines.append(f"错误：{result['error']}")
        lines.append(f"当前页：{result['url'] or '-'}")
        return {
            "ok": bool(result.get("ok")),
            "output": "\n".join(lines),
            "error": result.get("error"),
            "result": result,
        }

    return reg


def close() -> None:
    xhs_web.close_session()
