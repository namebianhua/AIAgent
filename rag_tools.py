"""RAG 工具集（第三阶段）：把知识库检索注册成 Agent 的标准工具。

和第二阶段的「被动注入」的区别：
- 被动注入：任务一开始就猜你要什么，把资料塞进系统提示词（省一轮调用，但可能塞错）；
- 主动检索：Agent 自己判断「这块我不知道」，然后调 rag_search 去查（准，但多一轮）。
两者共存：注入负责常见背景，工具负责临场补查。

工具清单：
- rag_search   语义检索知识库（Agent 的主要入口）
- rag_ingest   把文件/目录入库（dangerous + require_confirm，会写向量库）
- rag_status   知识库健康状态与统计
- exp_recall   检索历史任务经验（语义版，和 task_memory 的关键词召回互补）

所有工具在 RAG 不可用（chromadb 未装 / Ollama 未起）时返回 ok=True + 明确说明，
而不是报错——让模型知道「这条路不通，换别的办法」，比让它反复重试更省事。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from engine import ToolRegistry
from sandbox import Sandbox

MAX_OUTPUT = 6000


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[内容过长已截断，原始 {len(text)} 字符]"


def _degraded(reason: str) -> Dict[str, Any]:
    """知识库不可用时的统一返回：ok=True，让模型换方案而不是重试。"""
    return {
        "ok": True,
        "output": f"知识库当前不可用（{reason}）。"
                  f"请改用其他方式完成任务：读工作区文件、search_text 正则搜索、"
                  f"或 search 联网查询，并在答复里说明未能使用本地知识库。",
        "error": None,
    }


def _format_hits(hits: List[Dict[str, Any]], query: str) -> str:
    if not hits:
        return (
            f"知识库中没有与「{query}」相关的内容。\n"
            f"可能原因：文档尚未入库，或问法与原文差异较大。\n"
            f"建议：换关键词重试一次；仍无结果就改用 search_text（工作区正则搜索）"
            f"或 search（联网）。不要反复用同一个词查询。"
        )
    lines = [f"检索「{query}」命中 {len(hits)} 条："]
    for i, h in enumerate(hits, 1):
        src = h.get("source") or h.get("metadata", {}).get("source_file") or "未知来源"
        score = h.get("relevance_score", 0)
        lines.append(f"\n--- [{i}] 相关度 {score:.2f} | 来源：{src} ---")
        lines.append((h.get("content") or "").strip())
    lines.append("\n（以上为知识库原文片段，回答时请据此作答并说明出处；"
                 "若片段不足以支撑结论，请明确说明缺口，不要补写。）")
    return _truncate("\n".join(lines))


def register_rag_tools(
    reg: ToolRegistry,
    engine: Any,
    *,
    sandbox: Optional[Sandbox] = None,
    experience: Any = None,
    default_collection: str = "agent_general_knowledge",
    experience_collection: str = "agent_experience",
) -> List[str]:
    """把 RAG 相关工具挂进注册表，返回实际注册成功的工具名。

    engine 为 None 时注册「降级版」工具（返回不可用说明），保证模型看到的工具清单稳定。
    """
    added: List[str] = []

    @reg.register(name="rag_search")
    def rag_search(query: str, top_k: int = 3, collection: str = "") -> Dict[str, Any]:
        """在本地知识库中做语义检索，用于回答涉及项目文档/资料/规范的问题。

        什么时候用：任务需要的信息不在当前对话里、也不在工作区文件中，
        而可能来自此前入库的文档时（例如「我们的部署规范是什么」）。
        什么时候不用：能直接 read_file / search_text 拿到的内容就别绕一圈。

        query: 自然语言问题或关键词（中文可用，语义匹配不要求字面相同）
        top_k: 返回条数（1-8）
        collection: 知识库名，留空用默认库；填 "all" 表示知识库+经验库一起查
        """
        if engine is None:
            return _degraded("未初始化")
        q = (query or "").strip()
        if not q:
            return {"ok": False, "output": "", "error": "query 不能为空"}
        k = max(1, min(int(top_k), 8))
        coll = (collection or "").strip()
        try:
            if coll.lower() == "all":
                hits = engine.search_multi(
                    q, [default_collection, experience_collection], top_k=k, min_score=None
                )
            else:
                hits = engine.search(
                    q, top_k=k, collection_name=coll or default_collection, min_score=None
                )
        except Exception as e:
            return _degraded(f"{type(e).__name__}: {e}")
        return {
            "ok": True,
            "output": _format_hits(hits, q),
            "error": None,
            "hit_count": len(hits),
            "top_score": hits[0].get("relevance_score", 0) if hits else 0,
        }

    added.append("rag_search")

    @reg.register(name="rag_ingest", dangerous=True, require_confirm=True)
    def rag_ingest(path: str, collection: str = "", recursive: bool = True) -> Dict[str, Any]:
        """把文件或目录解析后存入本地知识库（需要 Ollama 向量化，耗时较长）。

        path: 文件路径或目录路径（相对工作区或绝对路径均可）
        collection: 目标知识库名，留空用默认库
        recursive: 目录时是否递归子目录
        """
        if engine is None:
            return _degraded("未初始化")
        target = sandbox.resolve(path) if sandbox is not None else Path(path).expanduser()
        coll = (collection or "").strip() or default_collection
        try:
            if target.is_dir():
                res = engine.add_directory(target, collection_name=coll, recursive=recursive)
            elif target.is_file():
                res = engine.add_document(target, collection_name=coll)
            else:
                return {"ok": False, "output": "", "error": f"路径不存在：{path}"}
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}

        lines = [f"入库完成：{target}", f"知识库：{coll}"]
        if res.get("status") == "unchanged":
            lines.append("内容未变化，已跳过（知识库中已有该版本）。")
        else:
            lines.append(f"新增 {res.get('added', 1)} 个文档，共 {res.get('chunks', res.get('chunk_count', 0))} 个文本块")
            if res.get("unchanged"):
                lines.append(f"其中 {res['unchanged']} 个文档内容未变，已跳过")
        for err in (res.get("errors") or [])[:10]:
            lines.append(f"  失败：{err.get('file')} — {err.get('reason')}")
        if res.get("scanned") is not None:
            lines.append(f"扫描到 {res['scanned']} 个受支持的文件")
        return {"ok": True, "output": _truncate("\n".join(lines)), "error": None}

    added.append("rag_ingest")

    @reg.register(name="rag_status")
    def rag_status() -> Dict[str, Any]:
        """查看本地知识库状态：有哪些库、各存了多少文档、Ollama 向量化服务是否正常。"""
        if engine is None:
            return _degraded("未初始化")
        try:
            st = engine.stats()
        except Exception as e:
            return _degraded(f"{type(e).__name__}: {e}")
        lines = [
            f"向量库目录：{st.get('db_path')}",
            f"embedding 模型：{st.get('embed_model')} @ {st.get('ollama_url')}",
            f"分块设置：{st.get('chunk_size')} 字符 / 重叠 {st.get('chunk_overlap')}",
        ]
        colls = st.get("collections") or []
        if not colls:
            lines.append("（还没有任何知识库，用 rag_ingest 入库一个文档）")
        else:
            lines.append(f"知识库 {len(colls)} 个：")
            for c in colls:
                lines.append(
                    f"  - {c.get('name')}: {c.get('chunks')} 块 / {c.get('sources')} 个来源"
                    f"（模型 {c.get('embed_model') or '-'}，维度 {c.get('dim') or '-'}）"
                )
        if experience is not None:
            try:
                es = experience.stats()
                lines.append(
                    f"经验库：{es.get('experiences', 0)} 条历史任务经验"
                    if es.get("ok") else f"经验库：不可用（{es.get('reason')}）"
                )
            except Exception:
                pass
        return {"ok": True, "output": _truncate("\n".join(lines)), "error": None}

    added.append("rag_status")

    if experience is not None:

        @reg.register(name="exp_recall")
        def exp_recall(goal: str, top_k: int = 3) -> Dict[str, Any]:
            """检索历史任务经验（语义匹配）：以前做过类似的事吗？当时怎么做的、踩了什么坑？

            什么时候用：接到一个不熟悉的任务、或前几步反复失败时，先查历史经验往往比硬试更快。

            goal: 当前任务或当前卡住的问题描述
            top_k: 返回条数（1-5）
            """
            try:
                hits = experience.recall((goal or "").strip(), k=max(1, min(int(top_k), 5)))
            except Exception as e:
                return _degraded(f"{type(e).__name__}: {e}")
            if not hits:
                return {
                    "ok": True,
                    "output": f"没有找到与「{(goal or '').strip()[:80]}」语义相近的历史任务经验。"
                              f"按常规方式继续即可，不必重复查询。",
                    "error": None,
                }
            body = "\n\n".join(h.to_card(i) for i, h in enumerate(hits, 1))
            return {
                "ok": True,
                "output": _truncate(
                    f"命中 {len(hits)} 条历史经验（供参考，不是必须照做）：\n\n{body}"
                ),
                "error": None,
            }

        added.append("exp_recall")

    return added


# ---------- 第三阶段：工具使用引导（写进系统提示词）----------

RAG_TOOL_GUIDANCE = """

【知识库与经验库】
- rag_search：需要项目文档、内部规范、既有资料时先查知识库，不要凭空作答。
  查不到就换一次说法；两次都无结果说明库里没有，改用文件读取或联网搜索，并如实说明。
- exp_recall：接到不熟的任务或连续失败时，先查有没有类似的历史经验（含当时踩的坑）。
- rag_ingest：只有用户明确要求「把这个文件/目录加进知识库」时才用（属高危操作，会弹确认）。
- 引用知识库内容时说明来源文件名；片段不足以支撑结论时指出缺口，禁止编造原文没有的细节。"""
