# test_rag.py
"""RAG 知识库四阶段端到端验证。

用临时目录建向量库，跑完即弃，不污染项目的 agent_knowledge_base。
需要本机 Ollama 已拉取默认 embedding 模型（见 rag_engine.DEFAULT_EMBED_MODEL）。
运行：.venv\\Scripts\\python.exe test_rag.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from rag_engine import DEFAULT_EMBED_MODEL, RagConfig, RagEngine
from experience import ExperienceStore
from engine import ToolRegistry
from rag_tools import register_rag_tools


def _ok(label: str, cond: bool, extra: str = "") -> bool:
    print(f"{'✓' if cond else '✗'} {label}{(' — ' + extra) if extra else ''}")
    return cond


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_test_"))
    print(f"临时向量库：{tmp}\n")

    # ---------- 探活 ----------
    engine = RagEngine(RagConfig(db_path=str(tmp)))
    probe = engine.probe()
    if not _ok("阶段0 Ollama 向量化服务", probe.get("ok", False),
               probe.get("reason") or f"dim={probe.get('dim')}"):
        print(f"\nOllama 未就绪，后续阶段无法验证。请先：ollama pull {DEFAULT_EMBED_MODEL}")
        return 1

    # ---------- 分块质量（结构感知分块，纯本地，不调 embedding）----------
    print("\n== 分块质量：Markdown 结构感知 ==")
    md = (
        "# 盘后报告\n\n"
        "## 4. 十五维评分表\n\n"
        "| # | 维度 | 得分 | 依据 |\n"
        "|---|---|---|---|\n"
        + "".join(f"| {i} | 维度{i} | {i}.5 | 依据说明{i} |\n" for i in range(1, 16))
        + "\n**加权总分 = 35.5 / 100**\n\n"
        "## 5. RSG 三层门控结论\n\n"
        "- **R = 下跌末段**：连续缩量阴跌。\n"
        "- **S = 中性偏空**：一致性 空 7/10。\n"
        "- **G = 收窄·冻结**：两条成立一条半成立。\n"
    )
    chunks = engine._chunk_text(md)
    _ok("按标题分节（块数 > 1）", len(chunks) > 1, f"{len(chunks)} 块")
    _ok("每块带标题路径前缀", all(c.startswith("【") for c in chunks),
        chunks[0][:28] if chunks else "")

    rsg = [c for c in chunks if "RSG" in c and "R = 下跌末段" in c]
    _ok("RSG 三条结论落在同一块（未被跨节切碎）", bool(rsg),
        f"{len(rsg)} 块" if rsg else "被切散了")
    if rsg:
        has_all = all(k in rsg[0] for k in ("R = 下跌末段", "S = 中性偏空", "G = 收窄"))
        _ok("该块同时含 R/S/G 三条", has_all)

    score_chunks = [c for c in chunks if "加权总分" in c]
    _ok("加权总分与所属标题同块", bool(score_chunks) and "十五维评分表" in score_chunks[0])

    tbl = [c for c in chunks if "| 15 | 维度15" in c]
    _ok("长表格切片重复表头（脱离表头即噪音）",
        bool(tbl) and "| # | 维度 |" in tbl[0])

    # overlap 会把上一块尾部拼到本块开头，导致块以「上节残渣」开头、
    # 或一块里出现两个标题前缀。结构分块必须两者都没有。
    _ok("结构分块不做 overlap（无块以非标题开头）",
        all(c.lstrip().startswith("【") for c in chunks))
    _ok("无块内混入第二个标题前缀（overlap 的典型痕迹）",
        all(c.count("【") == c.count("】") and "】\n【" not in c for c in chunks))

    plain = engine._chunk_text("第一段没有标题。\n" + ("很长的句子。" * 200))
    _ok("无结构文本仍走段落聚合（回归保护）", len(plain) >= 1, f"{len(plain)} 块")

    # ---------- 第一阶段：入库 + 检索 ----------
    print("\n== 第一阶段：文档入库与相似度检索 ==")
    doc = tmp / "arch.txt"
    doc.write_text(
        "Agent 系统的架构设计包含三个核心层：\n"
        "1. 规划层：负责任务拆解与步骤编排。\n"
        "2. 执行层：负责调用工具完成每一步。\n"
        "3. 安全层：负责任务监控、人工确认与失败回滚。\n"
        "知识库使用 ChromaDB 做本地向量存储，用 Ollama 做文本向量化。\n",
        encoding="utf-8",
    )
    res = engine.add_document(str(doc))
    _ok("文档入库", res.get("status") == "success", f"{res.get('chunk_count')} 块")

    res2 = engine.add_document(str(doc))
    _ok("幂等入库（内容未变应跳过）", res2.get("status") == "unchanged", res2.get("status", ""))

    hits = engine.search("系统分成哪几层？", top_k=2)
    _ok("相似度检索命中", len(hits) > 0,
        f"top={hits[0]['relevance_score']:.2f}" if hits else "无结果")
    if hits:
        print("   命中片段：" + hits[0]["content"].strip().replace("\n", " ")[:80])

    # ---------- 第二阶段：知识注入上下文渲染 ----------
    print("\n== 第二阶段：检索结果注入（上下文渲染）==")
    ctx = engine.format_context(hits, max_chars=1200)
    _ok("上下文块渲染", "相关度" in ctx and "来源" in ctx, f"{len(ctx)} 字符")

    # ---------- 第三阶段：工具注册 + 调用 ----------
    print("\n== 第三阶段：rag_search 注册为标准工具 ==")
    reg = ToolRegistry()
    store = ExperienceStore(engine=engine)
    added = register_rag_tools(reg, engine, sandbox=None, experience=store)
    _ok("工具已注册", "rag_search" in reg.names(), ", ".join(added))
    call = reg.invoke("rag_search", {"query": "安全层负责什么", "top_k": 1})
    _ok("Agent 可调用 rag_search", call.get("ok", False),
        (call.get("output") or call.get("error") or "")[:60])

    # ---------- 第四阶段：经验沉淀 + 语义召回 ----------
    print("\n== 第四阶段：任务经验向量化与语义召回 ==")
    fake_result = {
        "task_id": "20260101-120000-abcd",
        "answer": "已统计出三个模块的代码行数并写入报表。",
        "ok": True,
        "stop_reason": "finished",
        "steps": 3,
        "model": "qwen2.5:7b-instruct",
        "plan": ["列出目录", "统计各模块行数", "汇总成报表"],
        "trace": [
            {"step": 1, "tool": "list_dir", "args": {"path": "src"}, "ok": True},
            {"step": 2, "tool": "run_python", "args": {"code": "print(len(open('a.py').readlines()))"},
             "ok": False, "error": "FileNotFoundError: a.py"},
            {"step": 3, "tool": "write_file", "args": {"path": "report.csv"}, "ok": True},
        ],
        "artifacts": ["report.csv"],
    }
    ing = store.ingest("统计各模块代码行数并生成汇总报表", fake_result)
    _ok("经验入库", ing.get("status") in ("success", "unchanged"), ing.get("status", ""))

    exp_hits = store.recall("帮我算一下每个模块有多少行代码", k=2, min_score=0.25)
    _ok("语义召回（用词不同但语义相近）", len(exp_hits) > 0,
        f"top={exp_hits[0].score:.2f}" if exp_hits else "无命中")
    if exp_hits:
        print("   " + exp_hits[0].to_card(1).replace("\n", "\n   ")[:300])

    card_text = store.build_card("统计代码行数", fake_result)
    _ok("脱敏：经验卡不含代码原文", "open('a.py'" not in card_text,
        "敏感参数已省略" if "open('a.py'" not in card_text else "泄露了代码")

    print("\n统计：", engine.stats()["collections"])
    print(f"\n全部验证完成，临时库位于 {tmp}（可手动删除）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
