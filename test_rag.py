# test_rag.py
"""RAG 知识库五阶段端到端验证（40 项断言）。

用临时目录建向量库，跑完即弃，不污染项目的 agent_knowledge_base。
需要本机 Ollama 已拉取默认 embedding 模型（见 rag_engine.DEFAULT_EMBED_MODEL）。
有任何断言失败时退出码为 1，可直接用于 CI / 回归门禁。

覆盖范围：
  阶段0  向量化服务探活
  分块   结构感知分块、三级标题根路径保留、碎片合并、分隔线过滤、算法版本感知
  一     文档入库、幂等入库、相似度检索
  二     检索结果注入上下文渲染
  三     rag_search 注册为标准工具
  四     任务经验向量化、语义召回、脱敏
  五     混合检索：BM25 词法通道、RRF 融合分量纲统一、top_k 与候选池解耦、
        无重复块、纯词法命中带出处、跨库二次融合、改写解析分流与标签过滤

运行：.venv\\Scripts\\python.exe test_rag.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from rag_engine import DEFAULT_EMBED_MODEL, RagConfig, RagEngine
from experience import ExperienceStore
from engine import ToolRegistry
from rag_tools import register_rag_tools

# 失败计数：让退出码真实反映测试结果。
# 早期版本 main() 硬编码 return 0，失败项只打印✗却不影响退出码，
# 回归测试形同虚设——CI/脚本无法据此判断是否通过。
_FAILURES = {"n": 0}


def _ok(label: str, cond: bool, extra: str = "") -> bool:
    print(f"{'✓' if cond else '✗'} {label}{(' — ' + extra) if extra else ''}")
    if not cond:
        _FAILURES["n"] += 1
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

    # ---------- 分块质量：三级标题 + 表格（精确复现「根标题丢失」缺陷）----------
    # 上面的 md 只有两级标题，heading_stack[-2:] 恰好含根标题，测不出该缺陷。
    # 真实故障场景是三级嵌套：深层小节的前缀会丢掉承载产品名的根标题，
    # 导致查询带产品名时「浅层章节压过真正含答案的深层章节」（排序倒挂）。
    print("\n== 分块质量：三级标题根路径保留 ==")
    md3 = (
        "# HaloSync 技术手册\n\n"
        "## 1 产品概述\n\n"
        "HaloSync 是企业级数据同步平台。\n\n"
        "## 3 部署要求\n\n"
        "### 3.1 硬件规格\n\n"
        "| 项 | 最低配置 | 推荐配置 |\n"
        "|---|---|---|\n"
        "| CPU | 4 核 | 8 核 |\n"
        "| 内存 | 8 GB | 16 GB |\n\n"
        "> 吞吐上限由软件硬限决定，增加 CPU 不会突破。\n\n"
        "---\n\n"
        "### 3.2 端口清单\n\n"
        "| 端口 | 协议 | 用途 |\n"
        "|---|---|---|\n"
        "| 8443 | HTTPS | 管理 API |\n"
        "| 9092 | gRPC | 组件间通信 |\n"
    )
    chunks3 = engine._chunk_text(md3)
    deep = [c for c in chunks3 if "硬件规格" in c.split("\n")[0]]
    _ok("三级小节块保留文档根标题", bool(deep) and "HaloSync" in deep[0].split("\n")[0],
        deep[0].split("\n")[0] if deep else "未找到硬件规格块")

    # 碎片合并：表格与其后的补充说明属同一小节，应装进同一块
    if deep:
        _ok("表格与补充说明合并（不再产生碎片块）",
            "| CPU |" in deep[0] and "吞吐上限" in deep[0],
            f"len={len(deep[0])}")

    # 分隔线不得进入块内容
    _ok("--- 分隔线已过滤", not any(c.strip() == "---" or "\n---\n" in c for c in chunks3))

    # 浅层块同样带根标题（回归保护，确保修复没有破坏原有行为）
    shallow = [c for c in chunks3 if "产品概述" in c.split("\n")[0]]
    _ok("二级小节块仍保留根标题", bool(shallow) and "HaloSync" in shallow[0].split("\n")[0])

    # ---------- 分块算法版本：改算法必须强制重建 ----------
    print("\n== 幂等入库感知分块算法版本 ==")
    h1 = engine._content_hash("同一份正文", "x.md")
    engine.cfg.chunk_size = 601  # 改参数
    h2 = engine._content_hash("同一份正文", "x.md")
    engine.cfg.chunk_size = 600
    _ok("分块参数变化 → 指纹变化（会重建）", h1 != h2)
    import rag_engine as _re
    saved = _re.CHUNKER_VERSION
    _re.CHUNKER_VERSION = saved + 1
    h3 = engine._content_hash("同一份正文", "x.md")
    _re.CHUNKER_VERSION = saved
    _ok("分块算法版本变化 → 指纹变化（避免静默沿用旧向量）", h1 != h3)

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

    # ---------- 第五阶段：混合检索（BM25 + RRF + top_k 解耦 + 量纲统一）----------
    # 这些能力本轮新增却一度零测试保护；下面是针对每个修复点的回归断言。
    print("\n== 第五阶段：混合检索 ==")

    # 先入库一篇带精确标识符的技术文档（BM25 的主场）
    tech = tmp / "tech.md"
    tech.write_text(
        "# HaloSync 技术手册\n\n"
        "## 1 产品概述\n\nHaloSync 是企业级数据同步平台。\n\n"
        "## 7 错误码\n\n"
        "| 错误码 | 含义 |\n|---|---|\n"
        "| E4201 | 源端认证失败 |\n| E4211 | 触发限流 |\n\n"
        "## 3 部署要求\n\n### 3.2 端口清单\n\n"
        "| 端口 | 用途 |\n|---|---|\n| 8443 | 管理 API |\n| 9092 | 组件间通信 |\n",
        encoding="utf-8",
    )
    engine.add_document(str(tech))

    # BM25 词法通道：精确标识符（错误码）应被检索到
    lex = engine.search_lexical("E4201 是什么错误", top_k=3)
    _ok("BM25 词法通道命中错误码", any("E4201" in (h.get("content") or "") for h in lex),
        f"{len(lex)} 条")

    # 混合检索：纯词法命中的块必须带出处（否则用户看到答案却不知来源）
    hyb = engine.search_hybrid("E4201 错误码含义", top_k=3)
    _ok("混合检索返回结果", len(hyb) > 0, f"{len(hyb)} 条")
    if hyb:
        only_lex = [h for h in hyb if h.get("matched_by") == "lexical"]
        _ok("纯词法命中也带出处（source 非空）",
            all(h.get("source") for h in only_lex) if only_lex else True,
            f"{len(only_lex)} 条纯词法" if only_lex else "无纯词法命中")
        _ok("结果含 matched_by 标记", all(h.get("matched_by") for h in hyb))

    # top_k 解耦：run(k) 必须是 run(K) 的前缀，即 top_k 只截断不改排序
    big = engine.search_hybrid("端口和错误码", top_k=8)
    small = engine.search_hybrid("端口和错误码", top_k=3)
    prefix_ok = all(
        abs(a["relevance_score"] - b["relevance_score"]) < 1e-6
        for a, b in zip(small, big[:len(small)])
    )
    _ok("top_k 与候选池解耦（前缀性质）", prefix_ok,
        f"k=3 前{len(small)}名与 k=8 一致" if prefix_ok else "排序随 top_k 漂移")

    # rrf_score 量纲统一：短路分支与融合分支都应是名次倒数量级（<0.2），
    # 不能出现短路填余弦分（~0.6）导致跨库比较失真
    rrf_vals = [h.get("rrf_score", 0) for h in big if h.get("rrf_score")]
    _ok("融合分量纲统一（均为名次倒数量级）",
        all(0 < v < 0.2 for v in rrf_vals) if rrf_vals else True,
        f"max={max(rrf_vals):.4f}" if rrf_vals else "无融合分")

    # 无重复块：同一查询的多次返回不应含重复 id
    ids = [h.get("id") for h in big if h.get("id")]
    _ok("混合检索无重复块", len(ids) == len(set(ids)), f"{len(ids)} 个 id")

    # 跨库二次 RRF：不报错且量纲统一
    multi = engine.search_multi("端口", ["agent_general_knowledge", "agent_experience"],
                                top_k=3, hybrid=True)
    _ok("跨库混合检索不报错", isinstance(multi, list), f"{len(multi)} 条")
    if multi:
        xscores = [h.get("rrf_score", 0) for h in multi]
        _ok("跨库融合分量纲统一", all(0 < s < 0.2 for s in xscores), f"{xscores}")

    # 查询改写器解析：标签不得混入问句改写（修复过的 bug）
    try:
        from query_expander import QueryExpander
        parsed = QueryExpander._parse(
            "改写：\n数据库连接池怎么配\n术语：\n连接池 配置 参数",
            "怎么配连接池", 3, 2)
        _ok("改写解析：问句与术语分流", 
            parsed.paraphrases == ["数据库连接池怎么配"] and "术语" not in "".join(parsed.paraphrases),
            f"问句={parsed.paraphrases} 术语={parsed.terms}")
        _ok("改写解析：标签行被过滤", "术语：" not in parsed.paraphrases)
    except Exception as e:
        _ok("改写解析可运行", False, f"{type(e).__name__}: {e}")

    print("\n统计：", engine.stats()["collections"])
    if _FAILURES["n"]:
        print(f"\n✗ {_FAILURES['n']} 项未通过，临时库保留在 {tmp} 供排查")
        return 1
    print(f"\n✓ 全部通过，临时库位于 {tmp}（可手动删除）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
