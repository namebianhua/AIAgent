from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import xhs_web
from runner import RunOptions, build_registry_for_cli, execute
from skills import delete_skill, load_skills
from task_memory import MemoryStore
from tracing import list_traces, read_trace, render_events


def xhs_login(profile_dir: str = ".agent/browser/xhs", timeout: int = 180) -> int:
    """打开有头浏览器，等用户扫码登录小红书，登录态持久化到 profile_dir。"""
    cfg = xhs_web.XhsConfig(profile_dir=profile_dir, headless=False)
    session = xhs_web.XhsSession(cfg)
    print("[xhs] 正在启动浏览器……")
    try:
        session.start(offscreen=False)
    except Exception as e:
        print(f"[xhs] 浏览器启动失败：{e}", file=sys.stderr)
        return 1

    try:
        if session.is_logged_in():
            print("[xhs] 已经是登录状态，无需重复登录。")
            print(f"[xhs] 登录态目录：{session.profile_dir}")
            return 0
        ok = session.ensure_login(wait_seconds=timeout)
    finally:
        session.close()

    if ok:
        print("[xhs] 登录成功，登录态已保存。之后可以直接运行任务。")
        print(f"[xhs] 登录态目录：{cfg.profile_dir}")
        return 0
    print("[xhs] 登录超时，未检测到登录态。请重新运行 --xhs-login。", file=sys.stderr)
    return 1


def run(
    goal: str,
    mode: str = "web",
    workspace: str = ".",
    readonly: bool = False,
    headless: bool = False,
    auto_yes: bool = False,
    max_steps: int = 15,
    snapshot: bool = False,
    use_tools: bool = True,
    use_xhs: bool = False,
    xhs_headless: bool = True,
    xhs_profile: str = ".agent/browser/xhs",
    autonomous: bool = False,
    save_skill: Optional[str] = None,
    use_memory: bool = True,
    use_critic: bool = True,
    max_plan_steps: int = 6,
    max_replans: int = 2,
    max_seconds: int = 900,
    use_local: bool = False,
    ollama_base: str = "",
    ollama_model: str = "",
    full_computer: bool = False,
    **rag_opts: Any,
) -> Dict[str, Any]:
    """兼容旧入口：内部统一走 runner.execute。

    rag_opts: 透传给 RunOptions 的知识库字段（use_rag / rag_inject / rag_tools /
              use_experience / rag_collection 等），未提供时用 RunOptions 默认值。
    """
    opts = RunOptions(
        mode=mode,
        workspace=workspace,
        readonly=readonly,
        headless=headless,
        auto_yes=auto_yes,
        max_steps=max_steps,
        snapshot=snapshot,
        use_tools=use_tools,
        use_xhs=use_xhs,
        xhs_headless=xhs_headless,
        xhs_profile=xhs_profile,
        autonomous=autonomous,
        save_skill=save_skill,
        use_memory=use_memory,
        use_critic=use_critic,
        max_plan_steps=max_plan_steps,
        max_replans=max_replans,
        max_seconds=max_seconds,
        use_local=use_local,
        ollama_base=ollama_base,
        ollama_model=ollama_model,
        full_computer=full_computer,
        **rag_opts,
    )
    return execute(goal, opts)



def print_trace(result: Dict[str, Any]) -> None:
    if result.get("autonomous"):
        print_step_results(result)
        return
    trace = result.get("trace") or []
    if not trace:
        return
    print(f"\n--- 执行轨迹（{len(trace)} 步 / 终止原因：{result.get('stop_reason')}）---")
    for item in trace:
        status = "OK  " if item.get("ok") else "FAIL"
        tool = item.get("tool", "?")
        args = json.dumps(item.get("args") or {}, ensure_ascii=False)
        if len(args) > 90:
            args = args[:90] + "..."
        line = f"  {item.get('step', '?'):>2}. [{status}] {tool} {args} ({item.get('elapsed_ms')}ms)"
        if not item.get("ok") and item.get("error"):
            line += f"  <- {str(item['error'])[:120]}"
        print(line)


def print_step_results(result: Dict[str, Any]) -> None:
    plan = result.get("plan") or []
    if plan:
        print(f"\n--- 计划（{len(plan)} 步）---")
        for i, s in enumerate(plan, 1):
            print(f"  {i}. {s}")
    print(f"\n--- 步骤执行（{len(result.get('step_results') or [])} 步 / {result.get('stop_reason')}）---")
    for item in result.get("step_results") or []:
        flag = "OK  " if item.get("ok") else "FAIL"
        print(f"  {item.get('index')}. [{flag}] {item.get('step')}")
        if item.get("critique"):
            print(f"       校验：{item['critique']}")
    if result.get("replans"):
        print(f"  重规划 {result['replans']} 次")
    if result.get("artifacts"):
        print(f"  产物：{', '.join(result['artifacts'])}")


def _memory_store(workspace: Path) -> MemoryStore:
    return MemoryStore(workspace / ".agent" / "memory.db")


def _rag_opts_from_args(args, workspace: Path) -> Dict[str, Any]:
    """把 CLI 的 --no-rag* / --rag-* / --no-experience 等映射成 RunOptions 字段。"""
    from rag_engine import DEFAULT_EMBED_MODEL

    db_path = args.rag_db or str(workspace / "agent_knowledge_base")
    return {
        "use_rag": not args.no_rag,
        "rag_db_path": db_path,
        "rag_collection": args.rag_collection or "agent_general_knowledge",
        "rag_embed_model": args.rag_embed_model or os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        "rag_ollama_url": args.ollama_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        "rag_inject": not args.no_rag_inject,
        "rag_tools": not args.no_rag_tools,
        "rag_top_k": args.kb_top_k,
        "rag_min_score": args.rag_min_score,     # None = 沿用库内标定门槛
        "use_experience": not (args.no_experience or args.no_rag),
        "exp_collection": args.exp_collection or "agent_experience",
        "exp_min_score": args.exp_min_score,      # None = 沿用库内标定门槛
    }


# ===================== 知识库 CLI（第一~四阶段）=====================

def _rag_engine(args, workspace: Path):
    """按 CLI 参数构造 RagEngine；不可用时打印原因并返回 None。"""
    from rag_engine import DEFAULT_EMBED_MODEL, RagConfig, RagEngine, RagUnavailable

    cfg = RagConfig(
        db_path=args.rag_db or str(workspace / "agent_knowledge_base"),
        embed_model=args.rag_embed_model or os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        ollama_url=args.ollama_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        chunk_size=args.rag_chunk_size,
        chunk_overlap=args.rag_chunk_overlap,
    )
    try:
        return RagEngine(cfg)
    except RagUnavailable as e:
        print(f"[知识库] 不可用：{e}", file=sys.stderr)
        return None


def _print_rag_results(results, query: str) -> None:
    if not results:
        print(f"（知识库中没有与「{query}」相关的内容）")
        return
    for i, r in enumerate(results, 1):
        print(f"\n--- [{i}] 相关度 {r['relevance_score']:.2f} | 来源：{r.get('source') or '-'} ---")
        print(r["content"].strip())


def kb_add(args, workspace: Path) -> int:
    """入库文件/目录。"""
    engine = _rag_engine(args, workspace)
    if engine is None:
        return 1
    probe = engine.probe()
    if not probe.get("ok"):
        print(f"[知识库] 向量化服务未就绪：{probe.get('reason')}", file=sys.stderr)
        return 1
    print(f"[知识库] embedding 模型 {probe['embed_model']}，向量维度 {probe.get('dim')}")

    coll = args.rag_collection or engine.default_collection
    targets = args.kb_path
    total_chunks = 0
    failed = 0
    for t in targets:
        p = Path(t).expanduser()
        if not p.is_absolute():
            p = (workspace / p).resolve()
        try:
            if p.is_dir():
                res = engine.add_directory(p, collection_name=coll, recursive=not args.no_recursive)
                print(f"[目录] {p}")
                print(f"  扫描 {res.get('scanned', 0)} 个受支持文件，新增 {res.get('added', 0)} 个，"
                      f"共 {res.get('chunks', 0)} 块，未变化 {res.get('unchanged', 0)} 个")
                for err in (res.get("errors") or []):
                    failed += 1
                    print(f"  ✗ {err.get('file')}：{err.get('reason')}")
                total_chunks += int(res.get("chunks", 0))
            elif p.is_file():
                res = engine.add_document(p, collection_name=coll)
                status = res.get("status")
                if status == "unchanged":
                    print(f"[跳过] {p.name}：内容未变化（库中已有该版本，{res.get('chunk_count')} 块）")
                    total_chunks += int(res.get("chunk_count", 0))
                elif status == "success":
                    print(f"[入库] {p.name}：{res.get('chunk_count')} 块")
                    total_chunks += int(res.get("chunk_count", 0))
                else:
                    failed += 1
                    print(f"[失败] {p.name}：{res.get('reason')}", file=sys.stderr)
            else:
                failed += 1
                print(f"[失败] 路径不存在：{t}", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"[失败] {t}：{type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n知识库「{coll}」当前共 {engine._collection(coll).count()} 个文本块")
    return 1 if failed and not total_chunks else 0


def kb_search(args, workspace: Path) -> int:
    engine = _rag_engine(args, workspace)
    if engine is None:
        return 1
    coll = args.rag_collection or engine.default_collection
    results = engine.search(args.kb_query, top_k=args.kb_top_k, collection_name=coll)
    _print_rag_results(results, args.kb_query)
    return 0


def kb_status(args, workspace: Path) -> int:
    engine = _rag_engine(args, workspace)
    if engine is None:
        return 1
    print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
    probe = engine.probe()
    print("\n[向量化服务]", "就绪" if probe.get("ok") else f"不可用：{probe.get('reason')}")
    return 0


def kb_list(args, workspace: Path) -> int:
    engine = _rag_engine(args, workspace)
    if engine is None:
        return 1
    coll = args.rag_collection or engine.default_collection
    sources = engine.list_sources(coll)
    if not sources:
        print(f"（知识库「{coll}」为空，用 --kb-add 入库文档）")
        return 0
    print(f"知识库「{coll}」共 {len(sources)} 个来源：")
    for s in sources:
        print(f"- {s['source_name'] or s['source_id']}  ({s['chunks']} 块)  入库于 {s['indexed_at']}")
    return 0


def kb_calibrate(args, workspace: Path) -> int:
    """用库内已有文档自动标定相关度门槛（换 embedding 模型后必须重跑）。"""
    engine = _rag_engine(args, workspace)
    if engine is None:
        return 1

    targets = [args.rag_collection or engine.default_collection]
    if args.exp_collection and args.exp_collection not in targets:
        targets.append(args.exp_collection)

    any_ok = False
    for coll in targets:
        if coll not in engine.list_collections():
            print(f"[跳过] 知识库「{coll}」不存在")
            continue
        print(f"\n== 标定「{coll}」（embedding 模型：{engine.cfg.embed_model}）==")
        res = engine.auto_calibrate(collection_name=coll)
        if not res.get("ok"):
            print(f"  ✗ {res.get('error')}")
            continue
        any_ok = True
        s = res["samples"]
        basis = ("跨文档取样" if res.get("sample_basis") == "cross_document"
                 else "跨小节取样（单文档降级）")
        print(f"  样本：{s['docs']} 篇文档 / {s['groups']} 个分组 / {s['chunks']} 块，"
              f"正例对 {s['pos_pairs']}，负例对 {s['neg_pairs']}")
        print(f"  取样方式：{basis}")
        print(f"  正例 P10={res['pos_p10']}（最低 {res['pos_min']}）｜"
              f"负例 P90={res['neg_p90']}（最高 {res['neg_max']}）")
        print(f"  分离度 margin={res['margin']}（>0 表示正负例可区分）")
        if res.get("persisted"):
            print(f"  → 建议门槛 {res['suggested_min_score']}，已写入库元数据并自动沿用")
        else:
            print(f"  → 建议门槛 {res['suggested_min_score']}（未写入）")
            if res.get("not_persisted_reason"):
                print(f"  ⚠️ {res['not_persisted_reason']}")
        print(f"  结论：{res['verdict']}")
        if res.get("caveat"):
            print(f"  ⚠️ {res['caveat']}")

    if not any_ok:
        print("\n没有可标定的知识库。请先用 --kb-add 入库文档"
              "（至少 2 块；单篇文档会降级为跨小节取样）。", file=sys.stderr)
        return 1
    return 0


def kb_exp(args, workspace: Path) -> int:
    """经验库：状态 / 语义召回 / 从历史轨迹回灌。"""
    from experience import ExperienceStore

    engine = _rag_engine(args, workspace)
    if engine is None:
        return 1
    store = ExperienceStore(engine=engine, collection=args.exp_collection)

    if args.exp_query:
        hits = store.recall(args.exp_query, k=args.kb_top_k, min_score=args.exp_min_score)
        if not hits:
            print(f"（没有找到与「{args.exp_query}」语义相近的历史任务经验）")
            return 0
        for i, h in enumerate(hits, 1):
            print(h.to_card(i))
            print()
        return 0

    if args.exp_reindex:
        traces_dir = workspace / ".agent" / "traces"
        files = sorted(traces_dir.glob("*.jsonl")) if traces_dir.is_dir() else []
        if not files:
            print(f"（{traces_dir} 下没有轨迹文件）")
            return 0
        print(f"从 {len(files)} 份历史轨迹回灌经验库……")
        ok = skipped = failed = 0
        for f in files:
            res = store.ingest_from_trace_file(f)
            status = res.get("status")
            if status in ("success", "unchanged"):
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  ✗ {f.name}：{res.get('reason')}")
        print(f"完成：入库 {ok}，跳过 {skipped}，失败 {failed}")
        return 0 if failed == 0 else 1

    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="模型 Agent：T1 工具调用 / T2 网页操作 / T3 自主任务 / 控制台（--serve）"
    )
    parser.add_argument("goal_pos", nargs="?", help="任务目标（位置参数）")
    parser.add_argument("-p", "--prompt", help="任务目标（同位置参数，二选一）")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["api", "web", "hybrid"],
        default="web",
        help="api=纯 API；web=纯网页（免 Key）；hybrid=API 优先、网页兜底",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=os.getenv("AGENT_WORKSPACE", "."),
        help="agent 的沙箱工作区根目录（默认当前目录）",
    )
    parser.add_argument("--readonly", action="store_true", help="只读模式：禁止任何写入")
    parser.add_argument("--headless", action="store_true", help="浏览器无头模式")
    parser.add_argument("-y", "--yes", action="store_true", help="自动批准所有高危操作")
    parser.add_argument("--max-steps", type=int, default=15, help="单任务最大工具调用步数")
    parser.add_argument("--snapshot", action="store_true", help="任务前对工作区打快照")
    parser.add_argument("--no-tools", action="store_true", help="纯问答模式，不启用工具")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    parser.add_argument("--tools", action="store_true", help="只打印工具清单后退出")

    # ---- T3 ----
    t3 = parser.add_argument_group("T3 自主任务")
    t3.add_argument("--autonomous", action="store_true", help="启用规划+执行+校验的自主模式")
    t3.add_argument("--max-plan-steps", type=int, default=6, help="计划最多几步")
    t3.add_argument("--max-replans", type=int, default=2, help="最多重规划几次")
    t3.add_argument("--max-seconds", type=int, default=900, help="整个任务的秒级预算")
    t3.add_argument("--no-critic", action="store_true", help="关闭步骤验收（快但容易假成功）")
    t3.add_argument("--no-memory", action="store_true", help="不读写长期记忆")
    t3.add_argument("--no-skills", action="store_true", help="不加载技能库")
    t3.add_argument("--save-skill", metavar="NAME", help="任务成功后把轨迹固化成技能")
    t3.add_argument("--memory", action="store_true", help="查看长期记忆统计后退出")
    t3.add_argument("--recall", metavar="QUERY", help="检索历史经验后退出")
    t3.add_argument("--skills", action="store_true", help="列出技能库后退出")
    t3.add_argument("--forget-skill", metavar="NAME", help="删除一个技能后退出")
    t3.add_argument("--traces", action="store_true", help="列出最近任务轨迹后退出")
    t3.add_argument("--show-trace", metavar="TASK_ID", help="回放某个任务的轨迹后退出")
    t3.add_argument("--serve", action="store_true", help="启动 HTTP 控制台（SSE 实时进度）")
    t3.add_argument("--ui", action="store_true", help="同 --serve（交互界面）")
    t3.add_argument("--port", type=int, default=8787, help="服务端口（默认 8787）")
    t3.add_argument("--host", default="127.0.0.1", help="监听地址（非本机时必须配 --token）")
    t3.add_argument("--token", default=os.getenv("AGENT_UI_TOKEN", ""), help="访问令牌（几个人共用时建议设置）")
    t3.add_argument("--demo", action="store_true", help="演示模式：脚本化假模型，不联网不启浏览器")
    t3.add_argument("--confirm-timeout", type=int, default=30, help="确认弹窗超时秒数（超时自动拒绝）")
    t3.add_argument("--confirm-writes", action="store_true", help="写文件也要人工确认（默认只拦高危命令）")
    t3.add_argument("--pause-on-failure", action="store_true", help="工具失败时暂停，人工选择重试/跳过/改参数")
    t3.add_argument("--with-scheduler", action="store_true", help="服务同时启用定时任务")

    # ---- 本地模型（Ollama）----
    parser.add_argument(
        "--local-model",
        metavar="MODEL",
        default=os.getenv("OLLAMA_MODEL", ""),
        help="启用本地 Ollama 通道（本地优先，失败自动降级 API/网页）。"
        "例如 qwen2.5:7b-instruct；留空且未设 OLLAMA_MODEL 时不启用",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Ollama 服务地址（默认 http://127.0.0.1:11434）",
    )

    # ---- 电脑操控 ----
    parser.add_argument(
        "--full-computer",
        action="store_true",
        help="放开到整台电脑：可访问工作区外的文件路径，并启用 sys_*（系统查询）"
        "与 ui_*（桌面操作：鼠标/键盘/窗口/程序）工具。系统目录禁写，"
        "工作区外写入与所有桌面动作需人工确认",
    )

    # ---- 知识库 RAG（第一~四阶段）----
    kb = parser.add_argument_group("知识库（RAG）")
    kb.add_argument("--kb-add", nargs="+", metavar="PATH", dest="kb_path",
                    help="把文件/目录解析并向量化后存入知识库，完成后退出（可传多个路径）")
    kb.add_argument("--kb-search", metavar="QUERY", dest="kb_query",
                    help="在知识库中做语义检索后退出")
    kb.add_argument("--kb-status", action="store_true", help="查看知识库统计与向量化服务状态后退出")
    kb.add_argument("--kb-list", action="store_true", help="列出知识库中已入库的文档后退出")
    kb.add_argument("--kb-top-k", type=int, default=5, help="检索返回条数（默认 5）")
    kb.add_argument("--rag-collection", default="",
                    help="知识库名（默认 agent_general_knowledge）")
    kb.add_argument("--rag-db", default="",
                    help="向量库目录（默认 <工作区>/agent_knowledge_base）")
    kb.add_argument("--rag-embed-model", default="",
                    help="embedding 模型（默认 bge-m3，需先 ollama pull bge-m3）")
    kb.add_argument("--rag-chunk-size", type=int, default=600, help="分块字符数（默认 600）")
    kb.add_argument("--rag-chunk-overlap", type=int, default=80, help="分块重叠字符数（默认 80）")
    kb.add_argument("--no-recursive", action="store_true", help="入库目录时不递归子目录")
    kb.add_argument("--kb-calibrate", action="store_true",
                    help="用库内已有文档自动标定相关度门槛后退出（换 embedding 模型后必须重跑）")
    kb.add_argument("--rag-min-score", type=float, default=None,
                    help="临时覆盖知识库的相关度门槛（默认沿用 --kb-calibrate 标定值）")
    kb.add_argument("--no-rag", action="store_true",
                    help="跑任务时完全关闭知识库（不建引擎、不注册工具、不注入）")
    kb.add_argument("--no-rag-inject", action="store_true",
                    help="关闭第二阶段的知识自动注入（仍保留 rag_search 主动检索工具）")
    kb.add_argument("--no-rag-tools", action="store_true",
                    help="不注册第三阶段的 rag_search / rag_ingest / rag_status / exp_recall 工具")
    kb.add_argument("--no-experience", action="store_true",
                    help="关闭第四阶段的经验库（不再沉淀任务经验，也不做语义召回）")
    kb.add_argument("--exp-collection", default="agent_experience",
                    help="经验库 collection 名（默认 agent_experience）")
    kb.add_argument("--exp-query", metavar="GOAL",
                    help="语义检索历史任务经验后退出（第四阶段）")
    kb.add_argument("--exp-reindex", action="store_true",
                    help="把 .agent/traces 下的历史轨迹回灌进经验库后退出")
    kb.add_argument("--exp-min-score", type=float, default=None,
                    help="临时覆盖经验召回的相关度门槛（默认沿用标定值）")

    # ---- T2 ----
    parser.add_argument("--xhs", action="store_true", help="启用小红书工具（T2 网页版操作）")
    parser.add_argument(
        "--xhs-login", action="store_true", help="打开窗口扫码登录小红书，成功后退出"
    )
    parser.add_argument(
        "--xhs-headless",
        action="store_true",
        help="小红书浏览器用无头模式（默认有头但移出屏幕，指纹更真实）",
    )
    parser.add_argument(
        "--xhs-profile",
        default=".agent/browser/xhs",
        help="小红书登录态（浏览器 profile）目录",
    )
    parser.add_argument(
        "--xhs-login-timeout", type=int, default=180, help="扫码登录等待秒数"
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()

    # ---------- 一次性动作 ----------
    if args.xhs_login:
        return xhs_login(args.xhs_profile, timeout=args.xhs_login_timeout)

    # ---------- 知识库一次性动作（RAG）----------
    if args.kb_path:
        return kb_add(args, workspace)
    if args.kb_query:
        return kb_search(args, workspace)
    if args.kb_status:
        return kb_status(args, workspace)
    if args.kb_list:
        return kb_list(args, workspace)
    if args.kb_calibrate:
        return kb_calibrate(args, workspace)
    if args.exp_query or args.exp_reindex:
        return kb_exp(args, workspace)

    if args.memory:
        stats = _memory_store(workspace).stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    if args.recall:
        store = _memory_store(workspace)
        records = store.recall(args.recall, k=5)
        if not records:
            print("（没有找到相似的历史任务）")
            return 0
        for i, r in enumerate(records, 1):
            print(r.to_card(i))
            print()
        return 0

    if args.skills:
        skills = load_skills(workspace / ".agent")
        if not skills:
            print("（技能库为空。用 --save-skill NAME 在成功任务后固化一个）")
            return 0
        for s in skills:
            print(f"- skill_{s.name}: {s.description}")
            print(f"    流程：{' → '.join(st['tool'] for st in s.steps)}")
            if s.params:
                print(f"    参数：{', '.join(s.params)}")
            print(f"    调用 {s.uses} 次 / 成功 {s.successes} 次")
        return 0

    if args.forget_skill:
        ok = delete_skill(workspace / ".agent", args.forget_skill)
        print("已删除" if ok else "没有这个技能")
        return 0

    if args.traces:
        rows = list_traces(workspace / ".agent", limit=20)
        if not rows:
            print("（还没有任务轨迹）")
            return 0
        for r in rows:
            flag = "成功" if r.get("ok") else (r.get("stop_reason") or "未完成")
            print(f"- {r['task_id']}  [{flag}]  {r.get('tools', 0)} 次工具调用  {r.get('mtime')}")
            print(f"    {str(r.get('goal') or '')[:90]}")
        return 0

    if args.show_trace:
        events = read_trace(workspace / ".agent", args.show_trace)
        print(render_events(events))
        return 0

    if args.serve or args.ui:
        from server import serve

        opts = RunOptions(
            mode=args.mode,
            workspace=str(workspace),
            headless=args.headless,
            auto_yes=args.yes,
            max_steps=args.max_steps,
            autonomous=args.autonomous,
            use_memory=not args.no_memory,
            use_critic=not args.no_critic,
            use_skills=not args.no_skills,
            max_plan_steps=args.max_plan_steps,
            max_replans=args.max_replans,
            max_seconds=args.max_seconds,
            confirm_timeout=args.confirm_timeout,
            confirm_writes=args.confirm_writes,
            pause_on_failure=args.pause_on_failure,
            use_xhs=args.xhs,
            xhs_headless=args.xhs_headless,
            xhs_profile=args.xhs_profile,
            use_local=bool(args.local_model),
            ollama_base=args.ollama_url,
            ollama_model=args.local_model or "",
            full_computer=args.full_computer,
            **_rag_opts_from_args(args, workspace),
        )
        serve(
            opts,
            port=args.port,
            host=args.host,
            with_scheduler=args.with_scheduler,
            token=args.token,
            demo=args.demo,
        )
        return 0

    if args.xhs and args.no_tools:
        print("[提示] --xhs 依赖工具注册表，已自动忽略 --no-tools", file=sys.stderr)
        args.no_tools = False

    if args.tools:
        opts = RunOptions(
            workspace=str(workspace),
            use_xhs=args.xhs,
            xhs_headless=args.xhs_headless,
            xhs_profile=args.xhs_profile,
            use_skills=not args.no_skills,
            full_computer=args.full_computer,
        )
        reg = build_registry_for_cli(opts)
        print(json.dumps(reg.list_tools(), ensure_ascii=False, indent=2))
        return 0

    # ---------- 跑任务 ----------
    goal = args.prompt or args.goal_pos
    if not goal:
        try:
            goal = input("目标：").strip()
        except EOFError:
            goal = ""
    if not goal:
        print("没有输入目标", file=sys.stderr)
        return 2

    try:
        result = run(
            goal,
            mode=args.mode,
            workspace=str(workspace),
            readonly=args.readonly,
            headless=args.headless,
            auto_yes=args.yes,
            max_steps=args.max_steps,
            snapshot=args.snapshot,
            use_tools=not args.no_tools,
            use_xhs=args.xhs,
            xhs_headless=args.xhs_headless,
            xhs_profile=args.xhs_profile,
            autonomous=args.autonomous,
            save_skill=args.save_skill,
            use_memory=not args.no_memory,
            use_critic=not args.no_critic,
            max_plan_steps=args.max_plan_steps,
            max_replans=args.max_replans,
            max_seconds=args.max_seconds,
            use_local=bool(args.local_model),
            ollama_base=args.ollama_url,
            ollama_model=args.local_model or "",
            full_computer=args.full_computer,
            **_rag_opts_from_args(args, workspace),
        )
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:
        print(f"执行失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.json:
        payload = {k: v for k, v in result.items() if k not in ("messages", "options")}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\n[{result.get('model', 'unknown')}] {result.get('answer', '')}")
        print_trace(result)
        if result.get("skill_saved"):
            print(f"\n技能已固化：{result['skill_saved']}")
        print(f"\n工作区：{result.get('workspace')}")
        if result.get("task_id"):
            print(f"轨迹：--show-trace {result['task_id']}")
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
