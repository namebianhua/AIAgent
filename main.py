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
) -> Dict[str, Any]:
    """兼容旧入口：内部统一走 runner.execute。"""
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
