"""统一执行入口（T3）：CLI / HTTP 服务 / 定时任务共用同一条装配逻辑。

把「建沙箱 → 建注册表 → 装技能 → 配模型 → 跑任务 → 收尾审计」收敛到一处，
避免三个入口各写一遍、行为不一致。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import xhs_tools
from agent import Agent, AgentConfig
from api_clients import ModelClient, ModelConfig, ModelRouter
from autonomous import AutoConfig, AutonomousAgent
from builtin_tools import build_registry
import computer_tools
from gateway import ModelGateway
from sandbox import Sandbox
from skills import load_skills, register_skills
from task_memory import MemoryStore
from tracing import TraceLogger
from web_clients import QwenWebClient, WenxinWebClient, ZhipuWebClient, DeepseekWebClient
from web_router import WebModelRouter

DEFAULT_SYSTEM_PROMPT = (
    "你是一个能操作电脑的 AI 助手。你只能通过注册表里的工具动手，"
    "不能编造工具名，也不能假装执行过动作。"
)


# 电脑操控模式（--full-computer）下的附加系统提示
COMPUTER_SYSTEM_ADDENDUM = """

【电脑操控模式已开启】你现在可以直接操作整台真实电脑：
1. sys_* 是只读查询：sys_info（系统信息）/ sys_drives（盘符）/ sys_find_file（全盘找文件）/
   sys_screenshot（截屏给用户看）/ sys_screen_size / sys_windows / sys_active_window /
   sys_processes / sys_clipboard_read。
2. ui_* 是真实桌面动作：ui_mouse_move / ui_mouse_click / ui_scroll / ui_key（组合键，如 "ctrl+s"）/
   ui_type_text（输入文字，中文自动走剪贴板粘贴）/ ui_window_focus（切换窗口，hwnd 来自 sys_windows）/
   ui_clipboard_write / ui_open_app / ui_kill_process。
3. 文件路径可以写绝对路径访问工作区之外；但 Windows / Program Files / ProgramData /
   盘符根目录禁止写入，工作区之外禁止递归删除目录。
4. 所有 ui_* 动作、以及工作区外的写入/删除都会弹人工确认；被拒绝就换方案或向用户说明，
   不要原样重试同一个调用。
5. 操作前建议先用 sys_screen_size 看屏幕尺寸、sys_windows 看当前窗口，
   你自己看不到屏幕画面，截屏（sys_screenshot）是给用户核对的。"""

XHS_SYSTEM_PROMPT = (
    "你是能操作小红书网页版的 AI 助手。\n"
    "规则：\n"
    "1. 只能用注册表里的 xhs_* 工具操作小红书，不允许假装完成动作。\n"
    "2. 任务开始先用 xhs_status 确认登录状态与当前页面。\n"
    "3. xhs_search 返回的笔记链接必须原样传给 xhs_read_note，不要自己拼 URL。\n"
    "4. 想找更多结果时用 xhs_scroll 多次滚动，不要重复搜索同一关键词。\n"
    "5. 发布类操作只填草稿（submit=false），是否真正发布由用户决定，不要擅自提交。\n"
    "6. 如果工具返回 not_logged_in 或触发风控验证，立刻停止并如实告知用户，"
    "不要反复重试。\n"
    "7. 回答用中文，先给结论，再列依据（标题/作者/链接/数据）。"
)


@dataclass
class RunOptions:
    mode: str = "web"
    workspace: str = "."
    readonly: bool = False
    headless: bool = False
    auto_yes: bool = False
    max_steps: int = 15
    snapshot: bool = False
    use_tools: bool = True
    use_xhs: bool = False
    xhs_headless: bool = True
    xhs_profile: str = ".agent/browser/xhs"
    # T3
    autonomous: bool = False
    save_skill: Optional[str] = None
    use_memory: bool = True
    use_critic: bool = True
    use_skills: bool = True
    max_plan_steps: int = 6
    max_replans: int = 2
    max_seconds: int = 900
    # 交互层
    confirm_writes: bool = False  # True 时连写文件也要人工确认（默认只拦高危命令）
    confirm_timeout: int = 30  # 确认弹窗超时秒数，超时自动拒绝
    pause_on_failure: bool = False  # True 时工具失败会挂起，等人选择重试/跳过/改参数
    # 本地模型通道（Ollama）
    use_local: bool = False  # True 时本地优先，超时/连接失败/5xx 自动降级到 API/网页
    ollama_base: str = ""  # 留空用 http://127.0.0.1:11434 或环境变量 OLLAMA_URL
    ollama_model: str = ""  # 留空用环境变量 OLLAMA_MODEL，默认 qwen2.5:7b-instruct
    # 电脑操控
    full_computer: bool = False  # True 时放开到整台电脑（scope="computer"）并启用 sys_*/ui_* 工具

    # ---- RAG 知识库（第一~四阶段）----
    use_rag: bool = True            # 总开关：关掉则不建引擎、不注册工具、不注入
    rag_db_path: str = ""           # 留空用 ./agent_knowledge_base 或环境变量 RAG_DB_PATH
    rag_collection: str = "agent_general_knowledge"
    rag_embed_model: str = ""       # 留空用 bge-m3（rag_engine.DEFAULT_EMBED_MODEL）或环境变量 RAG_EMBED_MODEL
    rag_ollama_url: str = ""        # 留空复用 ollama_base / OLLAMA_URL
    rag_inject: bool = True         # 第二阶段：任务开始时把检索结果注入系统提示词
    rag_tools: bool = True          # 第三阶段：注册 rag_search / rag_ingest / rag_status
    rag_top_k: int = 3
    # None = 沿用各知识库标定出的门槛（推荐）。余弦分数不可跨模型比较，
    # 写死会在换 embedding 模型后失效，故默认交给库内标定值。
    rag_min_score: Optional[float] = None
    # 第四阶段：经验库
    use_experience: bool = True     # 任务结束后把轨迹向量化沉淀，开始时语义召回
    exp_collection: str = "agent_experience"
    exp_include_failures: bool = True  # 失败任务也存（踩过的坑同样有价值）
    exp_top_k: int = 2
    exp_min_score: Optional[float] = None  # None = 沿用经验库标定门槛

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}



def describe_models(opts: RunOptions) -> List[Dict[str, Any]]:
    """给界面用的模型路由状态（不实例化浏览器，只看配置）。"""
    import os

    rows: List[Dict[str, Any]] = []
    if opts.use_local:
        rows.append(
            {
                "kind": "local",
                "name": opts.ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
                "label": "本地 Ollama（优先，失败自动降级）",
                "available": True,
                "primary": True,
            }
        )
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if opts.mode in ("api", "hybrid"):
        rows.append(
            {
                "kind": "api",
                "name": os.getenv("QWEN_MODEL", "qwen-plus"),
                "label": "API（通义千问兼容接口）",
                "available": bool(api_key),
                "primary": not rows,
            }
        )
    if opts.mode in ("web", "hybrid"):
        for name, label in (
            ("zhipu", "智谱清言（网页）"),
            ("wenxin", "文心一言（网页）"),
            ("qwen", "通义千问（网页）"),
        ):
            rows.append(
                {
                    "kind": "web",
                    "name": name,
                    "label": label,
                    "available": True,
                    "primary": not rows,
                }
            )
    return rows


def build_local_client(opts: RunOptions):
    """本地 Ollama 通道；use_local=False 时返回 None。"""
    import os

    if not opts.use_local:
        return None
    from ollama_client import OllamaClient, OllamaConfig

    return OllamaClient(
        OllamaConfig(
            base_url=opts.ollama_base or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            model=opts.ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        )
    )


def build_rag(opts: RunOptions) -> Dict[str, Any]:
    """装配知识库与经验库（第一 + 第四阶段）。

    返回 {"engine": RagEngine|None, "experience": ExperienceStore|None, "error": str}
    —— 任何环节失败都降级为 None，绝不抛异常打断任务启动：
    RAG 是增强能力，chromadb 没装或 Ollama 没起时，Agent 应当照常工作。
    """
    import os

    out: Dict[str, Any] = {"engine": None, "experience": None, "error": ""}
    if not opts.use_rag:
        out["error"] = "已通过配置关闭知识库"
        return out

    try:
        from rag_engine import DEFAULT_EMBED_MODEL, RagConfig, RagEngine, RagUnavailable, get_engine

        engine = get_engine(
            db_path=opts.rag_db_path or os.getenv("RAG_DB_PATH", "./agent_knowledge_base"),
            embed_model=opts.rag_embed_model or os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            ollama_url=(
                opts.rag_ollama_url or opts.ollama_base
                or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
            ),
        )
        engine.default_collection = opts.rag_collection or engine.default_collection
        out["engine"] = engine
    except ImportError as e:
        out["error"] = f"chromadb 未安装（pip install chromadb）：{e}"
        return out
    except Exception as e:
        out["error"] = f"知识库初始化失败：{type(e).__name__}: {e}"
        return out

    if opts.use_experience:
        try:
            from experience import ExperienceStore

            out["experience"] = ExperienceStore(
                engine=engine,
                collection=opts.exp_collection or "agent_experience",
                include_failures=opts.exp_include_failures,
            )
        except Exception as e:
            out["error"] = f"经验库初始化失败：{type(e).__name__}: {e}"
    return out


def build_api_router() -> Optional[ModelRouter]:

    import os

    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return ModelRouter(
        [
            ModelClient(
                ModelConfig(
                    name=os.getenv("QWEN_MODEL", "qwen-plus"),
                    api_url=(
                        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
                    ),
                    api_key=api_key,
                    priority=1,
                )
            )
        ]
    )


def build_web_router(headless: bool = False) -> WebModelRouter:
    return WebModelRouter(
        [
            ZhipuWebClient(headless=headless, login_wait=not headless),
            DeepseekWebClient(headless=headless, login_wait=not headless),
            WenxinWebClient(headless=headless, login_wait=not headless),
            QwenWebClient(headless=headless, login_wait=not headless),
        ]
    )


def make_confirmer(auto_yes: bool):
    """高危工具确认闸门：y=本次允许，a=本任务内全部允许，其他=拒绝。"""
    import sys

    approvals: set[str] = set()

    def confirmer(spec, args: Dict[str, Any]) -> bool:
        if auto_yes or "__all__" in approvals:
            return True
        sig = f"{spec.name}:{sorted(args)}"
        if sig in approvals:
            return True
        if not sys.stdin or not sys.stdin.isatty():
            print(f"[拒绝] {spec.name} 需要人工确认，但当前不是交互式终端", file=sys.stderr)
            return False
        import json

        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 400:
            preview = preview[:400] + "..."
        print(f"\n⚠️  高危操作：{spec.name}")
        print(f"   参数：{preview}")
        try:
            ans = input("   执行？[y] 允许一次 / [a] 本任务全部允许 / [N] 拒绝：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("a", "all"):
            approvals.add("__all__")
            return True
        if ans in ("y", "yes"):
            approvals.add(sig)
            return True
        return False

    return confirmer


def execute(
    goal: str,
    opts: Optional[RunOptions] = None,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    control: Optional[Any] = None,
    gate: Optional[Any] = None,
) -> Dict[str, Any]:
    opts = opts or RunOptions()
    t0 = time.time()
    scope = "computer" if opts.full_computer else "workspace"
    sandbox = Sandbox(opts.workspace, readonly=opts.readonly, scope=scope)
    agent_root = sandbox.root / ".agent"
    agent_root.mkdir(parents=True, exist_ok=True)

    logger = TraceLogger(agent_root)
    # 自主模式由 AutonomousAgent 自己记 task_start（带计划信息），这里不重复写
    if not opts.autonomous:
        logger.task_start(
            goal, mode=opts.mode, autonomous=opts.autonomous, workspace=str(sandbox.root)
        )

    def on_event(ev: Dict[str, Any]) -> None:
        logger.log(ev.get("event", "event"), **{k: v for k, v in ev.items() if k != "event"})
        if emit is not None:
            try:
                emit({**ev, "task_id": logger.task_id})
            except Exception:
                pass

    # ---------- 知识库 / 经验库（第一 + 第四阶段）----------
    # 放在注册表之前：第三阶段的工具要挂进 registry，第二阶段的注入要交给 Agent
    rag = build_rag(opts)
    rag_engine = rag.get("engine")
    experience = rag.get("experience")
    if rag_engine is None:
        on_event(
            {
                "event": "log",
                "level": "warn",
                "message": f"知识库未启用：{rag.get('error') or '未知原因'}（Agent 将不使用本地资料）",
            }
        )
    else:
        on_event(
            {
                "event": "rag_ready",
                "db_path": rag_engine.db_path.name if hasattr(rag_engine, "db_path") else "",
                "collection": rag_engine.default_collection,
                "embed_model": rag_engine.cfg.embed_model,
                "experience": bool(experience),
            }
        )

    # ---------- 工具注册表 ----------
    registry = build_registry(sandbox) if opts.use_tools else None
    if registry is not None:
        registry.cancel_token = control.token if control is not None else None
        registry.confirmer = make_confirmer(opts.auto_yes)
        if gate is not None:
            gate.control = control
            gate.confirm_writes = opts.confirm_writes
            gate.auto_yes = gate.auto_yes or opts.auto_yes
            registry.approver = gate

        # 第三阶段：把知识库检索注册成标准工具，Agent 遇到盲区可以自己查
        if opts.rag_tools:
            try:
                from rag_tools import register_rag_tools

                added = register_rag_tools(
                    registry,
                    rag_engine,
                    sandbox=sandbox,
                    experience=experience,
                    default_collection=(
                        rag_engine.default_collection if rag_engine else opts.rag_collection
                    ),
                    experience_collection=opts.exp_collection or "agent_experience",
                )
                on_event({"event": "rag_tools", "tools": added})
            except Exception as e:
                on_event(
                    {"event": "rag_tools_error", "error": f"{type(e).__name__}: {e}"}
                )

        if opts.use_skills:

            try:
                skills = load_skills(agent_root)
                report = register_skills(registry, skills, root=agent_root)
                if report:
                    on_event({"event": "skills", "report": report})
            except Exception as e:
                on_event({"event": "skills_error", "error": f"{type(e).__name__}: {e}"})

        if opts.use_xhs:
            xhs_tools.configure(
                profile_dir=opts.xhs_profile,
                headless=opts.xhs_headless,
                offscreen=True,
                audit=lambda ev: sandbox.audit({**ev, "kind": "xhs_action"}),
            )
            xhs_tools.register_xhs_tools(registry)

        # 电脑操控工具：只在 full_computer 作用域注册（模型看不到就不可能滥用）
        if opts.full_computer:
            try:
                computer_tools.register(registry, sandbox)
            except Exception as e:
                on_event({"event": "log", "level": "error", "message": f"电脑工具注册失败：{e}"})

    # ---------- 模型 ----------
    api_router = build_api_router() if opts.mode in ("api", "hybrid") else None
    web_router = build_web_router(opts.headless) if opts.mode in ("web", "hybrid") else None
    if opts.mode == "api" and api_router is None:
        raise SystemExit("API 模式需要设置环境变量 QWEN_API_KEY（或 DASHSCOPE_API_KEY）")
    if opts.mode == "hybrid" and api_router is None:
        import sys

        print("[提示] 未检测到 API Key，hybrid 模式将只走网页模型", file=sys.stderr)

    # 本地通道（本地优先 → 失败降级 API/网页）；降级事件实时送上界面
    local_client = build_local_client(opts)
    gateway = ModelGateway(
        api_router=api_router,
        web_router=web_router,
        local=local_client,
        on_note=lambda msg, level="warn": on_event(
            {"event": "log", "level": level, "message": str(msg)}
        ),
    )
    if local_client is not None:
        if local_client.probe():
            on_event(
                {
                    "event": "log",
                    "level": "info",
                    "message": f"本地模型就绪：{local_client.cfg.model}（本地优先，失败自动降级）",
                }
            )
        else:
            on_event(
                {
                    "event": "log",
                    "level": "warn",
                    "message": f"本地 Ollama 不可用（{local_client.cfg.base_url}），"
                    "本次任务将直接使用后备通道",
                }
            )

    snap = sandbox.snapshot() if opts.snapshot and not opts.readonly else None
    sandbox.audit(
        {
            "kind": "task_start",
            "goal": goal,
            "mode": opts.mode,
            "workspace": str(sandbox.root),
            "readonly": opts.readonly,
            "snapshot": str(snap) if snap else None,
            "autonomous": opts.autonomous,
        }
    )

    memory = MemoryStore(agent_root / "memory.db") if opts.use_memory else None

    system_prompt = XHS_SYSTEM_PROMPT if opts.use_xhs else DEFAULT_SYSTEM_PROMPT
    if opts.full_computer:
        system_prompt += COMPUTER_SYSTEM_ADDENDUM
    # 第三阶段：告诉模型知识库工具该怎么用、什么时候用
    if opts.rag_tools and (rag_engine is not None or experience is not None):
        try:
            from rag_tools import RAG_TOOL_GUIDANCE

            system_prompt += RAG_TOOL_GUIDANCE
        except Exception:
            pass

    agent = Agent(
        config=AgentConfig(
            system_prompt=system_prompt,
            max_steps=opts.max_steps,
            # 第二阶段：检索结果注入系统提示词
            use_rag=bool(opts.rag_inject and rag_engine is not None),
            rag_top_k=opts.rag_top_k,
            rag_min_score=opts.rag_min_score,
            rag_collections=[opts.rag_collection] if opts.rag_collection else None,
            # 第四阶段：语义经验召回注入
            use_experience=bool(experience is not None),
            exp_top_k=opts.exp_top_k,
            exp_min_score=opts.exp_min_score,
        ),
        api_router=api_router,
        web_router=web_router,
        registry=registry,
        gateway=gateway,
        control=control,
        rag=rag_engine if opts.rag_inject else None,
        experience=experience,
    )

    if control is not None:
        control.ask_on_failure = bool(opts.pause_on_failure)

    try:
        if opts.autonomous:
            if memory is None:
                memory = MemoryStore(agent_root / "memory.db")
            auto = AutonomousAgent(
                agent=agent,
                memory=memory,
                config=AutoConfig(
                    max_plan_steps=opts.max_plan_steps,
                    max_replans=opts.max_replans,
                    step_max_steps=max(3, min(opts.max_steps, 10)),
                    max_seconds=opts.max_seconds,
                    use_critic=opts.use_critic,
                    recall_k=3,
                    save_skill=opts.save_skill,
                ),
                logger=logger,
                on_event=emit,
                should_stop=should_stop,
                workspace=str(sandbox.root),
                control=control,
            )
            result = auto.run(goal)
        else:
            result = agent.run_task(goal, max_steps=opts.max_steps, on_event=on_event)
            result["ok"] = result.get("stop_reason") == "finished"
            result["autonomous"] = False
            result["task_id"] = logger.task_id
            result["duration_ms"] = int((time.time() - t0) * 1000)
            result["artifacts"] = [
                str(t.get("args", {}).get("path"))
                for t in result.get("trace", [])
                if t.get("ok") and t.get("tool") in ("write_file", "make_dir")
            ]
            if memory is not None:
                try:
                    memory.remember_trace(goal, result)
                except Exception:
                    pass
            logger.task_end(
                bool(result["ok"]),
                str(result.get("stop_reason")),
                str(result.get("answer") or ""),
                steps=result.get("steps"),
            )
            if emit is not None:
                emit(
                    {
                        "event": "task_end",
                        "task_id": logger.task_id,
                        "ok": result["ok"],
                        "stop_reason": result.get("stop_reason"),
                        "answer": result.get("answer"),
                    }
                )
    finally:
        if web_router is not None:
            web_router.close_all()
        if opts.use_xhs:
            xhs_tools.close()

    # ---------- 第四阶段：经验自我进化 ----------
    # 任务结束（成功或失败）后把轨迹向量化沉淀，下次相似任务能被语义召回。
    # 单轮与自主模式都走这里：autonomous.py 只负责 sqlite 台账，向量经验统一在此落库。
    if experience is not None:
        try:
            exp_res = experience.ingest(
                goal, result, task_id=str(result.get("task_id") or logger.task_id),
                workspace=str(sandbox.root),
            )
        except Exception as e:
            exp_res = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
        status = exp_res.get("status")
        if status in ("success", "unchanged"):
            on_event(
                {
                    "event": "exp_saved",
                    "task_id": exp_res.get("task_id"),
                    "chunks": exp_res.get("chunk_count", 0),
                    "status": status,
                }
            )
        else:
            on_event(
                {
                    "event": "exp_skipped",
                    "status": status,
                    "reason": str(exp_res.get("reason") or "")[:200],
                }
            )
        result["experience_saved"] = status

    sandbox.audit(
        {
            "kind": "task_end",
            "goal": goal,
            "steps": result.get("steps"),
            "stop_reason": result.get("stop_reason"),
            "ok": result.get("ok"),
            "answer": (result.get("answer") or "")[:300],
        }
    )
    result["workspace"] = str(sandbox.root)
    result["options"] = opts.to_dict()
    return result


def build_registry_for_cli(opts: RunOptions):
    """给 --tools 用：不跑任务，只看清单。"""
    scope = "computer" if opts.full_computer else "workspace"
    sandbox = Sandbox(opts.workspace, readonly=True, scope=scope)
    reg = build_registry(sandbox)
    if opts.full_computer:
        computer_tools.register(reg, sandbox)
    if opts.use_skills:
        skills = load_skills(sandbox.root / ".agent")
        register_skills(reg, skills, root=sandbox.root / ".agent")
    if opts.use_xhs:
        xhs_tools.configure(profile_dir=opts.xhs_profile, headless=opts.xhs_headless)
        xhs_tools.register_xhs_tools(reg)
    return reg
