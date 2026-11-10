"""演示用脚本化模型：不联网、不启浏览器，也能把整条交互链路跑通。

用途有两个：
1. 第一次打开控制台时，能立刻看到真实的时间线、确认弹窗、失败干预，而不是等模型；
2. selfcheck_ui.py 用它做端到端回归，CI 里不依赖任何外部服务。

它实现的是和 ModelGateway 相同的最小接口：chat(messages) / describe() / available。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

DEMO_CONFIG = """# 演示用配置文件
service:
  name: freeAgent
  mode: web
  max_steps: 15
features:
  tools: true
  memory: true
  critic: true
"""

DEMO_CSV = """name,steps,ok
read_file,3,true
write_file,2,true
run_shell,1,false
"""


def prepare_demo_workspace(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    if not config.is_file():
        config.write_text(DEMO_CONFIG, encoding="utf-8")
    csv = root / "data.csv"
    if not csv.is_file():
        csv.write_text(DEMO_CSV, encoding="utf-8")
    readme = root / "README.md"
    if not readme.is_file():
        readme.write_text(
            "# 演示工作区\n\n这是 `--demo` 模式自动创建的工作区，随便改，删了会重建。\n",
            encoding="utf-8",
        )
    return root


def _tool_call(tool: str, **args: Any) -> str:
    return "TOOL_CALL: " + json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


class ScriptedGateway:
    """按对话轮次返回预置动作；规划/校验/总结按 system 提示词分流。"""

    name = "demo-script"

    def __init__(self, delay: float = 0.6) -> None:
        self.delay = delay
        self.calls = 0

    # ModelGateway 兼容
    @property
    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return f"demo:{self.name}"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        system = str((messages[0] or {}).get("content") or "")
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content") or "")
                break
        # 目标始终是「第一条 user 消息」；后面的 user 消息是工具结果回灌
        goal = last_user
        for m in messages:
            if m.get("role") == "user":
                goal = str(m.get("content") or "")
                break

        if "任务规划器" in system:
            return self._plan(last_user)
        if "结果校验员" in system:
            return self._critique(last_user)
        if "任务总结员" in system:
            return {"content": self._summary(last_user), "model": self.name}
        return self._agent_turn(messages, goal)

    # ---------- 子角色 ----------
    def _plan(self, goal: str) -> Dict[str, Any]:
        plan = {
            "steps": ["读取工作区里的 config.yaml", "把要点写入 notes.md"],
            "success_criteria": "notes.md 里能看到 config.yaml 的关键配置",
        }
        return {"content": json.dumps(plan, ensure_ascii=False), "model": self.name}

    def _critique(self, evidence: str) -> Dict[str, Any]:
        ok = "FAIL" not in evidence and "错误" not in evidence
        body = {"ok": ok, "reason": "工具返回了可验证的观测" if ok else "证据里出现失败", "advice": "" if ok else "换个路径重试"}
        return {"content": json.dumps(body, ensure_ascii=False), "model": self.name}

    def _summary(self, records: str) -> str:
        return "演示任务完成：已读取 config.yaml 的关键配置，并把要点写入 notes.md。"

    # ---------- 主循环 ----------
    def _agent_turn(self, messages: List[Dict[str, str]], goal: str) -> Dict[str, Any]:
        turns = sum(1 for m in messages if m.get("role") == "assistant")

        if "慢" in goal:
            time.sleep(2.0)
        elif self.delay:
            time.sleep(self.delay)

        # 自主模式：每一步是独立的一次执行，只做「当前步骤」这一件事
        if "当前步骤" in goal:
            step_text = goal.split("当前步骤", 1)[1]
            if "写入" in step_text:
                seq_step = [
                    _tool_call("write_file", path="notes.md",
                               content=f"# 要点\n\n- service.name = freeAgent\n- mode = web\n- 时间：{time.strftime('%H:%M:%S')}\n"),
                    "已把 config.yaml 的要点写入 notes.md。",
                ]
            elif "读取" in step_text:
                seq_step = [
                    _tool_call("read_file", path="config.yaml"),
                    "config.yaml 的要点：service.name=freeAgent，mode=web，max_steps=15。",
                ]
            else:
                seq_step = ["这一步没有需要执行的动作，直接给出结论。"]
            return {"content": seq_step[min(turns, len(seq_step) - 1)], "model": self.name}

        # 失败演示：第一次调用故意读一个不存在的文件，触发「重试 / 跳过 / 修正」
        if "失败" in goal:
            seq = [
                _tool_call("read_file", path="不存在的文件.txt"),
                _tool_call("read_file", path="config.yaml"),
                "演示结束：第一次读取失败，按用户的选择继续后完成了任务。",
            ]
            return {"content": seq[min(turns, len(seq) - 1)], "model": self.name}

        if "慢" in goal:
            seq_slow = [_tool_call("list_dir", path=".") for _ in range(5)]
            seq_slow.append("慢任务结束：用于演示暂停与停止。")
            return {"content": seq_slow[min(turns, len(seq_slow) - 1)], "model": self.name}

        seq_default = [
            _tool_call("read_file", path="config.yaml"),
            _tool_call(
                "write_file",
                path="notes.md",
                content=(
                    "# 演示笔记\n\n"
                    "- 服务名：freeAgent\n"
                    "- 运行模式：web\n"
                    "- 最大步数：15\n"
                    f"- 本次演示时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                ),
            ),
            _tool_call("run_shell", cmd="python -V"),
            "演示任务完成：已读取 config.yaml，写入 notes.md，并执行了一条 shell 命令。",
        ]
        return {"content": seq_default[min(turns, len(seq_default) - 1)], "model": self.name}


# ---------------------------------------------------------------------
# 演示执行入口：与 runner.execute 同签名，只把「模型」换成脚本
# ---------------------------------------------------------------------
def demo_execute(
    goal: str,
    opts: Any = None,
    emit: Any = None,
    should_stop: Any = None,
    control: Any = None,
    gate: Any = None,
) -> Dict[str, Any]:
    from agent import Agent, AgentConfig
    from autonomous import AutoConfig, AutonomousAgent
    from builtin_tools import build_registry
    import computer_tools
    from runner import COMPUTER_SYSTEM_ADDENDUM, DEFAULT_SYSTEM_PROMPT, RunOptions
    from sandbox import Sandbox
    from skills import load_skills, register_skills
    from task_memory import MemoryStore
    from tracing import TraceLogger

    opts = opts or RunOptions()
    t0 = time.time()
    scope = "computer" if opts.full_computer else "workspace"
    sandbox = Sandbox(opts.workspace, readonly=opts.readonly, scope=scope)
    agent_root = sandbox.root / ".agent"
    agent_root.mkdir(parents=True, exist_ok=True)
    logger = TraceLogger(agent_root)
    if not opts.autonomous:
        logger.task_start(goal, mode=opts.mode, autonomous=opts.autonomous, workspace=str(sandbox.root))
    channel = ScriptedGateway()

    def on_event(ev: Dict[str, Any]) -> None:
        logger.log(ev.get("event", "event"), **{k: v for k, v in ev.items() if k != "event"})
        if emit is not None:
            try:
                emit(ev)
            except Exception:
                pass

    registry = build_registry(sandbox)
    registry.approver = gate
    registry.cancel_token = control.token if control is not None else None
    if gate is not None:
        gate.control = control
        gate.confirm_writes = opts.confirm_writes
    if opts.use_skills:
        try:
            register_skills(registry, load_skills(agent_root), root=agent_root)
        except Exception:
            pass
    if opts.full_computer:
        try:
            computer_tools.register(registry, sandbox)
        except Exception:
            pass

    system_prompt = DEFAULT_SYSTEM_PROMPT + (COMPUTER_SYSTEM_ADDENDUM if opts.full_computer else "")
    agent = Agent(
        config=AgentConfig(system_prompt=system_prompt, max_steps=opts.max_steps),
        registry=registry,
        gateway=channel,
        control=control,
    )
    if control is not None:
        control.ask_on_failure = bool(opts.pause_on_failure)

    if opts.autonomous:
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
            if t.get("ok") and t.get("tool") == "write_file"
        ]
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
                    "artifacts": result["artifacts"],
                }
            )

    result["workspace"] = str(sandbox.root)
    result["options"] = opts.to_dict()
    return result
