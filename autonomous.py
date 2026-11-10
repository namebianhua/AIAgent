"""T3 自主任务：Plan → Execute → Critique → Replan。

和 T1 的单轮 run_task 的区别：
- 单轮：一个目标一路做到底，模型说完成就结束；
- 自主：先拆步骤，每步单独执行，再由 Critic 拿证据验收；
  验收不过就带着失败原因重新规划，超过重规划次数才认输。

诚实性约定：宁可报"未完成"，也不要因为模型自称完成就宣布成功。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent import Agent
from planner import Plan, Planner
from task_memory import MemoryStore
from tracing import TraceLogger, new_task_id

# 执行器出现这些终止原因，说明是硬失败，不交给 Critic 洗白
_HARD_FAIL = {"too_many_failures", "bad_tool_format", "loop_detected"}


@dataclass
class AutoConfig:
    max_plan_steps: int = 6
    max_replans: int = 2
    step_max_steps: int = 8
    max_seconds: int = 900
    use_critic: bool = True
    recall_k: int = 3
    save_skill: Optional[str] = None


class AutonomousAgent:
    def __init__(
        self,
        agent: Agent,
        memory: MemoryStore,
        config: Optional[AutoConfig] = None,
        logger: Optional[TraceLogger] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        workspace: str = ".",
        control: Optional[Any] = None,
    ) -> None:
        self.agent = agent
        self.memory = memory
        self.config = config or AutoConfig()
        self.logger = logger or TraceLogger(Path(workspace) / ".agent")
        self.on_event = on_event
        self.should_stop = should_stop or (lambda: False)
        self.workspace = Path(workspace).expanduser().resolve()
        self.control = control
        self.planner = Planner(gateway=agent.gateway, max_steps=self.config.max_plan_steps)

    # ---------- 事件 ----------
    def _emit(self, event: str, **fields: Any) -> None:
        record = self.logger.log(event, **fields)
        if self.on_event is not None:
            try:
                self.on_event(record)
            except Exception:
                pass

    def _forward(self, ev: Dict[str, Any], plan_step: int) -> None:
        """把执行器（Agent）的事件补上「第几个计划步骤」后上行。"""
        payload = {k: v for k, v in ev.items() if k != "event"}
        payload["plan_step"] = plan_step
        self._emit(str(ev.get("event") or "event"), **payload)

    # ---------- 主流程 ----------
    def run(self, goal: str) -> Dict[str, Any]:
        t0 = time.time()
        deadline = t0 + max(30, int(self.config.max_seconds))
        cfg = self.config

        self._emit(
            "task_start",
            goal=goal,
            autonomous=True,
            model=self.agent.gateway.describe(),
        )

        experience = ""
        recalled: List[Dict[str, Any]] = []
        if cfg.recall_k > 0:
            records = self.memory.recall(goal, k=cfg.recall_k)
            recalled = [{"goal": r.goal, "score": round(r.score, 3), "ok": r.ok} for r in records]
            experience = "\n".join(r.to_card(i) for i, r in enumerate(records, 1))
            self._emit("recall", hits=recalled)

        tools_brief = self.agent.registry.to_brief()
        plan = self.planner.make_plan(goal, tools_brief, experience)
        self._emit("plan", steps=plan.steps, criteria=plan.criteria, revision=0, parsed=plan.parsed)

        remaining: List[str] = list(plan.steps)
        executed: List[Dict[str, Any]] = []
        all_trace: List[Dict[str, Any]] = []
        replans = 0
        idx = 0
        stop_reason = "finished"
        plans: List[List[str]] = [list(plan.steps)]

        while remaining:
            if self.should_stop():
                stop_reason = "cancelled"
                break
            if self.control is not None:
                try:
                    self.control.checkpoint()
                except Exception:
                    stop_reason = "cancelled"
                    break
            if idx >= cfg.max_plan_steps:
                stop_reason = "max_plan_steps"
                break
            if time.time() > deadline:
                stop_reason = "timeout"
                break

            step = remaining.pop(0)
            idx += 1
            total = idx + len(remaining)
            self._emit("step_start", index=idx, total=total, step=step)

            step_goal = self._compose_step_goal(goal, step, idx, total, executed, plan.criteria)
            try:
                # on_event 实时转发工具事件，界面才能边跑边看（而不是一步跑完才刷出来）
                res = self.agent.run_task(
                    step_goal,
                    max_steps=cfg.step_max_steps,
                    on_event=lambda ev, _idx=idx: self._forward(ev, _idx),
                )
            except Exception as e:
                res = {
                    "answer": "",
                    "stop_reason": "executor_error",
                    "trace": [],
                    "steps": 0,
                    "model": self.planner.last_model,
                    "error": f"{type(e).__name__}: {e}",
                }

            for item in res.get("trace") or []:
                entry = dict(item)
                entry["phase"] = f"step{idx}"
                all_trace.append(entry)

            if res.get("stop_reason") == "cancelled":
                stop_reason = "cancelled"
                executed.append(
                    {
                        "index": idx,
                        "step": step,
                        "ok": False,
                        "answer": res.get("answer") or "",
                        "stop_reason": "cancelled",
                        "tools": len(res.get("trace") or []),
                        "critique": "用户停止",
                        "advice": "",
                        "superseded": True,
                    }
                )
                break

            hard_fail = res.get("stop_reason") in _HARD_FAIL or res.get("stop_reason") == "executor_error"
            evidence = self._evidence(res)
            if cfg.use_critic:
                critique = self.planner.judge(goal, step, evidence, plan.criteria)
            else:
                critique = type("C", (), {"ok": res.get("stop_reason") == "finished",
                                          "reason": "未启用校验", "advice": ""})()

            step_ok = bool(critique.ok) and not hard_fail
            self._emit(
                "critique",
                step=idx,
                ok=step_ok,
                reason=critique.reason,
                advice=getattr(critique, "advice", ""),
            )

            executed.append(
                {
                    "index": idx,
                    "step": step,
                    "ok": step_ok,
                    "answer": res.get("answer") or "",
                    "stop_reason": res.get("stop_reason"),
                    "tools": len(res.get("trace") or []),
                    "critique": getattr(critique, "reason", ""),
                    "advice": getattr(critique, "advice", ""),
                }
            )

            if step_ok:
                continue

            # 本步失败：要么重规划，要么认输
            if replans >= cfg.max_replans or idx >= cfg.max_plan_steps:
                stop_reason = "critic_rejected" if not hard_fail else str(res.get("stop_reason"))
                break

            # 这次失败的尝试会被新计划取代，标记后不计入最终成败
            executed[-1]["superseded"] = True
            replans += 1
            feedback = (
                f"步骤「{step}」未通过校验。原因：{getattr(critique, 'reason', '')}。"
                f"建议：{getattr(critique, 'advice', '') or '换一种做法'}。"
            )
            self._emit("replan", revision=replans, reason=feedback)
            new_plan = self.planner.make_plan(goal, tools_brief, experience, feedback)
            plans.append(list(new_plan.steps))
            self._emit(
                "plan",
                steps=new_plan.steps,
                criteria=new_plan.criteria,
                revision=replans,
                parsed=new_plan.parsed,
            )
            if new_plan.criteria:
                plan.criteria = new_plan.criteria
            if new_plan.steps:
                plan.steps = list(new_plan.steps)
            budget = cfg.max_plan_steps - idx
            remaining = list(new_plan.steps)[: max(1, budget)]

        effective = [s for s in executed if not s.get("superseded")]
        all_ok = bool(effective) and all(s["ok"] for s in effective) and stop_reason == "finished"
        artifacts = self._collect_artifacts(all_trace)

        if all_ok:
            answer = self._final_answer(goal, effective)
        else:
            answer = self._failure_report(effective, stop_reason)

        duration_ms = int((time.time() - t0) * 1000)
        result: Dict[str, Any] = {
            "task_id": self.logger.task_id,
            "goal": goal,
            "answer": answer,
            "ok": all_ok,
            "stop_reason": stop_reason,
            "plan": plan.steps,
            "plans": plans,
            "criteria": plan.criteria,
            "step_results": executed,
            "steps": len(all_trace),
            "trace": all_trace,
            "replans": replans,
            "artifacts": artifacts,
            "duration_ms": duration_ms,
            "model": self.planner.last_model or self.agent.gateway.describe(),
            "autonomous": True,
            "memory_hits": recalled,
            "workspace": str(self.workspace),
        }

        # 沉淀记忆（成功和失败都记：失败经验同样能避免下次踩坑）
        try:
            self.memory.remember_trace(goal, result)
        except Exception as e:
            self._emit("memory_error", error=f"{type(e).__name__}: {e}")

        # 可选：把成功轨迹固化成技能
        if cfg.save_skill and all_ok:
            try:
                from skills import crystallize, save_skill

                skill = crystallize(cfg.save_skill, goal, all_trace)
                path = save_skill(self.workspace / ".agent", skill)
                result["skill_saved"] = str(path)
                self._emit("skill_saved", name=skill.name, path=str(path))
            except Exception as e:
                result["skill_error"] = f"{type(e).__name__}: {e}"
                self._emit("skill_error", error=result["skill_error"])

        self._emit(
            "task_end",
            ok=all_ok,
            stop_reason=stop_reason,
            answer=answer,
            steps=len(all_trace),
            replans=replans,
            artifacts=artifacts,
            duration_ms=duration_ms,
        )
        return result

    # ---------- 组装 ----------
    def _compose_step_goal(
        self,
        goal: str,
        step: str,
        index: int,
        total: int,
        executed: List[Dict[str, Any]],
        criteria: str,
    ) -> str:
        lines = [f"总目标：{goal}", f"当前步骤（{index}/{total}）：{step}"]
        if criteria:
            lines.append(f"整体完成标准：{criteria}")
        if executed:
            lines.append("已完成的步骤（不要重复）：")
            for e in executed[-5:]:
                summary = (e.get("answer") or "").strip().replace("\n", " ")[:160]
                lines.append(f"  {e['index']}. {e['step']} → {summary or '（无文字结论）'}")
        lines.append("只做当前这一步，完成后直接给出本步结论。")
        return "\n".join(lines)

    @staticmethod
    def _evidence(res: Dict[str, Any]) -> str:
        parts = [
            f"执行器结论：{(res.get('answer') or '（空）')[:600]}",
            f"执行器终止原因：{res.get('stop_reason')}",
        ]
        trace = res.get("trace") or []
        if trace:
            parts.append("最近的工具观测：")
            for item in trace[-4:]:
                body = item.get("output") or item.get("error") or ""
                parts.append(
                    f"- {item.get('tool')} ok={item.get('ok')}\n  {str(body)[:500]}"
                )
        if res.get("error"):
            parts.append(f"异常：{res['error']}")
        return "\n".join(parts)

    def _final_answer(self, goal: str, executed: List[Dict[str, Any]]) -> str:
        if len(executed) == 1:
            return (executed[0].get("answer") or "").strip() or "任务已完成。"
        return self.planner.summarize(goal, executed)

    @staticmethod
    def _failure_report(executed: List[Dict[str, Any]], stop_reason: str) -> str:
        done = [e for e in executed if e["ok"]]
        failed = [e for e in executed if not e["ok"]]
        lines = [f"任务未全部完成（终止原因：{stop_reason}）。"]
        if done:
            lines.append(f"已完成 {len(done)} 步：")
            lines += [f"  {e['index']}. {e['step']}" for e in done]
        if failed:
            last = failed[-1]
            lines.append(f"卡住的一步：{last['step']}")
            lines.append(f"原因：{last.get('critique') or last.get('stop_reason')}")
            if last.get("advice"):
                lines.append(f"建议：{last['advice']}")
        return "\n".join(lines)

    def _collect_artifacts(self, trace: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in trace:
            if not item.get("ok"):
                continue
            tool = item.get("tool")
            args = item.get("args") or {}
            if tool not in ("write_file", "make_dir", "move_file"):
                continue
            rel = args.get("dst") or args.get("path")
            if not rel:
                continue
            try:
                target = Path(rel)
                if not target.is_absolute():
                    target = self.workspace / rel
                if target.exists():
                    name = target.relative_to(self.workspace).as_posix() if target.is_relative_to(self.workspace) else str(target)
                    if name not in out:
                        out.append(name)
            except Exception:
                continue
        return out
