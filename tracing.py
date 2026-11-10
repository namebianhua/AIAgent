"""可观测性（T3）：每个任务一份 JSONL 轨迹，可回放、可统计。

为什么不用 logging 模块：轨迹要能被程序读取（server 的 /status、memory 的
失败点提取、事后复盘），所以统一成「一行一个事件」的 JSONL，追加写、崩溃不丢。
"""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def new_task_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


class TraceLogger:
    """一个任务一个实例；线程内使用，写文件加锁由调用方串行保证。"""

    def __init__(self, root: str | os.PathLike[str], task_id: Optional[str] = None) -> None:
        self.root = Path(root).expanduser()
        self.task_id = task_id or new_task_id()
        self.dir = self.root / "traces"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.task_id}.jsonl"
        self._t0 = time.time()
        self._seq = 0

    @property
    def elapsed_ms(self) -> int:
        return int((time.time() - self._t0) * 1000)

    def log(self, event: str, **fields: Any) -> Dict[str, Any]:
        self._seq += 1
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "t_ms": self.elapsed_ms,  # 任务内累计时间（时间轴用）
            "seq": self._seq,
            "task_id": self.task_id,
            "event": event,
        }
        # 调用方可以额外传 elapsed_ms（例如工具自身耗时），不要覆盖 t_ms
        record.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass  # 观测失败不能影响主流程
        return record

    # ---------- 语义化快捷方法 ----------
    def task_start(self, goal: str, **extra: Any) -> None:
        self.log("task_start", goal=goal, **extra)

    def plan(self, steps: List[str], criteria: str = "", revision: int = 0) -> None:
        self.log("plan", steps=steps, criteria=criteria, revision=revision)

    def step_start(self, index: int, total: int, step: str) -> None:
        self.log("step_start", index=index, total=total, step=step)

    def tool(self, step: int, tool: str, args: Any, ok: bool, elapsed_ms: int,
             error: Optional[str] = None, output_len: int = 0) -> None:
        self.log(
            "tool",
            step=step,
            tool=tool,
            args=args,
            ok=ok,
            elapsed_ms=elapsed_ms,
            error=error,
            output_len=output_len,
        )

    def critique(self, step: int, ok: bool, reason: str, advice: str = "") -> None:
        self.log("critique", step=step, ok=ok, reason=reason, advice=advice)

    def task_end(self, ok: bool, stop_reason: str, answer: str, **extra: Any) -> None:
        self.log(
            "task_end",
            ok=ok,
            stop_reason=stop_reason,
            answer=(answer or "")[:1000],
            **extra,
        )


# ---------- 读取 ----------
def list_traces(root: str | os.PathLike[str], limit: int = 20) -> List[Dict[str, Any]]:
    base = Path(root).expanduser() / "traces"
    if not base.is_dir():
        return []
    files = sorted(base.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for path in files[: max(1, limit)]:
        summary: Dict[str, Any] = {
            "task_id": path.stem,
            "size": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("event") == "task_start":
                        summary["goal"] = ev.get("goal", "")
                    elif ev.get("event") == "task_end":
                        summary["ok"] = ev.get("ok")
                        summary["stop_reason"] = ev.get("stop_reason")
                        summary["answer"] = ev.get("answer", "")
                        summary["elapsed_ms"] = ev.get("t_ms", ev.get("elapsed_ms"))
                    elif ev.get("event") in ("tool_end", "tool"):
                        summary["tools"] = summary.get("tools", 0) + 1
        except Exception:
            continue
        out.append(summary)
    return out


def read_trace(root: str | os.PathLike[str], task_id: str) -> List[Dict[str, Any]]:
    path = Path(root).expanduser() / "traces" / f"{task_id}.jsonl"
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def render_events(events: List[Dict[str, Any]]) -> str:
    """把轨迹渲染成人能读的时间线。"""
    if not events:
        return "（轨迹为空）"
    lines: List[str] = []
    for ev in events:
        kind = ev.get("event")
        t = ev.get("t_ms", ev.get("elapsed_ms", 0)) / 1000.0
        if kind == "task_start":
            lines.append(f"[{t:6.1f}s] ▶ 任务：{ev.get('goal', '')}")
        elif kind == "plan":
            rev = ev.get("revision") or 0
            tag = "计划" if not rev else f"重规划#{rev}"
            lines.append(f"[{t:6.1f}s] ☰ {tag}：")
            for i, s in enumerate(ev.get("steps") or [], 1):
                lines.append(f"          {i}. {s}")
            if ev.get("criteria"):
                lines.append(f"          完成标准：{ev.get('criteria')}")
        elif kind == "step_start":
            lines.append(
                f"[{t:6.1f}s] ▷ 步骤 {ev.get('index')}/{ev.get('total')}：{ev.get('step', '')}"
            )
        elif kind == "tool_start":
            lines.append(
                f"[{t:6.1f}s]   ▶ {ev.get('tool')} {json.dumps(ev.get('args') or {}, ensure_ascii=False)[:120]}"
            )
        elif kind == "tool":
            flag = "OK  " if ev.get("ok") else "FAIL"
            lines.append(
                f"[{t:6.1f}s]   [{flag}] {ev.get('tool')} "
                f"({ev.get('elapsed_ms')}ms)"
            )
            if not ev.get("ok") and ev.get("error"):
                lines.append(f"          ! {str(ev.get('error'))[:160]}")
        elif kind == "critique":
            flag = "通过" if ev.get("ok") else "不通过"
            lines.append(f"[{t:6.1f}s] ✓ 校验{flag}：{ev.get('reason', '')}")
            if ev.get("advice"):
                lines.append(f"          建议：{ev.get('advice')}")
        elif kind == "recall":
            hits = ev.get("hits") or []
            if not hits:
                lines.append(f"[{t:6.1f}s] ✦ 无相似历史任务")
            else:
                lines.append(f"[{t:6.1f}s] ✦ 命中 {len(hits)} 条历史经验：")
                for h in hits:
                    flag = "成功" if h.get("ok") else "失败"
                    lines.append(
                        f"          {h.get('score')} | {flag} | {str(h.get('goal') or '')[:60]}"
                    )
        elif kind == "replan":
            lines.append(
                f"[{t:6.1f}s] ↺ 重规划#{ev.get('revision')}：{str(ev.get('reason') or '')[:120]}"
            )
        elif kind == "skills":
            report = ev.get("report") or {}
            lines.append(f"[{t:6.1f}s] ⚙ 技能加载 {len(report)} 个")
            for name, state in list(report.items())[:10]:
                lines.append(f"          {name}: {state}")
        elif kind == "skill_saved":
            lines.append(f"[{t:6.1f}s] ★ 技能已固化：{ev.get('name')} -> {ev.get('path')}")
        elif kind == "task_end":
            flag = "成功" if ev.get("ok") else f"失败({ev.get('stop_reason')})"
            lines.append(f"[{t:6.1f}s] ■ 结束：{flag}")
            if ev.get("answer"):
                lines.append(f"          {str(ev.get('answer'))[:300]}")
        else:
            lines.append(f"[{t:6.1f}s] · {kind}: {json.dumps(ev, ensure_ascii=False)[:160]}")
    return "\n".join(lines)
