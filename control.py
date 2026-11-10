"""运行控制（交互层）：停止 / 暂停 / 失败时人工干预。

Agent 的主循环在每个「可中断点」（调模型前、调工具前）调用 checkpoint()：

- 已停止  → 抛 RunCancelled，循环立刻收尾（已产生的 trace 保留，可续跑/复盘）；
- 已暂停  → 原地阻塞，直到用户点「继续」；
- 工具失败且用户开了「失败时暂停」→ ask_failure() 挂起，等三选一：
  重试 / 跳过 / 手动改参数后继续。

设计原则：控制信号只由外部线程写入，工作线程只读 + 阻塞等待，
不共享可变的执行状态，避免两个线程同时改 Agent 上下文。
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional


class RunCancelled(RuntimeError):
    """用户在可中断点停止了任务。"""


class RunControl:
    """一个任务一个实例；线程安全。"""

    def __init__(
        self,
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
        ask_on_failure: bool = False,
        action_timeout: int = 300,
    ) -> None:
        from cancel import CancellationToken

        self._emit = emit or (lambda ev: None)
        self._stop = threading.Event()
        self._paused = threading.Event()  # set == 已暂停
        self._lock = threading.Lock()
        self._wake = threading.Event()

        # 工具级取消令牌：registry.cancel_token 指到这里。
        # stop() 时立刻触发，正在跑的 run_shell / run_python 会在 0.2s 内被杀掉，
        # 不用等命令超时跑完。
        self.token = CancellationToken()

        # 失败干预
        self.ask_on_failure = bool(ask_on_failure)
        self.action_timeout = max(10, int(action_timeout))
        self._pending_action: Optional[Dict[str, Any]] = None
        self._action_result: Optional[Dict[str, Any]] = None
        self.interventions = 0

    # ---------------- 外部控制 ----------------
    def stop(self) -> None:
        self._stop.set()
        self.token.cancel("用户在界面点击停止")
        self._paused.clear()
        self._wake.set()
        # 停的时候顺手把挂起的干预请求判为「继续/放弃」，避免工作线程永远等
        with self._lock:
            if self._pending_action is not None and self._action_result is None:
                self._action_result = {"action": "abort"}

    def pause(self) -> None:
        if self._stop.is_set():
            return
        self._paused.set()
        self._emit({"event": "status", "status": "paused"})

    def resume(self) -> None:
        self._paused.clear()
        self._emit({"event": "status", "status": "running"})

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    # ---------------- 工作线程侧 ----------------
    def checkpoint(self) -> None:
        """在可中断点调用：停止则抛异常，暂停则阻塞。"""
        if self._stop.is_set():
            raise RunCancelled("任务已被用户停止")
        if not self._paused.is_set():
            return

        self._emit({"event": "status", "status": "paused"})
        while self._paused.is_set():
            if self._stop.is_set():
                raise RunCancelled("任务已被用户停止")
            time.sleep(0.15)
        if self._stop.is_set():
            raise RunCancelled("任务已被用户停止")
        self._emit({"event": "status", "status": "running"})

    def ask_failure(
        self,
        turn: int,
        tool: str,
        args: Dict[str, Any],
        error: str,
        plan_step: int = 0,
    ) -> Dict[str, Any]:
        """工具失败后询问用户怎么走。返回 {"action": retry|skip|fix|abort, "args": {...}}。"""
        if self._stop.is_set():
            return {"action": "abort"}

        cid = uuid.uuid4().hex[:8]
        with self._lock:
            self._pending_action = {"cid": cid, "tool": tool, "turn": turn}
            self._action_result = None
        self._wake.clear()

        self.interventions += 1
        self._emit(
            {
                "event": "need_action",
                "cid": cid,
                "turn": turn,
                "plan_step": plan_step,
                "tool": tool,
                "args": args,
                "error": error,
                "options": ["retry", "skip", "fix"],
                "timeout_s": self.action_timeout,
            }
        )

        deadline = time.time() + self.action_timeout
        while time.time() < deadline:
            if self._wake.wait(0.25):
                break
            if self._stop.is_set():
                break

        with self._lock:
            result = self._action_result
            self._pending_action = None
            self._action_result = None

        if self._stop.is_set():
            return {"action": "abort"}
        if result is None:
            self._emit(
                {"event": "action_result", "cid": cid, "action": "timeout",
                 "note": "超时未处理，按继续执行"}
            )
            return {"action": "continue"}

        self._emit(
            {
                "event": "action_result",
                "cid": cid,
                "action": result.get("action"),
                "args": result.get("args"),
            }
        )
        return result

    def resolve_action(self, cid: str, action: str, args: Optional[Dict[str, Any]] = None) -> bool:
        """供 HTTP 层调用：回填用户的三选一决策。"""
        with self._lock:
            if self._pending_action is None:
                return False
            if cid and cid != self._pending_action.get("cid"):
                return False
            if self._action_result is not None:
                return False
            self._action_result = {"action": action, "args": args or {}}
        self._wake.set()
        return True

    @property
    def waiting(self) -> bool:
        with self._lock:
            return self._pending_action is not None
