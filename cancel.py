"""工具级取消信号（增量改造）：让「停止」在 0.3s 内真正生效。

现状：RunControl.stop() 只设了一个标志位，Agent 循环要到下一个可中断点
（调模型前 / 调工具前）才会发现；正在跑的 run_shell / run_python 会把
整段超时跑完。本模块提供一个跨线程取消令牌：

- 工具函数声明 `_cancel: CancellationToken = None` 形参即可接入（engine
  会自动注入，且不会出现在给模型的参数 Schema 里）；
- 耗时的原语（子进程轮询、循环）周期性检查 is_cancelled；
- CancellationToken.wait() 可替代 time.sleep，睡眠也能立刻被唤醒。

Agent 循环是同步的（线程 + 同步 Playwright），所以用 threading.Event，
不引入 asyncio —— 两套取消机制混用只会更难排查。
"""
from __future__ import annotations

import functools
import inspect
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional


class CancelledError(Exception):
    """工具内部抛出；执行引擎捕获后视为「已取消」，不算工具失败。"""


class CancellationToken:
    """跨线程取消信号。外部线程写（cancel），工作线程读（wait/检查）。"""

    def __init__(self) -> None:
        self._evt = threading.Event()
        self._reason = ""

    # ---------------- 外部写 ----------------
    def cancel(self, reason: str = "用户停止") -> None:
        self._reason = reason or self._reason
        self._evt.set()

    # ---------------- 工作线程读 ----------------
    @property
    def is_cancelled(self) -> bool:
        return self._evt.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._evt.is_set():
            raise CancelledError(self._reason)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """返回 True=被取消；False=正常等满 timeout。用于替代 time.sleep。"""
        return self._evt.wait(timeout)

    @contextmanager
    def period(self, seconds: float):
        """可中断的睡眠：with token.period(1): ... 代替 time.sleep(1)。"""
        if self._evt.wait(seconds):
            raise CancelledError(self._reason)
        yield


def accepts_cancel(fn: Callable[..., Any]) -> bool:
    """工具函数是否声明了 _cancel 形参（engine 据此决定是否注入令牌）。"""
    try:
        return "_cancel" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def cancelable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """装饰器：给工具函数注入 _cancel 形参（Schema 生成会跳过下划线参数）。

    @cancelable 修饰后，工具内部可通过 kwargs["_cancel"] 拿到令牌。
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, _cancel: Optional[CancellationToken] = None, **kwargs: Any):
        if _cancel is not None:
            _cancel.raise_if_cancelled()
            kwargs["_cancel"] = _cancel
        return fn(*args, **kwargs)

    return wrapper


def cancelled_result(token: CancellationToken) -> Dict[str, Any]:
    """工具被取消时的统一返回（engine._normalize 能吃下的形状）。"""
    return {
        "ok": False,
        "output": "",
        "error": f"已取消：{token.reason}",
        "_cancelled": True,
    }
