"""模型网关：本地优先 → API → 网页 三级路由，失败按策略降级。

设计要点（踩坑后固化下来的约定）：

1. 不是所有错误都值得降级。4xx（参数错/prompt 被拒）切到网页模型只会
   再错一次还烧额度；只有「连不上 / 超时 / 5xx」才降级。
2. prefer_local 是进程级偏好（SHARED），不是每个任务一份——界面上切开关，
   正在排队/运行的下一个调用立刻生效。
3. 小模型的工具调用 JSON 出错率明显高于大模型，所以有「低置信度重路由」：
   连续 N 次解析失败后，接下来几次调用直接跳过本地走大模型，
   避免用户看着它在本地死循环。
4. 对外只暴露 chat(messages, **kwargs)，Agent / Planner 完全无感知。

网页通道没有角色概念，把角色拼进文本（沿用原实现）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GatewayConfig:
    fallback_on_status: Tuple[int, ...] = (500, 502, 503, 504)
    fallback_on_timeout: bool = True
    fallback_on_connection_error: bool = True
    # 连续 N 次工具调用 JSON 解析失败后，接下来 M 次调用跳过本地直接走大模型
    reroute_after_parse_errors: int = 2
    reroute_cooldown_calls: int = 2


class _Shared:
    """跨网关实例的进程级状态（每任务一个 Gateway 实例，状态栏需要汇总视图）。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prefer_local = True
        self.stats: Dict[str, int] = {
            "local_calls": 0,
            "api_calls": 0,
            "web_calls": 0,
            "fallbacks": 0,
            "parse_errors": 0,
        }
        self.last_used = ""
        self.last_latency_ms = 0
        self.consecutive_parse_errors = 0
        self.reroute_left = 0  # 剩余的「跳过本地」调用数

    def bump(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + n

    def set_prefer_local(self, on: bool) -> None:
        with self.lock:
            self.prefer_local = bool(on)

    def note_parse_error(self, source: str = "") -> None:
        with self.lock:
            self.stats["parse_errors"] += 1
            if source == "local":
                self.consecutive_parse_errors += 1

    def note_parse_success(self) -> None:
        with self.lock:
            self.consecutive_parse_errors = 0

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "prefer_local": self.prefer_local,
                "stats": dict(self.stats),
                "last_used": self.last_used,
                "last_latency_ms": self.last_latency_ms,
                "consecutive_parse_errors": self.consecutive_parse_errors,
                "reroute_left": self.reroute_left,
            }


SHARED = _Shared()


class _Degrade(Exception):
    """值得降级到后备通道的本地错误（超时/连接失败/5xx）。"""


def set_prefer_local(on: bool) -> None:
    SHARED.set_prefer_local(on)


def shared_snapshot() -> Dict[str, Any]:
    return SHARED.snapshot()


class ModelGateway:
    """统一模型入口：本地 Ollama → API → 网页，对外只有 chat()。"""

    def __init__(
        self,
        api_router=None,
        web_router=None,
        local: Optional[Any] = None,
        cfg: Optional[GatewayConfig] = None,
        on_note: Optional[Any] = None,
    ) -> None:
        self.api_router = api_router
        self.web_router = web_router
        self.local = local
        self.cfg = cfg or GatewayConfig()
        self.on_note = on_note  # (message, level) 回调：把降级/重路由事件送上界面

    # ---------------- 状态 ----------------
    @property
    def available(self) -> bool:
        if self.local is not None:
            return True
        if self.api_router is not None and getattr(self.api_router, "clients", None):
            return True
        if self.web_router is not None and getattr(self.web_router, "clients", None):
            return True
        return False

    def describe(self) -> str:
        parts = []
        if self.local is not None:
            parts.append(f"local:{getattr(self.local.cfg, 'model', 'ollama')}")
        if self.api_router is not None and getattr(self.api_router, "clients", None):
            names = [getattr(c, "cfg", None) and c.cfg.name for c in self.api_router.clients]
            parts.append("api:" + ",".join(n for n in names if n))
        if self.web_router is not None and getattr(self.web_router, "clients", None):
            names = [getattr(c, "name", "web") for c in self.web_router.clients]
            parts.append("web:" + ",".join(names))
        return " | ".join(parts) or "none"

    def _note(self, message: str, level: str = "warn") -> None:
        if self.on_note is not None:
            try:
                self.on_note(message, level)
            except Exception:
                pass

    # ---------------- 主入口 ----------------
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        snap = SHARED.snapshot()
        use_local = (
            self.local is not None
            and snap["prefer_local"]
            and snap["reroute_left"] == 0
        )

        if self.local is not None and snap["prefer_local"] and snap["reroute_left"] > 0:
            with SHARED.lock:
                SHARED.reroute_left -= 1
            self._note(
                f"本地模型连续 {snap['consecutive_parse_errors']} 次产出无法解析的工具调用，"
                "本轮直接交给后备模型重算",
                "info",
            )

        if use_local:
            try:
                result = self._call_local(messages, **kwargs)
                return result
            except _Degrade as e:
                SHARED.bump("fallbacks")
                self._note(f"本地模型不可用（{e}），本轮已切换到后备通道", "warn")

        return self._call_remote(messages, **kwargs)

    # ---------------- 本地通道 ----------------
    def _call_local(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        import requests

        try:
            r = self.local.chat(messages, **kwargs)
        except requests.Timeout as e:
            if not self.cfg.fallback_on_timeout:
                raise
            raise _Degrade(f"超时：{e}") from e
        except requests.ConnectionError as e:
            if not self.cfg.fallback_on_connection_error:
                raise
            raise _Degrade(f"连接失败：{type(e).__name__}") from e
        except requests.HTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", 0) or 0
            if code in self.cfg.fallback_on_status:
                raise _Degrade(f"HTTP {code}") from e
            raise  # 4xx（参数错/内容被拒）切通道只会再错一次

        r = dict(r)
        r["_source"] = "local"
        with SHARED.lock:
            SHARED.stats["local_calls"] += 1
            SHARED.last_used = f"{self.local.name}:{self.local.cfg.model}"
            SHARED.last_latency_ms = int(r.get("latency_ms") or 0)
        return r

    # ---------------- 后备通道（API → 网页）----------------
    def _call_remote(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if self.api_router is not None and getattr(self.api_router, "clients", None):
            try:
                r = self.api_router.chat(messages, **kwargs)
                r = dict(r)
                r["_source"] = "api"
                SHARED.bump("api_calls")
                with SHARED.lock:
                    SHARED.last_used = str(r.get("model") or "api")
                return r
            except Exception as e:
                if self.web_router is None or not getattr(self.web_router, "clients", None):
                    raise
                self._note(f"API 通道失败（{type(e).__name__}），降级到网页模型", "warn")
                SHARED.bump("fallbacks")

        if self.web_router is not None and getattr(self.web_router, "clients", None):
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
                if m.get("content")
            )
            web_kwargs = {k: v for k, v in kwargs.items() if k == "timeout"}
            result = self.web_router.ask(prompt, **web_kwargs)
            out = {"content": result["content"], "model": result["model"], "_source": "web"}
            SHARED.bump("web_calls")
            with SHARED.lock:
                SHARED.last_used = str(result.get("model") or "web")
            return out

        raise RuntimeError("没有可用的模型：本地未启用，且未配置 API/网页模型")

    # ---------------- 解析质量反馈（Agent 回调）----------------
    def note_parse_error(self, source: str = "") -> None:
        """Agent 在工具调用 JSON 解析失败时回调；source 用上一轮的 _source。"""
        SHARED.note_parse_error(source)
        if (
            source == "local"
            and SHARED.snapshot()["consecutive_parse_errors"] >= self.cfg.reroute_after_parse_errors
        ):
            with SHARED.lock:
                SHARED.consecutive_parse_errors = 0
                SHARED.reroute_left = self.cfg.reroute_cooldown_calls
            self._note("本地模型格式错误过多，接下来改用后备模型", "warn")

    def note_parse_success(self) -> None:
        SHARED.note_parse_success()

    # ---------------- 运维接口（状态栏用）----------------
    def status(self) -> Dict[str, Any]:
        local_ok = False
        local_model = None
        if self.local is not None:
            local_ok = self.local.probe()
            local_model = self.local.cfg.model
        return {
            "local_available": local_ok,
            "local_model": local_model,
            "prefer_local": SHARED.snapshot()["prefer_local"],
            "api_available": bool(
                self.api_router is not None and getattr(self.api_router, "clients", None)
            ),
            "web_count": len(getattr(self.web_router, "clients", []) or []),
            **SHARED.snapshot(),
        }
