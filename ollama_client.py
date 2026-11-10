"""本地 Ollama 通道：原生 /api/chat，不依赖 OpenAI 兼容层。

为什么走原生接口：
1. 能拿到 total_duration / eval_count 等指标，界面能显示「本次推理耗时」；
2. /api/tags 一发就知道服务在不在、模型拉没拉；
3. keep_alive 让模型常驻显存，连续调用的首 token 延迟能低一个量级。

失败时由上层 Gateway 决定是否降级到 API / 网页模型——本模块只报错，不兜底。

依赖 requests（项目 requirements.txt 已有），不新增任何包。

用法（vLLM / LM Studio 等 OpenAI 兼容服务想复用这套降级逻辑时）：
把 chat() 换成 POST /v1/chat/completions，返回值保持
{"content": str, "model": str, "latency_ms": int} 即可，Gateway 完全不用改。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    num_ctx: int = 8192
    temperature: float = 0.2
    keep_alive: str = "5m"
    timeout: int = 120  # 本地模型也要设超时，防止 GGUF 卡死整个任务线程


class OllamaClient:
    """本地 Ollama 通道。失败时由上层 Gateway 降级到 API/网页模型。"""

    name = "ollama-local"
    kind = "local"

    def __init__(self, cfg: Optional[OllamaConfig] = None) -> None:
        self.cfg = cfg or OllamaConfig()
        self._healthy: Optional[bool] = None  # None = 未探测
        self.last_latency_ms = 0

    # ---------- 核心调用 ----------
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.cfg.base_url}/api/chat",
            json={
                "model": kwargs.get("ollama_model", self.cfg.model),
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": kwargs.get("num_ctx", self.cfg.num_ctx),
                    "temperature": kwargs.get("temperature", self.cfg.temperature),
                },
                "keep_alive": self.cfg.keep_alive,
            },
            timeout=kwargs.get("timeout", self.cfg.timeout),
        )
        resp.raise_for_status()
        body = resp.json()

        content = (body.get("message") or {}).get("content", "")
        if not content:
            content = str(body)

        latency_ms = round((time.perf_counter() - t0) * 1000)
        self.last_latency_ms = latency_ms
        # Ollama 的 total_duration 是纳秒；有些兼容实现没有这个字段
        total_ns = body.get("total_duration")
        if total_ns:
            latency_ms = max(latency_ms, int(total_ns / 1_000_000))

        self._healthy = True  # 调用成功说明服务和模型都活着
        return {
            "content": content,
            "model": f"{self.name}:{self.cfg.model}",
            "latency_ms": latency_ms,
            "prompt_tokens": body.get("prompt_eval_count"),
            "completion_tokens": body.get("eval_count"),
            "raw": body,
        }

    # ---------- 健康探测 ----------
    def probe(self, timeout: float = 3.0) -> bool:
        """探测 Ollama 是否在跑、配置的模型是否已拉取。结果缓存到下次调用。"""
        try:
            r = requests.get(f"{self.cfg.base_url}/api/tags", timeout=timeout)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            self._healthy = any(self.cfg.model == m or m.startswith(self.cfg.model + ":") for m in models)
        except Exception:
            self._healthy = False
        return bool(self._healthy)

    @property
    def healthy(self) -> bool:
        if self._healthy is None:
            self.probe()
        return bool(self._healthy)

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "label": f"本地 Ollama · {self.cfg.model}",
            "base_url": self.cfg.base_url,
            "num_ctx": self.cfg.num_ctx,
        }
