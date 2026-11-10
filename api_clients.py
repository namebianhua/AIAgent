from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import requests


@dataclass
class ModelConfig:
    name: str
    api_url: str
    api_key: str = ""
    max_tokens: int = 4096
    timeout: int = 30
    priority: int = 1


class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload = {
            "model": self.cfg.name,
            "messages": messages,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", self.cfg.max_tokens),
        }

        resp = requests.post(
            self.cfg.api_url,
            json=payload,
            headers=headers,
            timeout=kwargs.get("timeout", self.cfg.timeout),
        )
        resp.raise_for_status()
        body = resp.json()

        if "choices" in body and body["choices"]:
            content = body["choices"][0]["message"]["content"]
        else:
            content = json.dumps(body, ensure_ascii=False)

        return {
            "content": content,
            "model": self.cfg.name,
            "raw": body,
        }


class ModelRouter:
    def __init__(self, clients: List[ModelClient]):
        self.clients = sorted(clients, key=lambda c: c.cfg.priority)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        last_error = None
        for client in self.clients:
            try:
                return client.chat(messages, **kwargs)
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"所有 API 模型调用失败，最后错误: {last_error}")