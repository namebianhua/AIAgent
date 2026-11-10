from __future__ import annotations

from typing import List, Optional
from web_clients import BaseWebModelClient


class WebModelRouter:
    """免 Key 网页版多模型路由器"""

    def __init__(self, clients: Optional[List[BaseWebModelClient]] = None):
        self.clients: List[BaseWebModelClient] = clients or []

    def add(self, client: BaseWebModelClient):
        self.clients.append(client)

    def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> dict:
        if model:
            target = next((c for c in self.clients if c.name == model), None)
            if target is None:
                raise ValueError(f"未找到网页模型: {model}")
            answer = target.ask(prompt, timeout=timeout)
            return {"content": answer, "model": target.name}

        last_error = None
        for client in self.clients:
            try:
                answer = client.ask(prompt, timeout=timeout)
                return {"content": answer, "model": client.name}
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"所有网页模型都失败，最后错误: {last_error}")

    def close_all(self):
        for client in self.clients:
            client.close_session()