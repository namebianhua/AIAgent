"""交互层端到端自检：不需要任何 API Key / 浏览器 / 网络。

跑法：.venv/Scripts/python.exe selfcheck_ui.py
它会在随机端口起一个 demo 控制台，然后用 HTTP 走完这些路径：
  1. 静态页 + /api/config
  2. 运行 → 确认（改参数）→ 完成
  3. 拒绝高危操作 → 模型换方案收尾
  4. 失败干预：改参数后继续
  5. 优雅停止
  6. 快照 → 改文件 → 回滚
  7. 轨迹列表 / 回放数据 / 文件树 / 技能 / 记忆
  8. SSE 事件流（seq 单调 + task_start/task_end 齐全）
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from runner import RunOptions
from server import serve

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    return ok


class Client:
    def __init__(self, base: str, token: str = "") -> None:
        self.base = base.rstrip("/")
        self.token = token

    def _req(self, path: str, method: str = "GET", body=None, timeout: float = 30):
        url = self.base + path
        data = None
        headers = {}
        if self.token:
            headers["X-Auth-Token"] = self.token
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype:
                return json.loads(raw.decode("utf-8"))
            return raw.decode("utf-8", "replace")

    def get(self, path: str, **kw):
        return self._req(path, "GET", **kw)

    def post(self, path: str, body=None, **kw):
        return self._req(path, "POST", body or {}, **kw)

    def delete(self, path: str):
        return self._req(path, "DELETE")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def drive(client: Client, goal: str, options: dict, on_confirm=None, on_action=None, timeout: float = 90):
    """跑一个任务并处理所有确认/干预，返回 (task_id, 最终视图)。"""
    resp = client.post("/api/run", {"goal": goal, "options": options})
    task_id = resp["task_id"]
    deadline = time.time() + timeout
    handled: set[str] = set()

    while time.time() < deadline:
        view = client.get(f"/api/task?id={task_id}")
        for item in view.get("pending") or []:
            cid = item.get("cid")
            if cid in handled:
                continue
            handled.add(cid)
            decision, remember, args = ("approve", False, None)
            if on_confirm is not None:
                decision, remember, args = on_confirm(item)
            client.post(
                "/api/confirm",
                {
                    "task_id": task_id,
                    "cid": cid,
                    "decision": decision,
                    "remember": remember,
                    "args": args,
                    "reason": "selfcheck",
                },
            )
        action = view.get("pending_action")
        if action and action.get("cid") not in handled:
            handled.add(action.get("cid"))
            payload = on_action(action) if on_action else {"action": "skip"}
            client.post("/api/action", {"task_id": task_id, **payload})
        if view.get("status") in ("done", "failed", "cancelled"):
            return task_id, view
        time.sleep(0.25)
    return task_id, client.get(f"/api/task?id={task_id}")


def read_sse(client: Client, task_id: str, timeout: float = 60) -> list[dict]:
    events: list[dict] = []
    url = f"{client.base}/api/stream?id={task_id}"
    if client.token:
        url += f"&token={client.token}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                events.append(ev)
                if ev.get("event") == "task_end":
                    break
    except Exception as e:  # 任务结束时服务端关流是正常的
        if not events:
            print(f"    (SSE 读取中断: {type(e).__name__}: {e})")
    return events


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="freeagent-ui-")).resolve()
    port = free_port()
    opts = RunOptions(workspace=str(workspace), mode="web", max_steps=10, confirm_timeout=20)

    print(f"[selfcheck] 工作区 {workspace}")
    print(f"[selfcheck] 启动 demo 控制台 127.0.0.1:{port}")
    httpd, manager, _ = serve(opts, port=port, host="127.0.0.1", block=False, demo=True)
    time.sleep(0.4)
    client = Client(f"http://127.0.0.1:{port}")

    try:
        # ---------- 1. 静态页 / 配置 ----------
        print("\n[1] 静态资源与配置")
        html = client.get("/")
        check("首页可访问", "Agent Console" in html or "freeAgent" in html)
        check("app.css 可访问", "card-title" in client.get("/static/app.css"))
        check("app.js 可访问", "applyEvent" in client.get("/static/app.js"))
        cfg = client.get("/api/config")
        check("配置含模型与上限", bool(cfg.get("models")) and cfg["limits"]["max_steps"] > 0)
        check("演示模式已开启", cfg["features"]["demo"] is True)

        # ---------- 2. 正常任务：确认 + 改参数 ----------
        print("\n[2] 正常运行：写文件确认（改参数）+ shell 会话级放行")
        seen: list[str] = []

        def on_confirm(item):
            seen.append(item["tool"])
            if item["tool"] == "write_file":
                return ("approve", False, {"path": "notes.md", "content": "# 由 selfcheck 改写的笔记\n"})
            return ("approve_session", True, None)

        tid, view = drive(
            client,
            "读取 config.yaml，把关键配置写成 notes.md，再用 python -V 确认环境",
            {"confirm_writes": True, "max_steps": 8},
            on_confirm=on_confirm,
        )
        check("任务完成", view["status"] == "done" and view.get("ok") is True, str(view.get("stop_reason")))
        check("确认闸门被触发（写文件 + shell）", {"write_file", "run_shell"} <= set(seen), str(seen))
        notes = workspace / "notes.md"
        check("用户改后的参数生效", notes.is_file() and "selfcheck" in notes.read_text(encoding="utf-8"))
        tools = [c["tool"] for c in view["timeline"] if c.get("kind") == "tool"]
        check("时间线工具顺序正确", tools[:3] == ["read_file", "write_file", "run_shell"], str(tools))
        check("耗时与用量有值", view["elapsed_ms"] >= 0 and view["usage"]["tool_calls"] == 3)

        # ---------- 3. 拒绝 ----------
        print("\n[3] 拒绝高危操作：模型应换方案收尾")
        tid2, view2 = drive(
            client,
            "读取 config.yaml，把关键配置写成 notes.md，再用 python -V 确认环境",
            {"confirm_writes": False, "max_steps": 6},
            on_confirm=lambda item: ("deny", False, None),
        )
        shell_cards = [c for c in view2["timeline"] if c.get("tool") == "run_shell"]
        check("shell 步骤被标记为已拒绝", bool(shell_cards) and shell_cards[0]["status"] == "denied",
              str([(c["status"]) for c in shell_cards]))
        check("拒绝后任务仍能正常结束", view2["status"] in ("done", "failed"))

        # ---------- 4. 失败干预 ----------
        print("\n[4] 失败干预：改参数后继续")
        actions: list[str] = []

        def on_action(item):
            actions.append(item["tool"])
            return {"action": "fix", "args": {"path": "config.yaml"}}

        tid3, view3 = drive(
            client,
            "失败演示：先读一个不存在的文件，然后给出结论",
            {"confirm_writes": False, "pause_on_failure": True, "max_steps": 6},
            on_action=on_action,
        )
        retried = [c for c in view3["timeline"] if c.get("tool") == "read_file"]
        check("触发了失败干预", bool(actions), str(actions))
        check("改参数后这一步成功", any(c["status"] == "done" for c in retried), str([c["status"] for c in retried]))
        check("任务最终完成", view3["status"] == "done", str(view3.get("stop_reason")))

        # ---------- 5. 停止 ----------
        print("\n[5] 优雅停止")
        resp = client.post("/api/run", {"goal": "慢任务：演示暂停与停止", "options": {"max_steps": 10}})
        slow_id = resp["task_id"]
        for _ in range(40):
            v = client.get(f"/api/task?id={slow_id}")
            if v["status"] == "running" and any(c.get("kind") == "tool" for c in v["timeline"]):
                break
            time.sleep(0.15)
        client.post(f"/api/stop?id={slow_id}")
        deadline = time.time() + 20
        final = None
        while time.time() < deadline:
            final = client.get(f"/api/task?id={slow_id}")
            if final["status"] in ("cancelled", "done", "failed"):
                break
            time.sleep(0.25)
        check("停止后任务收尾", final and final["status"] == "cancelled", final["status"] if final else "none")

        # ---------- 6. 快照 / 回滚 ----------
        print("\n[6] 快照与回滚")
        snap = client.post("/api/snapshot")
        check("创建快照", bool(snap.get("ok")), str(snap))
        csv = workspace / "data.csv"
        original = csv.read_text(encoding="utf-8")
        csv.write_text("被改坏了\n", encoding="utf-8")
        rb = client.post("/api/rollback", {"snapshot": snap.get("snapshot")})
        check("回滚成功", bool(rb.get("ok")), str(rb))
        check("文件内容已恢复", csv.read_text(encoding="utf-8") == original)
        listed = client.get("/api/snapshots")
        check("快照列表非空", len(listed.get("snapshots") or []) >= 1)

        # ---------- 7. 文件 / 轨迹 / 技能 / 记忆 ----------
        print("\n[7] 文件树 / 轨迹 / 技能 / 记忆")
        tree = client.get("/api/files?path=.&depth=2")
        names = [n["name"] for n in tree["tree"]["children"]]
        check("文件树含 config.yaml", "config.yaml" in names, str(names))
        f = client.get("/api/file?path=config.yaml")
        check("单文件读取", "freeAgent" in f.get("content", ""))
        traces = client.get("/api/traces?limit=10")["traces"]
        check("轨迹列表非空", len(traces) >= 4, f"{len(traces)} 条")
        last = client.get(f"/api/trace?id={traces[0]['task_id']}")
        kinds = [e.get("event") for e in last["events"]]
        check("回放数据含 task_start / task_end", "task_start" in kinds and "task_end" in kinds, str(kinds[:6]))
        check("技能接口可用", "skills" in client.get("/api/skills"))
        check("记忆接口可用", "total" in client.get("/api/memory") or "cards" in client.get("/api/memory"))
        trust = client.get("/api/trust")
        check("信任规则已记录", isinstance(trust.get("rules"), dict) and len(trust["rules"]) >= 1,
              str(list(trust.get("rules", {}))[:3]))

        # ---------- 8. SSE ----------
        print("\n[8] SSE 事件流")
        resp = client.post("/api/run", {"goal": "读取 config.yaml 然后结束", "options": {"max_steps": 4}})
        sse_id = resp["task_id"]
        events = read_sse(client, sse_id, timeout=40)
        names = [e.get("event") for e in events]
        seqs = [e.get("seq") for e in events if isinstance(e.get("seq"), int)]
        check("SSE 收到 task_start", "task_start" in names, str(names[:5]))
        check("SSE 收到 task_end", "task_end" in names, str(names[-5:]))
        check("SSE seq 单调递增", seqs == sorted(seqs) and len(seqs) > 3, f"{len(seqs)} 条")
        check("SSE 事件带 card id", any(e.get("card") for e in events), "")
        # 允许任务在浏览器关掉后继续跑完
        for _ in range(40):
            if client.get(f"/api/task?id={sse_id}")["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.25)

    finally:
        httpd.shutdown()
        httpd.server_close()

    ok = [r for r in results if r[1] == PASS]
    bad = [r for r in results if r[1] == FAIL]
    print("\n" + "=" * 62)
    print(f"结果：{len(ok)} 通过 / {len(bad)} 失败（共 {len(results)} 项）")
    if bad:
        for name, _, detail in bad:
            print(f"  失败：{name}  {detail}")
    print("=" * 62)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
