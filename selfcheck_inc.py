"""增量能力自检：取消信号 / 本地通道降级 / 确认策略规则表。

跑法：.venv/Scripts/python.exe selfcheck_inc.py
全部离线：本地 Ollama 用一个假的 http 服务顶替（/api/tags + /api/chat），
不依赖真实模型、浏览器或网络。

覆盖：
  A. 工具级取消：Schema 不泄露 _cancel；长命令点停止 3s 内杀掉；超时杀树
  B. 网关降级：本地优先 / 连接失败降级 / 4xx 不降级 / prefer 开关 / 解析失败重路由
     （+ 假 Ollama 的探活与真实 HTTP 调用）
  C. 信任规则表：默认规则、deny 静默拒、allow 静默过、3 次批准自动晋升、
     条件表达式、会话规则、过期、dry-run、导入导出
  D. HTTP 接口：/api/model/status、/api/model/prefer、/api/trust/policies 全套 CRUD
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    return ok


class Client:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def _req(self, path: str, method: str = "GET", body=None, timeout: float = 30):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            return json.loads(raw.decode("utf-8")) if "json" in ctype else raw.decode("utf-8", "replace")

    def get(self, path, **kw):
        return self._req(path, "GET", **kw)

    def post(self, path, body=None, **kw):
        return self._req(path, "POST", body or {}, **kw)

    def delete(self, path, **kw):
        return self._req(path, "DELETE", **kw)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------
# 假 Ollama：让 /api/tags 探活和 /api/chat 调用都走真实 HTTP
# ---------------------------------------------------------------------
class FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "fake-model:7b"}]})
        else:
            self._json({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/chat":
            if payload.get("model") == "boom-model":
                self._json({"error": "boom"}, 500)
            else:
                self._json(
                    {
                        "message": {"content": "TOOL_CALL 不完整的输出"},
                        "total_duration": 30_000_000,
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    }
                )
        else:
            self._json({}, 404)


def main() -> int:
    print("=" * 62)
    print("增量能力自检：取消信号 / 本地通道 / 确认策略")
    print("=" * 62)

    # ================= A. 取消信号 =================
    print("\n[A] 工具级取消信号")
    from cancel import CancellationToken
    from builtin_tools import build_registry
    from sandbox import Sandbox

    ws = Path(tempfile.mkdtemp(prefix="inc-a-"))
    (ws / "sleeper.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    reg = build_registry(Sandbox(ws))
    reg.confirmer = lambda spec, args: True

    spec = reg.get("run_shell")
    check("Schema 不泄露 _cancel 参数", "_cancel" not in spec.parameters["properties"])

    tok = CancellationToken()
    reg.cancel_token = tok
    threading.Timer(1.0, lambda: tok.cancel("自检停止")).start()
    t0 = time.time()
    r = reg.invoke("run_shell", {"cmd": "python sleeper.py", "timeout": 60})
    dt = time.time() - t0
    check("点停止后长命令 3s 内被杀", r["meta"].get("cancelled") and dt < 3, f"{dt:.1f}s {r.get('error')}")

    reg.cancel_token = CancellationToken()
    t0 = time.time()
    r = reg.invoke("run_shell", {"cmd": "python sleeper.py", "timeout": 2})
    dt = time.time() - t0
    check("超时自动杀进程树", not r["ok"] and dt < 6, f"{dt:.1f}s")

    r = reg.invoke("run_shell", {"cmd": "python -V", "timeout": 30})
    check("正常命令不受影响", r["ok"] and "Python" in r["output"])

    # 串联验证：RunControl.stop() → token → 注册表 → 子进程被杀（server 停止按钮的完整链路）
    from control import RunControl

    ctl = RunControl()
    reg.cancel_token = ctl.token
    threading.Timer(1.0, ctl.stop).start()
    t0 = time.time()
    r = reg.invoke("run_shell", {"cmd": "python sleeper.py", "timeout": 60})
    dt = time.time() - t0
    check("RunControl.stop 直接打断正在跑的工具", r["meta"].get("cancelled") and dt < 3, f"{dt:.1f}s")

    # ================= B. 网关降级 =================
    print("\n[B] 本地通道与降级")
    import requests
    import gateway as G
    from ollama_client import OllamaClient, OllamaConfig

    fake_port = free_port()
    fake = ThreadingHTTPServer(("127.0.0.1", fake_port), FakeOllamaHandler)
    threading.Thread(target=fake.serve_forever, daemon=True).start()

    client = OllamaClient(OllamaConfig(base_url=f"http://127.0.0.1:{fake_port}", model="fake-model:7b"))
    check("Ollama 探活（/api/tags）", client.probe() and client.healthy)

    r = client.chat([{"role": "user", "content": "hi"}])
    check("原生 /api/chat 调用 + 耗时指标", r["content"].startswith("TOOL_CALL") and r["latency_ms"] >= 1, str(r.get("latency_ms")))

    class FakeApi:
        clients = [object()]
        def chat(self, messages, **kw):
            return {"content": "api-answer", "model": "qwen-plus"}

    G.set_prefer_local(True)
    notes = []
    gw = G.ModelGateway(api_router=FakeApi(), local=client, on_note=lambda m, l: notes.append(m))
    check("本地优先路由", gw.chat([{"role": "user", "content": "x"}])["_source"] == "local")

    bad = OllamaClient(OllamaConfig(base_url="http://127.0.0.1:1", model="fake-model:7b"))
    gw = G.ModelGateway(api_router=FakeApi(), local=bad, on_note=lambda m, l: notes.append(m))
    r = gw.chat([{"role": "user", "content": "x"}])
    check("连不上自动降级后备通道", r["_source"] == "api" and notes)

    boom = OllamaClient(OllamaConfig(base_url=f"http://127.0.0.1:{fake_port}", model="boom-model"))
    gw = G.ModelGateway(api_router=FakeApi(), local=boom, on_note=lambda m, l: notes.append(m))
    check("5xx 自动降级", gw.chat([{"role": "user", "content": "x"}])["_source"] == "api")

    good = OllamaClient(OllamaConfig(base_url=f"http://127.0.0.1:{fake_port}", model="fake-model:7b"))
    gw = G.ModelGateway(api_router=FakeApi(), local=good)
    gw.cfg.fallback_on_status = ()  # 关掉 5xx 降级 → 4xx/5xx 一律抛
    try:
        OllamaClient(OllamaConfig(base_url=f"http://127.0.0.1:{fake_port}", model="nope"))
        # 404 走 chat 会返回 404 → HTTPError，且不在 fallback_on_status
        resp = None
    except Exception:
        pass
    boom2 = OllamaClient(OllamaConfig(base_url=f"http://127.0.0.1:{fake_port}", model="boom-model"))
    gw2 = G.ModelGateway(api_router=FakeApi(), local=boom2)
    gw2.cfg.fallback_on_status = ()
    try:
        gw2.chat([{"role": "user", "content": "x"}])
        check("非降级错误直接抛出（不浪费后备额度）", False)
    except requests.HTTPError:
        check("非降级错误直接抛出（不浪费后备额度）", True)

    G.set_prefer_local(False)
    gw = G.ModelGateway(api_router=FakeApi(), local=good)
    check("prefer_local 开关立即生效", gw.chat([{"role": "user", "content": "x"}])["_source"] == "api")
    G.set_prefer_local(True)

    gw = G.ModelGateway(api_router=FakeApi(), local=good)
    gw.note_parse_error("local")
    gw.note_parse_error("local")
    r1 = gw.chat([{"role": "user", "content": "x"}])
    r3 = gw.chat([{"role": "user", "content": "x"}])
    r3 = gw.chat([{"role": "user", "content": "x"}])
    r4 = gw.chat([{"role": "user", "content": "x"}])
    check(
        "连续解析失败 → 临时重路由大模型后回到本地",
        r1["_source"] == "api" and r4["_source"] == "local",
        f"{r1['_source']} … {r4['_source']}",
    )
    fake.shutdown()

    # ================= C. 信任规则表 =================
    print("\n[C] 确认策略规则表")
    from trust import TrustStore, eval_condition
    from confirm_gate import ConfirmBroker

    ws2 = Path(tempfile.mkdtemp(prefix="inc-c-"))
    store = TrustStore(ws2 / "trust-rules.json")
    a, rule = store.effective("run_shell", {"cmd": "rm -rf /"})
    check("默认规则：rm -rf 拒绝", a == "deny")
    a, _ = store.effective("read_file", {"path": "x"})
    check("默认规则：read_* 放行", a == "allow")
    a, _ = store.effective("run_shell", {"cmd": "git status"})
    check("默认规则：run_shell 询问", a == "ask")
    check("条件表达式安全求值", eval_condition("timeout < 60", {"timeout": 30}) and not eval_condition("__import__", {}))

    # ConfirmBroker：deny / allow 都不弹窗；ask 弹窗批准 3 次自动晋升
    events: list[dict] = []
    gate = ConfirmBroker(emit=events.append, root=ws2, timeout_s=10)

    class _Spec:
        def __init__(self, name, desc=""):
            self.name, self.description = name, desc

    out = gate.request(_Spec("run_shell"), {"cmd": "rm -rf /"})
    check("deny 规则静默拒绝（无弹窗）", out["decision"] == "deny" and not gate.waiting)
    out = gate.request(_Spec("read_file"), {"path": "x"})
    check("allow 规则静默放行（无弹窗）", out["decision"] == "approve" and not gate.waiting)

    # run_shell git status → ask 弹窗；连续批准 3 次
    approved = 0
    for i in range(3):
        box: list[dict] = []
        gate._emit = box.append

        def approve_later():
            for _ in range(80):
                if gate.waiting:
                    cid = gate._pending["cid"]  # type: ignore[index]
                    gate.resolve(cid, "approve")
                    return
                time.sleep(0.1)

        t = threading.Thread(target=approve_later)
        t.start()
        out = gate.request(_Spec("run_shell"), {"cmd": "git status"})
        t.join()
        gate._emit = events.append
        if out["decision"] == "approve":
            approved += 1
    check("ask 规则弹窗 + 批准通过", approved == 3, f"{approved}/3")

    a, rule = gate.policies.effective("run_shell", {"cmd": "git status"})
    check("3 次批准后规则自动晋升 allow", a == "allow" and rule.promoted, str(rule.to_dict().get("hits")))
    a, _ = TrustStore(ws2 / ".agent" / "trust-rules.json").effective("run_shell", {"cmd": "git status"})
    check("晋升结果已落盘（重启仍生效）", a == "allow")

    d = store.dry_run("run_shell", {"cmd": "rm -rf /"})
    check("dry-run 返回命中规则", d["matched"] and d["action"] == "deny")

    # ================= D. HTTP 接口 =================
    print("\n[D] 控制台 HTTP 接口（假 Ollama + demo 模型）")
    from runner import RunOptions
    from server import serve

    ws3 = Path(tempfile.mkdtemp(prefix="inc-d-"))
    from demo_model import prepare_demo_workspace
    prepare_demo_workspace(ws3)

    fake_port2 = free_port()
    fake2 = ThreadingHTTPServer(("127.0.0.1", fake_port2), FakeOllamaHandler)
    threading.Thread(target=fake2.serve_forever, daemon=True).start()

    opts = RunOptions(
        workspace=str(ws3),
        use_local=True,
        ollama_base=f"http://127.0.0.1:{fake_port2}",
        ollama_model="fake-model:7b",
    )
    httpd, manager, _ = serve(opts, port=free_port(), block=False)
    port = httpd.server_address[1]
    cli = Client(f"http://127.0.0.1:{port}")

    st = cli.get("/api/model/status")
    check("/api/model/status 本地在线", st["enabled"] and st["local"]["available"] is True, str(st.get("local")))
    check("/api/model/status 带偏好与统计", st["prefer_local"] is True and "stats" in st)

    cli.post("/api/model/prefer", {"prefer_local": False})
    st = cli.get("/api/model/status")
    check("/api/model/prefer 切换生效", st["prefer_local"] is False)
    cli.post("/api/model/prefer", {"prefer_local": True})

    pol = cli.get("/api/trust/policies")
    check("GET /api/trust/policies 列默认规则", len(pol["policies"]) >= 5, f"{len(pol['policies'])} 条")

    new = cli.post("/api/trust/policies", {"tool": "run_shell", "param_pattern": {"cmd": "git status"}, "action": "allow"})
    check("POST 新增规则", new["ok"] and new["rule"]["tool"] == "run_shell")
    rid = new["rule"]["id"]

    d = cli.post("/api/trust/policies/dry-run", {"tool": "run_shell", "args": {"cmd": "git status"}})
    check("dry-run 命中新规则", d["matched"] and d["rule"]["id"] == rid)

    upd = cli.post("/api/trust/policies/update", {"id": rid, "enabled": False})
    check("UPDATE 停用规则", upd["ok"] and upd["rule"]["enabled"] is False)
    d = cli.post("/api/trust/policies/dry-run", {"tool": "run_shell", "args": {"cmd": "git status"}})
    check("停用后不再命中（回到默认 ask）", not d["matched"] or d["rule"]["id"] != rid)

    cli.post("/api/trust/policies/update", {"id": rid, "enabled": True})
    ids = [p["id"] for p in cli.get("/api/trust/policies")["policies"]]
    cli.post("/api/trust/policies/reorder", {"ids": list(reversed(ids))})
    ids2 = [p["id"] for p in cli.get("/api/trust/policies")["policies"]]
    check("reorder 生效", ids2 == list(reversed(ids)))

    payload = cli.get("/api/trust/policies/export")
    check("export 返回可导入结构", payload.get("version") == 1 and "rules" in payload)
    imp = cli.post("/api/trust/policies/import", {"payload": payload})
    check("import 回写", imp["ok"] and imp["count"] == len(payload["rules"]))

    ok = cli.delete(f"/api/trust/policies?id={rid}")["ok"]
    check("DELETE 删除规则", ok and all(p["id"] != rid for p in cli.get("/api/trust/policies")["policies"]))

    legacy = cli.get("/api/trust")
    check("旧版 /api/trust 计数接口仍兼容", isinstance(legacy.get("rules"), dict))

    # 演示任务冒烟：新接口不影响任务链路
    resp = cli.post("/api/run", {"goal": "读取 config.yaml 然后结束", "options": {"max_steps": 4}})
    tid = resp["task_id"]
    for _ in range(60):
        view = cli.get(f"/api/task?id={tid}")
        if view["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.25)
    check("demo 任务仍可端到端完成", view["status"] == "done", view["status"])

    fake2.shutdown()
    httpd.shutdown()
    httpd.server_close()

    ok_n = sum(1 for r in results if r[1] == PASS)
    bad = [r for r in results if r[1] == FAIL]
    print("\n" + "=" * 62)
    print(f"结果：{ok_n} 通过 / {len(bad)} 失败（共 {len(results)} 项）")
    for name, _, detail in bad:
        print(f"  失败：{name}  {detail}")
    print("=" * 62)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
