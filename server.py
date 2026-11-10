"""HTTP 控制台服务（交互层）：把 Agent 的执行过程变成「看得懂、管得住」的界面。

为什么还是只用标准库：这台机器 pip/npm 的 CDN 都不稳，交互层不该成为新的依赖风险；
http.server + SSE 足够支撑「自己用 / 几个人用」，还省掉了 Node 构建链。

结构：
  Job         一个任务的完整状态（时间线、待确认、待处理、用量、产物）
  TaskManager 单工作线程串行执行（避免两个任务抢浏览器会话），HTTP 线程只读状态
  路由         /api/* 见 README「接口契约」；/ 与 /static/* 是控制台静态资源

安全默认值：只监听 127.0.0.1；对外监听必须带 token，否则拒绝启动。
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from confirm_gate import TRUST_FILE, ConfirmBroker
from computer_tools import computer_tool_names
from control import RunControl
from runner import RunOptions, build_local_client, describe_models, execute
from skills import delete_skill, load_skills
from task_memory import MemoryStore
from tracing import list_traces, read_trace
import gateway as gateway_mod

SERVER_VERSION = "freeAgent/4.0-ui"
WEB_DIR = Path(__file__).resolve().parent / "web"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".agent", ".idea", ".mypy_cache"}


# =====================================================================
# 任务
# =====================================================================
class Job:
    """一个任务的全部可观测状态；emit() 是唯一写入口（工作线程调用）。"""

    def __init__(self, job_id: str, goal: str, options: RunOptions, control: RunControl, gate: ConfirmBroker) -> None:
        self.id = job_id
        self.goal = goal
        self.options = options
        self.control = control
        self.gate = gate

        self.status = "queued"  # queued|running|paused|waiting_confirm|waiting_action|done|failed|cancelled
        self.created = datetime.now().isoformat(timespec="seconds")
        self.started = ""
        self.finished = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.answer = ""
        self.error = ""
        self.stop_reason = ""
        self.ok: Optional[bool] = None
        self.model = ""
        self.workspace = str(Path(options.workspace).expanduser().resolve())
        self.plan: List[str] = []
        self.criteria = ""
        self.timeline: List[Dict[str, Any]] = []
        self.pending_confirms: Dict[str, Dict[str, Any]] = {}
        self.pending_action: Optional[Dict[str, Any]] = None
        self.usage: Dict[str, Any] = {"chars": 0, "est_tokens": 0, "turns": 0, "tool_calls": 0}
        self.artifacts: List[str] = []
        self.last_error = ""
        self.result: Optional[Dict[str, Any]] = None
        self.ended = False  # 执行器可能已经发过 task_end，避免重复

        self.history: List[Dict[str, Any]] = []
        self.seq = 0
        self._card_seq = 0
        self._open: Dict[Tuple[int, int, str], str] = {}
        self._current_card = ""
        self._subs: List["queue.Queue[Dict[str, Any]]"] = []
        self._lock = threading.Lock()

    # ---------------- 事件 ----------------
    def emit(self, event: Dict[str, Any]) -> None:
        ev = dict(event)
        with self._lock:
            self.seq += 1
            ev["seq"] = self.seq
            ev["task_id"] = self.id
            ev["at"] = time.time()
            ev["t_ms"] = int((ev["at"] - (self.started_at or ev["at"])) * 1000)
            try:
                self._ingest(ev)
            except Exception as e:  # 状态归并失败不能影响任务执行
                ev["ingest_error"] = f"{type(e).__name__}: {e}"
            self.history.append(ev)
            if len(self.history) > 800:
                del self.history[:200]
            for q in list(self._subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass

    def _card(self, kind: str, **fields: Any) -> Dict[str, Any]:
        self._card_seq += 1
        card = {
            "id": f"c{self._card_seq}",
            "kind": kind,
            "ts": time.time(),
            "status": "running",
        }
        card.update(fields)
        self.timeline.append(card)
        return card

    def _find(self, card_id: str) -> Optional[Dict[str, Any]]:
        for card in reversed(self.timeline):
            if card.get("id") == card_id:
                return card
        return None

    def _ingest(self, ev: Dict[str, Any]) -> None:
        kind = ev.get("event")
        key = (int(ev.get("plan_step") or 0), int(ev.get("turn") or 0), str(ev.get("tool") or ""))

        if kind == "task_start":
            self.status = "running"
            self.started = self.started or datetime.now().isoformat(timespec="seconds")
            self.started_at = self.started_at or ev["at"]
            self.model = ev.get("model") or self.model
            self.workspace = ev.get("workspace") or self.workspace

        elif kind == "plan":
            self.plan = list(ev.get("steps") or [])
            self.criteria = str(ev.get("criteria") or self.criteria)
            card = self._card(
                "plan",
                steps=self.plan,
                criteria=self.criteria,
                revision=int(ev.get("revision") or 0),
                status="done",
            )
            ev["card"] = card["id"]

        elif kind == "step_start":
            # 上一步没有显式的结束事件，靠下一步开始来收尾
            for prev in reversed(self.timeline):
                if prev.get("kind") == "step" and prev.get("status") == "running":
                    prev["status"] = "done"
                    break
            card = self._card(
                "step",
                index=ev.get("index"),
                total=ev.get("total"),
                text=str(ev.get("step") or ""),
                plan_step=int(ev.get("index") or 0),
            )
            ev["card"] = card["id"]

        elif kind == "tool_start":
            card = self._card(
                "tool",
                tool=str(ev.get("tool")),
                args=ev.get("args") or {},
                turn=ev.get("turn"),
                plan_step=ev.get("plan_step") or 0,
                dangerous=bool(ev.get("dangerous")),
                require_confirm=bool(ev.get("require_confirm")),
            )
            self._open[key] = card["id"]
            self._current_card = card["id"]
            self.usage["tool_calls"] = int(self.usage.get("tool_calls", 0)) + 1
            ev["card"] = card["id"]

        elif kind == "tool_end":
            card_id = self._open.pop(key, None)
            card = self._find(card_id) if card_id else None
            if card is None:
                card = self._card(
                    "tool",
                    tool=str(ev.get("tool")),
                    args=ev.get("args") or {},
                    turn=ev.get("turn"),
                    plan_step=ev.get("plan_step") or 0,
                )
            ok = bool(ev.get("ok"))
            if ev.get("denied"):
                status = "denied"
            elif ev.get("cancelled"):
                status = "cancelled"
            elif ev.get("skipped"):
                status = "skipped"
            else:
                status = "done" if ok else "failed"
            # 人工改参数重试成功后，第一次的错误仍然值得留着（复盘时最有用）
            first_error = card.get("firstError")
            if int(ev.get("attempt") or 1) > 1 and card.get("error") and not first_error:
                first_error = card.get("error")
            card.update(
                {
                    "firstError": first_error,
                    "status": status,
                    "ok": ok,
                    "output": ev.get("output") or "",
                    "output_len": ev.get("output_len"),
                    "error": ev.get("error"),
                    "elapsed_ms": ev.get("elapsed_ms"),
                    "attempt": ev.get("attempt") or 1,
                    "args": ev.get("args") or card.get("args"),
                }
            )
            if not ok and not ev.get("skipped"):
                self.last_error = str(ev.get("error") or "")[:300]
            if self._current_card == card["id"]:
                self._current_card = ""
            ev["card"] = card["id"]

        elif kind == "critique":
            for prev in reversed(self.timeline):
                if prev.get("kind") == "step" and prev.get("status") == "running":
                    prev["status"] = "done" if ev.get("ok") else "failed"
                    break
            card = self._card(
                "critique",
                ok=bool(ev.get("ok")),
                reason=ev.get("reason") or "",
                advice=ev.get("advice") or "",
                plan_step=ev.get("step") or ev.get("plan_step") or 0,
                status="done" if ev.get("ok") else "failed",
            )
            ev["card"] = card["id"]

        elif kind == "replan":
            card = self._card(
                "replan",
                revision=ev.get("revision"),
                reason=ev.get("reason") or "",
                status="done",
            )
            ev["card"] = card["id"]

        elif kind == "recall":
            hits = ev.get("hits") or []
            card = self._card("recall", hits=hits, status="done")
            ev["card"] = card["id"]

        elif kind in ("log", "skills", "skill_saved", "memory_error", "skills_error"):
            card = self._card(
                "log",
                level=ev.get("level") or ("error" if kind.endswith("error") else "info"),
                message=ev.get("message") or ev.get("error") or kind,
                status="done",
            )
            ev["card"] = card["id"]

        elif kind == "need_confirm":
            self.pending_confirms[str(ev.get("cid"))] = {
                "cid": ev.get("cid"),
                "tool": ev.get("tool"),
                "args": ev.get("args"),
                "preview": ev.get("preview"),
                "danger": ev.get("danger"),
                "pattern": ev.get("pattern"),
                "timeout_s": ev.get("timeout_s"),
                "deadline": ev["at"] + float(ev.get("timeout_s") or 30),
            }
            self.status = "waiting_confirm"
            card = self._find(self._current_card) or (self.timeline[-1] if self.timeline else None)
            if card is not None:
                card["status"] = "pending"
                card["preview"] = ev.get("preview")
                ev["card"] = card["id"]

        elif kind == "confirm_result":
            self.pending_confirms.pop(str(ev.get("cid")), None)
            if not self.pending_confirms and self.status == "waiting_confirm":
                self.status = "running"
            card = self._find(self._current_card) or (self.timeline[-1] if self.timeline else None)
            if card is not None and card.get("status") == "pending":
                card["status"] = "denied" if ev.get("decision") == "deny" else "running"
                card["decision"] = ev.get("decision")
                ev["card"] = card["id"]

        elif kind == "need_action":
            self.pending_action = {
                "cid": ev.get("cid"),
                "tool": ev.get("tool"),
                "args": ev.get("args"),
                "error": ev.get("error"),
                "options": ev.get("options") or ["retry", "skip", "fix"],
                "deadline": ev["at"] + float(ev.get("timeout_s") or 300),
            }
            self.status = "waiting_action"
            card = self._find(self._current_card) or (self.timeline[-1] if self.timeline else None)
            if card is not None:
                card["status"] = "interrupted"
                ev["card"] = card["id"]

        elif kind == "action_result":
            self.pending_action = None
            if self.status == "waiting_action":
                self.status = "running"

        elif kind == "usage":
            self.usage["chars"] = ev.get("chars") or self.usage["chars"]
            self.usage["est_tokens"] = ev.get("est_tokens") or self.usage["est_tokens"]
            self.usage["turns"] = ev.get("turn") or self.usage["turns"]
            if ev.get("model"):
                self.model = ev.get("model")

        elif kind == "status":
            state = str(ev.get("status") or "")
            if state in ("paused", "running"):
                self.status = state

        elif kind == "task_end":
            self.ended = True
            self.status = "cancelled" if ev.get("stop_reason") == "cancelled" else (
                "done" if ev.get("ok") else "failed"
            )
            self.ok = bool(ev.get("ok"))
            self.answer = str(ev.get("answer") or "")
            self.stop_reason = str(ev.get("stop_reason") or "")
            self.artifacts = list(ev.get("artifacts") or self.artifacts)
            self.finished = datetime.now().isoformat(timespec="seconds")
            self.finished_at = ev["at"]
            self.pending_confirms.clear()
            self.pending_action = None
            for card in self.timeline:
                if card.get("status") in ("running", "pending", "interrupted"):
                    card["status"] = "done" if self.ok else "failed"

    # ---------------- 订阅 ----------------
    def subscribe(self) -> Tuple["queue.Queue[Dict[str, Any]]", List[Dict[str, Any]]]:
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.append(q)
            history = list(self.history)
        return q, history

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def broadcast(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass

    # ---------------- 视图 ----------------
    def elapsed_ms(self) -> int:
        if not self.started_at:
            return 0
        end = self.finished_at or time.time()
        return int((end - self.started_at) * 1000)

    def to_view(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "goal": self.goal,
                "status": self.status,
                "created": self.created,
                "started": self.started,
                "finished": self.finished,
                "model": self.model,
                "workspace": self.workspace,
                "autonomous": self.options.autonomous,
                "plan": self.plan,
                "criteria": self.criteria,
                "timeline": list(self.timeline),
                "pending": list(self.pending_confirms.values()),
                "pending_action": self.pending_action,
                "usage": dict(self.usage),
                "answer": self.answer,
                "ok": self.ok,
                "stop_reason": self.stop_reason,
                "error": self.error,
                "artifacts": list(self.artifacts),
                "last_error": self.last_error,
                "elapsed_ms": self.elapsed_ms(),
                "limits": {
                    "max_steps": self.options.max_steps,
                    "max_plan_steps": self.options.max_plan_steps,
                    "max_seconds": self.options.max_seconds,
                },
                "options": {
                    "readonly": self.options.readonly,
                    "auto_yes": self.options.auto_yes,
                    "confirm_writes": self.options.confirm_writes,
                    "pause_on_failure": self.options.pause_on_failure,
                    "snapshot": self.options.snapshot,
                    "full_computer": self.options.full_computer,
                },
            }


class TaskManager:
    """单工作线程串行执行：网页模型/浏览器会话不能并发抢。"""

    def __init__(
        self,
        base_options: RunOptions,
        token: str = "",
        demo: bool = False,
        executor: Optional[Any] = None,
    ) -> None:
        self.base = base_options
        self.token = token
        self.demo = demo
        self.executor = executor or execute
        self.jobs: Dict[str, Job] = {}
        # 共享的本地通道探针：/api/model/status 用它探活，不在任务线程里建连接
        self.ollama = build_local_client(base_options)
        self._order: List[str] = []
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="agent-worker", daemon=True)
        self._worker.start()

    # ---------- 提交 ----------
    def submit(
        self,
        goal: str,
        autonomous: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Job:
        opts = RunOptions(**{**self.base.to_dict(), **(overrides or {})})
        if autonomous is not None:
            opts.autonomous = bool(autonomous)
        with self._lock:
            job_id = datetime.now().strftime("%H%M%S") + f"-{len(self.jobs) + 1:03d}"
            while job_id in self.jobs:
                job_id += "x"
            control = RunControl(ask_on_failure=opts.pause_on_failure)
            gate = ConfirmBroker(
                emit=lambda ev: None,  # 下面换成 job.emit
                root=opts.workspace,
                timeout_s=opts.confirm_timeout,
                auto_yes=opts.auto_yes,
                confirm_writes=opts.confirm_writes,
                control=control,
                scope="computer" if opts.full_computer else "workspace",
            )
            job = Job(job_id, goal, opts, control, gate)
            gate._emit = job.emit
            control._emit = job.emit
            self.jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def recent(self, n: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:n]
        out = []
        for i in ids:
            job = self.jobs.get(i)
            if job is None:
                continue
            view = job.to_view()
            view.pop("timeline", None)
            out.append(view)
        return out

    # ---------- 执行 ----------
    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._run(job)
            except Exception as e:  # 工作线程绝不能死
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.emit({"event": "log", "level": "error", "message": job.error})
                job.emit(
                    {
                        "event": "task_end",
                        "ok": False,
                        "stop_reason": "worker_error",
                        "answer": job.error,
                    }
                )
            finally:
                job.broadcast({"event": "__close__", "task_id": job.id})

    def _run(self, job: Job) -> None:
        if job.control.should_stop():
            job.status = "cancelled"
            job.emit({"event": "task_end", "ok": False, "stop_reason": "cancelled", "answer": "任务在排队时被取消"})
            return

        job.status = "running"
        job.started = datetime.now().isoformat(timespec="seconds")
        job.started_at = time.time()
        if not job.options.autonomous:
            # 自主模式的 task_start 由 AutonomousAgent 自己发（带计划信息），这里别重复
            job.emit(
                {
                    "event": "task_start",
                    "goal": job.goal,
                    "autonomous": False,
                    "model": (describe_models(job.options)[0]["name"] if describe_models(job.options) else "unknown"),
                    "workspace": job.workspace,
                    "engine": "demo" if self.demo else "live",
                    "limits": {"max_steps": job.options.max_steps, "max_seconds": job.options.max_seconds},
                }
            )

        result = self.executor(
            job.goal,
            job.options,
            emit=job.emit,
            should_stop=job.control.should_stop,
            control=job.control,
            gate=job.gate,
        )
        job.result = result
        job.answer = str(result.get("answer") or "")
        job.ok = bool(result.get("ok"))
        job.stop_reason = str(result.get("stop_reason") or "")
        job.artifacts = list(result.get("artifacts") or [])
        job.plan = list(result.get("plan") or job.plan) if result.get("autonomous") else job.plan

        if result.get("stop_reason") == "cancelled":
            job.status = "cancelled"
        else:
            job.status = "done" if job.ok else "failed"
        job.finished = datetime.now().isoformat(timespec="seconds")
        job.finished_at = time.time()

        if job.ended:
            return
        job.emit(
            {
                "event": "task_end",
                "ok": job.ok,
                "stop_reason": job.stop_reason,
                "answer": job.answer,
                "steps": result.get("steps"),
                "artifacts": job.artifacts,
                "duration_ms": job.elapsed_ms(),
                "model": result.get("model") or job.model,
            }
        )


# =====================================================================
# 只读辅助（静态资源 / 文件树 / 快照）
# =====================================================================
def build_tree(root: Path, rel: str = ".", depth: int = 2, max_entries: int = 300) -> Dict[str, Any]:
    rel = (rel or ".").strip()
    base = (root / rel).resolve()
    if not base.is_relative_to(root) or not base.is_dir():
        return {"path": rel, "children": [], "error": "目录不存在或越界"}

    count = 0

    def walk(path: Path, level: int) -> List[Dict[str, Any]]:
        nonlocal count
        out: List[Dict[str, Any]] = []
        try:
            items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return out
        for item in items:
            if item.name in SKIP_DIRS or item.name.startswith(".agent"):
                continue
            count += 1
            if count > max_entries:
                out.append({"name": "…", "type": "more", "path": ""})
                break
            node: Dict[str, Any] = {
                "name": item.name,
                "path": item.relative_to(root).as_posix(),
                "type": "dir" if item.is_dir() else "file",
            }
            if item.is_file():
                try:
                    node["size"] = item.stat().st_size
                except OSError:
                    node["size"] = 0
            elif level < depth:
                node["children"] = walk(item, level + 1)
            out.append(node)
        return out

    return {"path": base.relative_to(root).as_posix(), "children": walk(base, 1)}


def list_snapshots(root: Path) -> List[Dict[str, Any]]:
    snap_dir = root / ".agent" / "snapshots"
    if not snap_dir.is_dir():
        return []
    rows = []
    for p in sorted(snap_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
        st = p.stat()
        rows.append(
            {
                "name": p.name,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


# =====================================================================
# HTTP
# =====================================================================
def make_handler(manager: TaskManager):
    root_dir = Path(manager.base.workspace).expanduser().resolve()
    web_dir = WEB_DIR
    from trust import TrustStore

    policy_store = TrustStore(root_dir / ".agent" / "trust-rules.json")

    class Handler(BaseHTTPRequestHandler):
        server_version = SERVER_VERSION
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        # ---------- 基础 ----------
        def _send(self, body: bytes, ctype: str, status: int = 200, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def _error(self, message: str, status: int = 400) -> None:
            self._json({"ok": False, "error": message}, status)

        def _query(self) -> Dict[str, str]:
            qs = parse_qs(urlparse(self.path).query)
            return {k: v[0] for k, v in qs.items()}

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        def _authorized(self) -> bool:
            if not manager.token:
                return True
            token = self.headers.get("X-Auth-Token") or self._query().get("token") or ""
            return secrets.compare_digest(str(token), manager.token)

        def _guard(self) -> bool:
            if self._authorized():
                return True
            self._error("未授权：需要访问令牌", 401)
            return False

        def _static(self, name: str) -> None:
            safe = unquote(name).lstrip("/")
            if ".." in safe or safe.startswith("."):
                return self._error("非法路径", 400)
            path = (web_dir / safe).resolve()
            if not path.is_relative_to(web_dir.resolve()) or not path.is_file():
                return self._error("文件不存在", 404)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(path.read_bytes(), ctype)

        # ---------- GET ----------
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path == "/favicon.ico":
                return self._send(b"", "image/x-icon", 204)
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/health":
                return self._json({"ok": True, "time": datetime.now().isoformat(timespec="seconds")})
            if path == "/api/config":
                return self._config()

            if not path.startswith("/api/") and path not in ("/run", "/status", "/events", "/tasks", "/memory", "/skills"):
                return self._error("未知路径", 404)
            if not self._guard():
                return

            q = self._query()
            if path in ("/api/stream", "/events"):
                return self._sse(q.get("id", ""))
            if path in ("/api/task", "/status"):
                job = manager.get(q.get("id", ""))
                if job is None:
                    return self._error("任务不存在", 404)
                return self._json(job.to_view())
            if path in ("/api/tasks", "/tasks"):
                return self._json({"tasks": manager.recent(int(q.get("limit") or 30))})
            if path == "/api/traces":
                rows = list_traces(root_dir / ".agent", limit=int(q.get("limit") or 30))
                return self._json({"traces": rows, "workspace": str(root_dir)})
            if path == "/api/trace":
                task_id = q.get("id", "")
                events = read_trace(root_dir / ".agent", task_id)
                if not events:
                    return self._error("轨迹不存在", 404)
                return self._json({"task_id": task_id, "events": events})
            if path == "/api/files":
                tree = build_tree(root_dir, q.get("path") or ".", int(q.get("depth") or 2))
                return self._json({"root": str(root_dir), "tree": tree})
            if path == "/api/file":
                return self._file(q.get("path") or "")
            if path == "/api/snapshots":
                return self._json({"snapshots": list_snapshots(root_dir)})
            if path in ("/api/skills", "/skills"):
                return self._skills()
            if path in ("/api/memory", "/memory"):
                try:
                    return self._json(MemoryStore(root_dir / ".agent" / "memory.db").stats())
                except Exception as e:
                    return self._error(f"{type(e).__name__}: {e}", 500)
            if path == "/api/trust":
                return self._trust()
            if path == "/api/trust/policies":
                return self._json({"policies": policy_store.list_rules()})
            if path == "/api/trust/policies/export":
                return self._json(policy_store.export_payload())
            if path == "/api/model/status":
                return self._model_status()
            if path == "/api/scope":
                return self._scope()
            return self._error("未知路径", 404)

        # ---------- POST ----------
        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if not self._guard():
                return
            q = self._query()
            data = self._body()

            if path in ("/api/run", "/run"):
                return self._run(data)
            if path in ("/api/stop", "/cancel"):
                return self._stop(data, q)
            if path == "/api/pause":
                return self._pause(data, q)
            if path == "/api/resume":
                return self._resume(data, q)
            if path == "/api/confirm":
                return self._confirm(data)
            if path == "/api/action":
                return self._action(data)
            if path == "/api/snapshot":
                return self._snapshot()
            if path == "/api/rollback":
                return self._rollback(data)
            if path == "/api/model/prefer":
                return self._model_prefer(data)
            if path == "/api/scope":
                return self._scope_set(data)
            if path == "/api/trust/policies":
                return self._policy_add(data)
            if path == "/api/trust/policies/update":
                return self._policy_update(data)
            if path == "/api/trust/policies/reorder":
                ids = data.get("ids") if isinstance(data.get("ids"), list) else []
                policy_store.reorder([str(i) for i in ids])
                return self._json({"ok": True, "count": len(policy_store.rules)})
            if path == "/api/trust/policies/dry-run":
                tool = str(data.get("tool") or "")
                if not tool:
                    return self._error("tool 不能为空", 400)
                args = data.get("args") if isinstance(data.get("args"), dict) else {}
                return self._json(policy_store.dry_run(tool, args))
            if path == "/api/trust/policies/import":
                payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
                count = policy_store.import_payload(payload)
                return self._json({"ok": count > 0, "count": count})
            return self._error("未知路径", 404)

        def do_DELETE(self) -> None:
            if not self._guard():
                return
            path = urlparse(self.path).path
            q = self._query()
            if path == "/api/skills":
                ok = delete_skill(root_dir / ".agent", q.get("name", ""))
                return self._json({"ok": ok})
            if path == "/api/trust/policies":
                ok = policy_store.delete(q.get("id", ""))
                return self._json({"ok": ok})
            if path == "/api/trust":
                try:
                    (root_dir / ".agent" / TRUST_FILE).unlink(missing_ok=True)
                    policy_store.reset()
                    return self._json({"ok": True})
                except Exception as e:
                    return self._error(f"{type(e).__name__}: {e}", 500)
            return self._error("未知路径", 404)

        def _policy_add(self, data: Dict[str, Any]) -> None:
            tool = str(data.get("tool") or "").strip()
            if not tool:
                return self._error("tool 不能为空（支持 glob，如 run_shell / read_*）", 400)
            action = str(data.get("action") or "ask")
            if action not in ("allow", "deny", "ask"):
                return self._error("action 只能是 allow / deny / ask", 400)
            if isinstance(data.get("param_pattern"), dict):
                data["param_pattern"] = {
                    str(k): str(v) for k, v in data["param_pattern"].items() if str(k).strip()
                }
            try:
                rule = policy_store.add(data)
            except Exception as e:
                return self._error(f"{type(e).__name__}: {e}", 400)
            return self._json({"ok": True, "rule": rule.to_dict()})

        def _policy_update(self, data: Dict[str, Any]) -> None:
            rule_id = str(data.get("id") or "")
            if not rule_id:
                return self._error("id 不能为空", 400)
            updates = {k: v for k, v in data.items() if k != "id"}
            rule = policy_store.update(rule_id, updates)
            if rule is None:
                return self._error("规则不存在", 404)
            return self._json({"ok": True, "rule": rule.to_dict()})

        # ---------- 具体实现 ----------
        def _config(self) -> None:
            opts = manager.base
            models = describe_models(opts)
            self._json(
                {
                    "version": SERVER_VERSION,
                    "workspace": str(root_dir),
                    "mode": opts.mode,
                    "models": models,
                    "limits": {
                        "max_steps": opts.max_steps,
                        "max_plan_steps": opts.max_plan_steps,
                        "max_replans": opts.max_replans,
                        "max_seconds": opts.max_seconds,
                    },
                    "confirm": {
                        "timeout": opts.confirm_timeout,
                        "confirm_writes": opts.confirm_writes,
                        "trust_threshold": 3,
                    },
                    "local": {
                        "enabled": manager.ollama is not None,
                        "model": (manager.ollama.cfg.model if manager.ollama else None),
                    },
                    "computer": {
                        "enabled": opts.full_computer,
                        "supported": os.name == "nt",
                        "tools": computer_tool_names(),
                    },
                    "features": {
                        "xhs": opts.use_xhs,
                        "memory": opts.use_memory,
                        "skills": opts.use_skills,
                        "critic": opts.use_critic,
                        "demo": manager.demo,
                        "scheduler": False,
                    },
                    "token_required": bool(manager.token),
                }
            )

        def _run(self, data: Dict[str, Any]) -> None:
            goal = str(data.get("goal") or "").strip()
            if not goal:
                return self._error("goal 不能为空", 400)
            if len(goal) > 4000:
                return self._error("goal 过长（上限 4000 字）", 400)

            overrides: Dict[str, Any] = {}
            raw = data.get("options") if isinstance(data.get("options"), dict) else {}
            merged = {**raw, **{k: v for k, v in data.items() if k != "options"}}
            for key in ("auto_yes", "readonly", "confirm_writes", "pause_on_failure", "snapshot", "headless", "full_computer"):
                if key in merged:
                    overrides[key] = bool(merged[key])
            for key in ("max_steps", "max_plan_steps", "max_replans", "max_seconds", "confirm_timeout"):
                if isinstance(merged.get(key), (int, float)):
                    overrides[key] = int(merged[key])
            if isinstance(merged.get("workspace"), str) and merged["workspace"].strip():
                overrides["workspace"] = merged["workspace"].strip()

            autonomous = merged.get("autonomous")
            try:
                job = manager.submit(goal, autonomous=None if autonomous is None else bool(autonomous), overrides=overrides)
            except Exception as e:
                return self._error(f"{type(e).__name__}: {e}", 400)
            self._json({"task_id": job.id, "status": job.status}, 202)

        def _job(self, data: Dict[str, Any], q: Dict[str, str]) -> Optional[Job]:
            job_id = str(data.get("task_id") or q.get("id") or "")
            job = manager.get(job_id)
            if job is None and job_id:
                self._error("任务不存在", 404)
            return job

        def _stop(self, data: Dict[str, Any], q: Dict[str, str]) -> None:
            job = self._job(data, q)
            if job is None:
                return
            job.control.stop()
            job.emit({"event": "log", "level": "warn", "message": "已请求停止：当前步骤结束后不再派新步骤"})
            self._json({"ok": True, "note": "优雅停止：当前工具调用结束后收尾"})

        def _pause(self, data: Dict[str, Any], q: Dict[str, str]) -> None:
            job = self._job(data, q)
            if job is None:
                return
            job.control.pause()
            self._json({"ok": True})

        def _resume(self, data: Dict[str, Any], q: Dict[str, str]) -> None:
            job = self._job(data, q)
            if job is None:
                return
            job.control.resume()
            self._json({"ok": True})

        def _confirm(self, data: Dict[str, Any]) -> None:
            job = manager.get(str(data.get("task_id") or ""))
            if job is None:
                return self._error("任务不存在", 404)
            decision = str(data.get("decision") or "")
            if decision not in ("approve", "approve_session", "deny"):
                return self._error("decision 只能是 approve / approve_session / deny", 400)
            args = data.get("args") if isinstance(data.get("args"), dict) else None
            ok = job.gate.resolve(
                cid=str(data.get("cid") or ""),
                decision=decision,
                remember=bool(data.get("remember")),
                args=args,
                reason=str(data.get("reason") or ""),
            )
            self._json({"ok": ok})

        def _action(self, data: Dict[str, Any]) -> None:
            job = manager.get(str(data.get("task_id") or ""))
            if job is None:
                return self._error("任务不存在", 404)
            action = str(data.get("action") or "")
            if action not in ("retry", "skip", "fix", "continue", "abort"):
                return self._error("action 不合法", 400)
            args = data.get("args") if isinstance(data.get("args"), dict) else None
            ok = job.control.resolve_action(str(data.get("cid") or ""), action, args)
            self._json({"ok": ok})

        def _file(self, rel: str) -> None:
            try:
                target = (root_dir / rel).resolve()
            except Exception:
                return self._error("路径非法", 400)
            if not target.is_relative_to(root_dir) or not target.is_file():
                return self._error("文件不存在或越界", 404)
            if target.stat().st_size > 512 * 1024:
                return self._error("文件太大（>512KB），请到编辑器里看", 413)
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return self._error(f"{type(e).__name__}: {e}", 500)
            self._json({"path": target.relative_to(root_dir).as_posix(), "content": text, "size": target.stat().st_size})

        def _skills(self) -> None:
            try:
                skills = load_skills(root_dir / ".agent")
            except Exception as e:
                return self._error(f"{type(e).__name__}: {e}", 500)
            self._json(
                {
                    "count": len(skills),
                    "skills": [
                        {
                            "name": s.name,
                            "description": s.description,
                            "steps": [st.get("tool") for st in s.steps],
                            "params": list(s.params),
                            "uses": s.uses,
                            "successes": s.successes,
                        }
                        for s in skills
                    ],
                }
            )

        def _trust(self) -> None:
            path = root_dir / ".agent" / TRUST_FILE
            rules: Dict[str, Any] = {}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rules = data.get("rules") or {}
            except Exception:
                pass
            self._json({"rules": rules, "threshold": 3})

        def _model_status(self) -> None:
            """模型路由状态：本地通道健康 + 全局偏好 + 累计用量（状态栏数据源）。"""
            snap = gateway_mod.shared_snapshot()
            local = {"available": False, "model": None, "base_url": None}
            if manager.ollama is not None:
                local = {
                    "available": manager.ollama.probe(),
                    "model": manager.ollama.cfg.model,
                    "base_url": manager.ollama.cfg.base_url,
                }
            self._json(
                {
                    "enabled": manager.ollama is not None,
                    "local": local,
                    "fallback": {
                        "api": any(
                            m.get("kind") == "api" and m.get("available")
                            for m in describe_models(manager.base)
                        ),
                        "web": any(m.get("kind") == "web" for m in describe_models(manager.base)),
                    },
                    **snap,
                }
            )

        def _model_prefer(self, data: Dict[str, Any]) -> None:
            prefer = bool(data.get("prefer_local"))
            gateway_mod.set_prefer_local(prefer)
            self._json({"ok": True, "prefer_local": prefer})

        def _scope(self) -> None:
            """电脑操控作用域：当前开关、平台支持、可启用的工具清单。"""
            running = sum(1 for j in manager.jobs.values() if j.status == "running")
            self._json(
                {
                    "enabled": bool(manager.base.full_computer),
                    "supported": os.name == "nt",
                    "tools": computer_tool_names(),
                    "running_jobs": running,
                }
            )

        def _scope_set(self, data: Dict[str, Any]) -> None:
            """切换电脑作用域。只影响之后提交的任务（跑着的不动）。"""
            enabled = bool(data.get("enabled"))
            manager.base.full_computer = enabled
            self._json(
                {
                    "ok": True,
                    "enabled": enabled,
                    "note": "已切换。只对之后提交的新任务生效。",
                }
            )

        def _snapshot(self) -> None:
            from sandbox import Sandbox

            try:
                snap = Sandbox(root_dir).snapshot()
            except Exception as e:
                return self._error(f"{type(e).__name__}: {e}", 500)
            self._json({"ok": bool(snap), "snapshot": snap.name if snap else None})

        def _rollback(self, data: Dict[str, Any]) -> None:
            from sandbox import Sandbox

            name = str(data.get("snapshot") or "")
            if not name or "/" in name or "\\" in name:
                return self._error("snapshot 名称非法", 400)
            result = Sandbox(root_dir).rollback(root_dir / ".agent" / "snapshots" / name)
            self._json(result, 200 if result.get("ok") else 400)

        # ---------- SSE ----------
        def _sse(self, job_id: str) -> None:
            job = manager.get(job_id)
            if job is None:
                return self._error("任务不存在", 404)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            q, history = job.subscribe()
            last_seq = 0
            try:
                # 先补历史 → 刷新/断线重连不会丢卡片，也不会重复（前端按 card id 幂等）
                for ev in history:
                    self._write_sse(ev)
                    last_seq = int(ev.get("seq") or 0)
                self._write_sse(
                    {
                        "event": "snapshot",
                        "task_id": job.id,
                        "seq": last_seq,
                        "status": job.status,
                        "usage": job.usage,
                        "view": job.to_view(),
                    }
                )

                while True:
                    try:
                        ev = q.get(timeout=15)
                    except queue.Empty:
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except OSError:
                            break
                        if job.status in ("done", "failed", "cancelled") and not job._subs:
                            break
                        continue
                    if ev.get("event") == "__close__":
                        break
                    seq = int(ev.get("seq") or 0)
                    if seq and seq <= last_seq:
                        continue
                    last_seq = seq or last_seq
                    self._write_sse(ev)
                    if ev.get("event") == "task_end":
                        break
            except OSError:
                pass  # 浏览器关页面/断网都可能在这里抛，属于正常结束
            finally:
                job.unsubscribe(q)

        def _write_sse(self, ev: Dict[str, Any]) -> None:
            payload = json.dumps(ev, ensure_ascii=False, default=str)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


def serve(
    options: Optional[RunOptions] = None,
    port: int = 8787,
    host: str = "127.0.0.1",
    with_scheduler: bool = False,
    block: bool = True,
    token: str = "",
    demo: bool = False,
):
    options = options or RunOptions()
    root = Path(options.workspace).expanduser().resolve()

    if demo:
        from demo_model import prepare_demo_workspace

        prepare_demo_workspace(root)

    loopback = host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not token:
        token = secrets.token_urlsafe(12)
        print(f"[serve] 非本机监听，已自动生成访问令牌：{token}")
    if token:
        print(f"[serve] 访问令牌已启用（?token=... 或 X-Auth-Token）")

    if demo:
        from demo_model import demo_execute

        manager = TaskManager(options, token=token, demo=True, executor=demo_execute)
    else:
        manager = TaskManager(options, token=token, demo=False)

    httpd = ThreadingHTTPServer((host, port), make_handler(manager))
    httpd.daemon_threads = True

    sched = None
    if with_scheduler:
        from scheduler import Scheduler, schedule_loop

        sched = Scheduler(options)
        threading.Thread(target=schedule_loop, args=(sched,), name="agent-scheduler", daemon=True).start()

    print(f"[serve] 控制台：http://{host}:{port}/")
    print(f"[serve] 工作区：{root}")
    print(f"[serve] 模型：{', '.join(m['name'] for m in describe_models(options)) or '未配置'}")
    if demo:
        print("[serve] 演示模式：使用脚本化假模型，不联网、不启浏览器")
    print("[serve] 只监听本机；对外提供请用 --host 0.0.0.0 并设置 --token")

    if not block:
        # 后台模式（自检 / 嵌入其它进程）：必须真的开一个 accept 线程，否则连接会挂住
        threading.Thread(target=httpd.serve_forever, name="agent-http", daemon=True).start()
        return httpd, manager, sched
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 正在退出……")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return httpd, manager, sched
