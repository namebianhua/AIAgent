"""长期记忆（T3）：把每次任务沉淀成可检索的「经验卡」。

设计取舍：
1. 不用向量库（本机装 Chroma/FAISS 会引入编译依赖，之前 CDN 已经踩过坑），
   改用「中文单字 + 双字 + 英文词」的 TF-IDF 余弦相似度，几十~几千条任务足够用；
2. 只存结构与摘要，**绝不存文件正文/代码内容**（写入前统一脱敏），
   否则记忆库会变成敏感数据仓库；
3. 每次操作都新开连接，天然支持多线程（server / scheduler 会并发读）。
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SENSITIVE_KEYS = {"content", "code", "text", "body", "cmd", "script", "query", "password", "token"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT,
    ts          TEXT,
    goal        TEXT,
    tokens      TEXT,
    plan        TEXT,
    answer      TEXT,
    stop_reason TEXT,
    ok          INTEGER,
    steps       INTEGER,
    model       TEXT,
    tools       TEXT,
    artifacts   TEXT,
    trace       TEXT,
    duration_ms INTEGER,
    workspace   TEXT,
    autonomous  INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tasks_ts ON tasks(ts);
"""


def tokenize(text: str) -> List[str]:
    """中英混合分词：英文/数字按词，中文按单字 + 双字（双字权重更高）。"""
    text = (text or "").lower()
    words = _WORD_RE.findall(text)
    cjk = "".join(_CJK_RE.findall(text))
    chars = list(cjk)
    bigrams = [cjk[i: i + 2] for i in range(len(cjk) - 1)]
    return words + chars + bigrams * 2


def _vectorize(tokens: Iterable[str]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # 遍历较短的一侧
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    if dot <= 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _sanitize_args(args: Any, limit: int = 120) -> Dict[str, Any]:
    """参数脱敏：敏感键只保留长度，长文本截断。"""
    if not isinstance(args, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in args.items():
        if k in _SENSITIVE_KEYS:
            if isinstance(v, str):
                out[k] = f"<{len(v)} 字符已省略>"
            else:
                out[k] = "<已省略>"
        elif isinstance(v, str):
            out[k] = v if len(v) <= limit else v[:limit] + f"...<共 {len(v)} 字符>"
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)[:limit]
    return out


@dataclass
class TaskRecord:
    goal: str
    ts: str
    ok: bool
    score: float
    plan: List[str]
    tools: List[str]
    artifacts: List[str]
    answer: str
    stop_reason: str
    failures: List[str]

    def to_card(self, index: int) -> str:
        mark = "成功" if self.ok else f"未完成({self.stop_reason})"
        lines = [f"[{index}] 相似度 {self.score:.2f} | {self.ts[:10]} | {mark}", f"目标：{self.goal}"]
        if self.plan:
            lines.append("做法：" + " → ".join(s[:60] for s in self.plan[:6]))
        if self.tools:
            lines.append("用到的工具：" + ", ".join(self.tools[:10]))
        if self.artifacts:
            lines.append("产物：" + ", ".join(self.artifacts[:8]))
        if self.failures:
            lines.append("踩过的坑：" + "；".join(self.failures[:3]))
        return "\n".join(lines)


class MemoryStore:
    """任务经验的持久化与检索（每个工作区一个 .agent/memory.db）。"""

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        if db_path is None:
            db_path = Path(".agent") / "memory.db"
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ---------- 基础 ----------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # 老库平滑升级：缺列就补
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
            for name, ddl in (("trace", "TEXT"), ("autonomous", "INTEGER DEFAULT 0")):
                if name not in cols:
                    conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {ddl}")

    # ---------- 写入 ----------
    def remember(
        self,
        goal: str,
        *,
        answer: str = "",
        plan: Optional[List[str]] = None,
        trace: Optional[List[Dict[str, Any]]] = None,
        stop_reason: str = "finished",
        ok: bool = True,
        steps: int = 0,
        model: str = "",
        artifacts: Optional[List[str]] = None,
        duration_ms: int = 0,
        workspace: str = "",
        task_id: str = "",
        autonomous: bool = False,
    ) -> int:
        trace = trace or []
        tools: List[str] = []
        for item in trace:
            name = item.get("tool")
            if name and name not in tools and not str(name).startswith("("):
                tools.append(str(name))

        record = {
            "task_id": task_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "goal": (goal or "")[:1000],
            "tokens": json.dumps(tokenize(goal), ensure_ascii=False),
            "plan": json.dumps(plan or [], ensure_ascii=False),
            "answer": (answer or "")[:2000],
            "stop_reason": stop_reason,
            "ok": 1 if ok else 0,
            "steps": steps or len(trace),
            "model": model,
            "tools": json.dumps(tools, ensure_ascii=False),
            "artifacts": json.dumps(artifacts or [], ensure_ascii=False),
            "trace": json.dumps(
                [
                    {
                        "step": item.get("step"),
                        "tool": item.get("tool"),
                        "args": _sanitize_args(item.get("args")),
                        "ok": bool(item.get("ok")),
                        "error": (str(item.get("error"))[:160] if item.get("error") else None),
                    }
                    for item in trace[:40]
                ],
                ensure_ascii=False,
            ),
            "duration_ms": duration_ms,
            "workspace": workspace,
            "autonomous": 1 if autonomous else 0,
        }
        cols = ", ".join(record)
        marks = ", ".join("?" for _ in record)
        with self._conn() as conn:
            cur = conn.execute(
                f"INSERT INTO tasks ({cols}) VALUES ({marks})", list(record.values())
            )
            # 轨迹另存一份到 answer 旁边？不——保持表窄，轨迹由 tracing.py 落盘
            return int(cur.lastrowid)

    def remember_trace(self, goal: str, result: Dict[str, Any], **kwargs) -> int:
        """从 run_task / autonomous 的结果字典直接落库。"""
        return self.remember(
            goal,
            answer=result.get("answer", ""),
            plan=result.get("plan") or [s.get("step", "") for s in result.get("step_results", [])],
            trace=result.get("trace") or [],
            stop_reason=result.get("stop_reason", "finished"),
            ok=bool(result.get("ok", result.get("stop_reason") == "finished")),
            steps=result.get("steps", 0),
            model=result.get("model", ""),
            artifacts=result.get("artifacts") or [],
            duration_ms=result.get("duration_ms", 0),
            workspace=result.get("workspace", ""),
            task_id=result.get("task_id", ""),
            autonomous=bool(result.get("autonomous")),
            **kwargs,
        )

    # ---------- 检索 ----------
    def recall(self, goal: str, k: int = 3, min_score: float = 0.05) -> List[TaskRecord]:
        if not goal:
            return []
        q = _vectorize(tokenize(goal))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY id DESC LIMIT 500"
            ).fetchall()

        # 用候选集计算 IDF，压制「任务/文件/帮我」这类高频词的权重
        docs: List[Dict[str, float]] = []
        dfs: Dict[str, int] = {}
        for r in rows:
            try:
                toks = json.loads(r["tokens"] or "[]")
            except Exception:
                toks = tokenize(r["goal"] or "")
            vec = _vectorize(toks)
            docs.append(vec)
            for term in vec:
                dfs[term] = dfs.get(term, 0) + 1

        n = max(1, len(rows))
        idf = {t: math.log((1 + n) / (1 + df)) + 1.0 for t, df in dfs.items()}

        def weigh(vec: Dict[str, float]) -> Dict[str, float]:
            return {t: v * idf.get(t, 1.0) for t, v in vec.items()}

        qv = weigh(q)
        scored: List[TaskRecord] = []
        for r, vec in zip(rows, docs):
            score = _cosine(qv, weigh(vec))
            if score < min_score:
                continue
            try:
                plan = json.loads(r["plan"] or "[]")
                tools = json.loads(r["tools"] or "[]")
                artifacts = json.loads(r["artifacts"] or "[]")
            except Exception:
                plan, tools, artifacts = [], [], []

            failures = self._failures_from_row(r)
            scored.append(
                TaskRecord(
                    goal=r["goal"] or "",
                    ts=r["ts"] or "",
                    ok=bool(r["ok"]),
                    score=score,
                    plan=plan if isinstance(plan, list) else [],
                    tools=tools if isinstance(tools, list) else [],
                    artifacts=artifacts if isinstance(artifacts, list) else [],
                    answer=r["answer"] or "",
                    stop_reason=r["stop_reason"] or "",
                    failures=failures,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(1, k)]

    def _failures_from_row(self, row: sqlite3.Row) -> List[str]:
        """失败点优先取自库内的紧凑轨迹，缺失时回落到独立轨迹文件。"""
        out: List[str] = []
        try:
            items = json.loads(row["trace"] or "[]")
        except Exception:
            items = []
        for item in items:
            if isinstance(item, dict) and not item.get("ok"):
                out.append(
                    f"{item.get('tool')} 失败：{str(item.get('error') or '')[:80]}"
                )
        if out:
            return out[:5]

        task_id = row["task_id"] if "task_id" in row.keys() else ""
        return self._failures_of(task_id) if task_id else []

    def _failures_of(self, task_id: str) -> List[str]:
        """从同名轨迹文件里提取失败点（轨迹是独立 JSONL，不占数据库）。"""
        path = self.db_path.parent / "traces" / f"{task_id}.jsonl"
        if not path.is_file():
            return []
        out: List[str] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("event") == "tool" and not ev.get("ok"):
                        out.append(
                            f"{ev.get('tool')} 失败：{str(ev.get('error') or '')[:80]}"
                        )
        except Exception:
            return []
        return out[:5]

    def format_experience(self, goal: str, k: int = 3) -> str:
        records = self.recall(goal, k=k)
        if not records:
            return ""
        return "\n".join(r.to_card(i) for i, r in enumerate(records, 1))

    # ---------- 观测 ----------
    def stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
            ok = conn.execute("SELECT COUNT(*) c FROM tasks WHERE ok=1").fetchone()["c"]
            since = conn.execute(
                "SELECT MIN(ts) a, MAX(ts) b FROM tasks"
            ).fetchone()
            top = conn.execute(
                "SELECT goal, ts, ok FROM tasks ORDER BY id DESC LIMIT 5"
            ).fetchall()
        return {
            "db": str(self.db_path),
            "total": total,
            "ok": ok,
            "failed": total - ok,
            "first": since["a"],
            "last": since["b"],
            "recent": [
                {"goal": (r["goal"] or "")[:80], "ts": r["ts"], "ok": bool(r["ok"])}
                for r in top
            ],
        }

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT task_id, ts, goal, ok, stop_reason, steps, artifacts "
                "FROM tasks ORDER BY id DESC LIMIT ?",
                (max(1, n),),
            ).fetchall()
        out = []
        for r in rows:
            try:
                arts = json.loads(r["artifacts"] or "[]")
            except Exception:
                arts = []
            out.append(
                {
                    "task_id": r["task_id"],
                    "ts": r["ts"],
                    "goal": r["goal"],
                    "ok": bool(r["ok"]),
                    "stop_reason": r["stop_reason"],
                    "steps": r["steps"],
                    "artifacts": arts,
                }
            )
        return out

    def forget_all(self) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM tasks")
            return cur.rowcount
