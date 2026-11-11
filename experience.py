"""经验库（第四阶段）：把任务轨迹向量化沉淀，让 Agent 越用越聪明。

和 task_memory.MemoryStore 的分工：
- MemoryStore  = 结构化台账（sqlite + TF-IDF 关键词召回），离线可用、零依赖，
  擅长「字面相似」的检索，也是统计/列表/回放的数据源；
- ExperienceStore = 语义经验库（Ollama 向量 + ChromaDB），擅长「意思相近但用词不同」
  的召回——例如以前做过「汇总各部门报表」，现在问「把这几个 Excel 合起来算个总数」
  也能命中。两者互补，不互相替代。

隐私约定（沿用 task_memory 的立场）：**只存结构与方法，不存文件正文/代码内容**。
经验卡里出现的是「用了哪个工具、参数是什么类型、踩了什么坑」，不是用户的文档内容。
写入前统一脱敏，避免经验库变成第二个敏感数据仓库。

降级约定：chromadb 没装 / Ollama 没起时，ingest 返回 skipped、recall 返回空列表，
绝不抛异常打断任务——经验库是增益能力，不是主链路依赖。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# 参数里这些键的值不进经验卡（可能是密码/正文/命令原文）
_SENSITIVE_KEYS = {
    "content", "code", "text", "body", "cmd", "script", "query", "password",
    "token", "secret", "cookie", "authorization",
}
# 这些工具的输出只是噪音，不进经验卡
_NOISE_TOOLS = {"sys_screenshot", "sys_screen_size"}

DEFAULT_EXPERIENCE_COLLECTION = "agent_experience"


def _short(value: Any, limit: int = 160) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "…"


def sanitize_args(args: Any, limit: int = 100) -> Dict[str, Any]:
    """参数脱敏：敏感键只留长度，长文本截断。"""
    if not isinstance(args, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in args.items():
        lk = str(k).lower()
        if lk in _SENSITIVE_KEYS or any(t in lk for t in ("password", "token", "secret", "key")):
            out[k] = f"<{len(v)} 字符已省略>" if isinstance(v, str) else "<已省略>"
        elif isinstance(v, str):
            out[k] = v if len(v) <= limit else v[:limit] + f"…<共 {len(v)} 字符>"
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = _short(v, limit)
    return out


@dataclass
class ExperienceHit:
    """一条召回到的历史经验。"""
    task_id: str
    goal: str
    score: float
    ok: bool
    tools: List[str]
    plan: List[str]
    pitfalls: List[str]
    answer: str
    ts: str
    content: str
    metadata: Dict[str, Any]

    def to_card(self, index: int = 1) -> str:
        mark = "成功" if self.ok else "未完成"
        lines = [f"[{index}] 相关度 {self.score:.2f} | {self.ts[:10]} | {mark}"]
        if self.goal:
            lines.append(f"历史任务：{self.goal[:200]}")
        if self.plan:
            lines.append("当时的做法：" + " → ".join(s[:70] for s in self.plan[:6]))
        if self.tools:
            lines.append("用到的工具：" + ", ".join(self.tools[:10]))
        if self.pitfalls:
            lines.append("踩过的坑：" + "；".join(self.pitfalls[:3]))
        if self.answer:
            lines.append("结论摘要：" + self.answer[:220])
        return "\n".join(lines)


class ExperienceStore:
    """向量化经验库：写入任务经验 + 语义召回。"""

    def __init__(
        self,
        engine: Any = None,
        collection: str = DEFAULT_EXPERIENCE_COLLECTION,
        *,
        enabled: bool = True,
        include_failures: bool = True,
        min_steps: int = 1,
        min_answer_chars: int = 2,
    ) -> None:
        self.engine = engine
        self.collection = collection
        self.enabled = enabled
        self.include_failures = include_failures
        self.min_steps = max(0, int(min_steps))
        self.min_answer_chars = max(0, int(min_answer_chars))
        self.last_error: str = ""
        self._indexed: Dict[str, str] = {}  # task_id -> doc_id，避免重复入库

    # ---------- 可用性 ----------
    @property
    def available(self) -> bool:
        return bool(self.enabled and self.engine is not None)

    def probe(self) -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "reason": self.last_error or "经验库未启用"}
        try:
            info = self.engine.probe()
        except Exception as e:
            info = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        info["collection"] = self.collection
        return info

    # ---------- 经验卡构造 ----------
    @staticmethod
    def _pitfalls(trace: Sequence[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in trace or []:
            if not isinstance(item, dict) or item.get("ok"):
                continue
            tool = str(item.get("tool") or "?")
            if tool in _NOISE_TOOLS or tool.startswith("("):
                continue
            err = _short(item.get("error") or "", 120)
            line = f"{tool} 失败：{err}" if err else f"{tool} 未成功"
            if line not in out:
                out.append(line)
        return out[:6]

    @staticmethod
    def _tool_chain(trace: Sequence[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in trace or []:
            if not isinstance(item, dict):
                continue
            name = item.get("tool")
            if not name or str(name).startswith("(") or str(name) in _NOISE_TOOLS:
                continue
            name = str(name)
            args = sanitize_args(item.get("args"))
            flag = "✓" if item.get("ok") else "✗"
            brief = ", ".join(f"{k}={_short(v, 60)}" for k, v in list(args.items())[:3])
            out.append(f"{flag} {name}({brief})" if brief else f"{flag} {name}")
        return out

    def build_card(
        self,
        goal: str,
        result: Dict[str, Any],
        *,
        task_id: str = "",
    ) -> str:
        """把一次任务渲染成可检索的经验卡（纯文本，向量化后进库）。"""
        trace = result.get("trace") or []
        plan = result.get("plan") or [
            str(s.get("step", "")) for s in (result.get("step_results") or [])
        ]
        tools: List[str] = []
        for item in trace:
            name = str(item.get("tool") or "") if isinstance(item, dict) else ""
            if name and name not in tools and not name.startswith("(") and name not in _NOISE_TOOLS:
                tools.append(name)
        ok = bool(result.get("ok", result.get("stop_reason") == "finished"))
        answer = _short(result.get("answer") or "", 500)
        pitfalls = self._pitfalls(trace)
        chain = self._tool_chain(trace)
        artifacts = [str(a) for a in (result.get("artifacts") or [])][:8]

        lines: List[str] = [
            f"任务目标：{(goal or '').strip()[:400]}",
            f"执行结果：{'成功' if ok else '未完成'}（终止原因 {result.get('stop_reason') or '-'}）",
            f"步数：{result.get('steps') or len(trace)}｜模型：{_short(result.get('model') or '', 60)}",
        ]
        if plan:
            lines.append("计划步骤：")
            lines += [f"  {i}. {_short(s, 120)}" for i, s in enumerate(plan[:8], 1)]
        if tools:
            lines.append("使用的工具：" + ", ".join(tools[:12]))
        if chain:
            lines.append("工具调用链（参数已脱敏）：")
            lines += [f"  {c[:180]}" for c in chain[:14]]
        if artifacts:
            lines.append("产出文件：" + ", ".join(artifacts))
        if pitfalls:
            lines.append("失败与教训：")
            lines += [f"  - {p}" for p in pitfalls]
        if answer:
            lines.append(f"最终结论：{answer}")
        if result.get("criteria"):
            lines.append(f"验收标准：{_short(result['criteria'], 200)}")
        return "\n".join(lines)

    # ---------- 写入 ----------
    def ingest(
        self,
        goal: str,
        result: Dict[str, Any],
        *,
        task_id: str = "",
        workspace: str = "",
    ) -> Dict[str, Any]:
        """把一次任务沉淀成经验（成功必存；失败按 include_failures 决定）。"""
        if not self.available:
            return {"status": "skipped", "reason": self.last_error or "经验库未启用"}

        tid = task_id or str(result.get("task_id") or "")
        if not tid:
            tid = "task-" + hashlib.sha256(
                ((goal or "") + json.dumps(result.get("trace") or [], ensure_ascii=False, default=str)).encode("utf-8")
            ).hexdigest()[:12]

        ok = bool(result.get("ok", result.get("stop_reason") == "finished"))
        if not ok and not self.include_failures:
            return {"status": "skipped", "reason": "任务未成功，且配置为只沉淀成功经验",
                    "task_id": tid}

        trace = result.get("trace") or []
        if len(trace) < self.min_steps:
            return {"status": "skipped", "reason": f"工具调用不足 {self.min_steps} 步，无沉淀价值",
                    "task_id": tid}

        card = self.build_card(goal, result, task_id=tid)
        if len(card) < self.min_answer_chars:
            return {"status": "skipped", "reason": "经验卡内容为空", "task_id": tid}

        meta = {
            "kind": "task_experience",
            "task_id": tid,
            "goal": (goal or "")[:500],
            "ok": ok,
            "stop_reason": str(result.get("stop_reason") or "")[:60],
            "steps": int(result.get("steps") or len(trace)),
            "tools": ", ".join(sorted({
                str(i.get("tool")) for i in trace
                if isinstance(i, dict) and i.get("tool") and not str(i.get("tool")).startswith("(")
            }))[:400],
            "artifacts": ", ".join(str(a) for a in (result.get("artifacts") or [])[:8])[:400],
            "pitfalls": " | ".join(self._pitfalls(trace))[:500],
            "answer": _short(result.get("answer") or "", 400),
            "model": _short(result.get("model") or "", 60),
            "workspace": str(workspace or result.get("workspace") or "")[:200],
            "autonomous": bool(result.get("autonomous")),
            "duration_ms": int(result.get("duration_ms") or 0),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        try:
            res = self.engine.add_text(
                card,
                source_id=f"task:{tid}",
                collection_name=self.collection,
                metadata=meta,
                source_name=(goal or tid)[:120],
            )
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return {"status": "failed", "reason": self.last_error, "task_id": tid}

        if res.get("status") == "success" and res.get("doc_id"):
            self._indexed[tid] = str(res["doc_id"])
        res["task_id"] = tid
        return res

    def ingest_from_trace_file(self, path: str | os.PathLike[str]) -> Dict[str, Any]:
        """从 TraceLogger 落盘的 JSONL 轨迹回灌经验（历史任务补录用）。"""
        p = Path(path).expanduser()
        if not p.is_file():
            return {"status": "failed", "reason": f"轨迹文件不存在：{p}"}
        events: List[Dict[str, Any]] = []
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

        goal, answer, ok, stop_reason, task_id = "", "", True, "finished", p.stem
        trace: List[Dict[str, Any]] = []
        plan: List[str] = []
        for ev in events:
            kind = ev.get("event")
            if kind == "task_start":
                goal = str(ev.get("goal") or "")
                task_id = str(ev.get("task_id") or task_id)
            elif kind == "plan" and not plan:
                plan = [str(s) for s in (ev.get("steps") or [])]
            elif kind in ("tool", "tool_end"):
                trace.append({
                    "step": ev.get("step") or ev.get("turn"),
                    "tool": ev.get("tool"),
                    "args": ev.get("args"),
                    "ok": bool(ev.get("ok")),
                    "error": ev.get("error"),
                    "elapsed_ms": ev.get("elapsed_ms") or 0,
                })
            elif kind == "task_end":
                ok = bool(ev.get("ok"))
                stop_reason = str(ev.get("stop_reason") or stop_reason)
                answer = str(ev.get("answer") or "")
        result = {"answer": answer, "ok": ok, "stop_reason": stop_reason,
                  "trace": trace, "steps": len(trace), "plan": plan,
                  "task_id": task_id, "artifacts": []}
        return self.ingest(goal, result, task_id=task_id)

    # ---------- 召回 ----------
    def recall(
        self,
        goal: str,
        k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[ExperienceHit]:
        """按语义相似度召回历史经验；不可用或无命中时返回空列表。"""
        if not self.available or not (goal or "").strip():
            return []
        try:
            rows = self.engine.search(
                goal, top_k=max(1, int(k)), collection_name=self.collection,
                min_score=min_score,
            )
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []

        hits: List[ExperienceHit] = []
        for r in rows:
            meta = r.get("metadata") or {}
            tools = [t.strip() for t in str(meta.get("tools") or "").split(",") if t.strip()]
            pitfalls = [p.strip() for p in str(meta.get("pitfalls") or "").split("|") if p.strip()]
            plan = [s.strip() for s in str(meta.get("plan") or "").split("\n") if s.strip()]
            hits.append(ExperienceHit(
                task_id=str(meta.get("task_id") or ""),
                goal=str(meta.get("goal") or ""),
                score=float(r.get("relevance_score") or 0.0),
                ok=bool(meta.get("ok")),
                tools=tools,
                plan=plan,
                pitfalls=pitfalls,
                answer=str(meta.get("answer") or ""),
                ts=str(meta.get("ts") or ""),
                content=str(r.get("content") or ""),
                metadata=dict(meta),
            ))
        return hits

    def format_experience(self, goal: str, k: int = 3, min_score: Optional[float] = None) -> str:
        hits = self.recall(goal, k=k, min_score=min_score)
        if not hits:
            return ""
        return "\n\n".join(h.to_card(i) for i, h in enumerate(hits, 1))

    # ---------- 观测 ----------
    def stats(self) -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "collection": self.collection,
                    "reason": self.last_error or "经验库未启用"}
        try:
            sources = self.engine.list_sources(self.collection)
            st = self.engine.stats()
        except Exception as e:
            return {"ok": False, "collection": self.collection,
                    "reason": f"{type(e).__name__}: {e}"}
        coll = next((c for c in st.get("collections", []) if c.get("name") == self.collection), {})
        return {
            "ok": True,
            "collection": self.collection,
            "experiences": len(sources),
            "chunks": coll.get("chunks", 0),
            "embed_model": coll.get("embed_model", ""),
            "indexed_this_session": len(self._indexed),
        }

    def forget(self, task_id: str) -> int:
        """删除某个任务的经验。"""
        if not self.available:
            return 0
        try:
            return int(self.engine.delete_by_source(f"task:{task_id}", self.collection))
        except Exception:
            return 0

    def forget_all(self) -> bool:
        if not self.available:
            return False
        try:
            self.engine.delete_collection(self.collection)
            self._indexed.clear()
            return True
        except Exception:
            return False


def build_experience_store(
    engine: Any = None,
    collection: str = DEFAULT_EXPERIENCE_COLLECTION,
    **kwargs: Any,
) -> ExperienceStore:
    """构造经验库；engine 为 None（chromadb 缺失/Ollama 不可用）时返回禁用态实例。"""
    return ExperienceStore(engine=engine, collection=collection, **kwargs)
