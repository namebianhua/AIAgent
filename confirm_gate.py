"""人工确认闸门（交互层）：三种粒度 + 信任规则 + 超时自动拒绝 + 变更预览。

为什么不用 engine.confirmer：
终端里的 y/n 只能表达「过 / 不过」，而界面上必须能表达三件事——
允许一次、本任务内同类全放行、拒绝并给出理由（回灌给模型换方案）。
所以这里是一个可阻塞的 broker，由 registry.approver 调用，UI 通过 HTTP 回填决策。

信任规则（从「烦人」到「好用」的分水岭）：
同一个 (工具, 参数名集合) 被明确允许 3 次后，自动升为白名单，落盘到
.agent/trust.json，之后同类操作不再打扰。可在设置页清空。
"""
from __future__ import annotations

import difflib
import json
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:  # 仅类型提示，避免与 engine 循环依赖
    from engine import ToolSpec
    from sandbox import Sandbox

TRUST_THRESHOLD = 3
TRUST_FILE = "trust.json"

_HIGH_RISK = {"run_shell", "run_python", "delete_file", "xhs_publish", "ui_kill_process", "ui_open_app"}
_WRITE_TOOLS = {"write_file", "move_file", "make_dir"}


def _short(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n…（已截断，共 {len(text)} 字符）"


class ConfirmBroker:
    """registry.approver 的实现；一个任务一个实例。"""

    def __init__(
        self,
        emit: Callable[[Dict[str, Any]], None],
        root: str | os.PathLike[str],
        timeout_s: int = 30,
        auto_yes: bool = False,
        confirm_writes: bool = False,
        trust_path: Optional[Path] = None,
        control: Any = None,
        scope: str = "workspace",
    ) -> None:
        from sandbox import Sandbox

        self._emit = emit
        self.root = Path(root).expanduser().resolve()
        self.sandbox = Sandbox(self.root, scope=scope)
        self.timeout_s = max(5, int(timeout_s))
        self.auto_yes = bool(auto_yes)
        self.confirm_writes = bool(confirm_writes)
        self.control = control  # 用于「任务已停止」时立刻拒绝
        self.trust_path = Path(trust_path) if trust_path else self.root / ".agent" / TRUST_FILE

        self._lock: Any = None
        self._pending: Optional[Dict[str, Any]] = None
        self._decision: Optional[Dict[str, Any]] = None
        self._session_all = False
        self._session_tools: set[str] = set()
        self._trust: Dict[str, Dict[str, Any]] = self._load_trust()

        # 确认策略规则表（trust.py）：在弹窗之前生效——deny 静默拒 / allow 静默过，
        # 只有 ask（或无匹配）才弹人工确认。
        from trust import TrustStore

        self.policies = TrustStore(self.root / ".agent" / "trust-rules.json")

        import threading

        self._lock = threading.Lock()
        self._wake = threading.Event()

    # ---------------- 信任规则 ----------------
    def _load_trust(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.trust_path.read_text(encoding="utf-8"))
            rules = data.get("rules") if isinstance(data, dict) else None
            return rules if isinstance(rules, dict) else {}
        except Exception:
            return {}

    def _save_trust(self) -> None:
        try:
            self.trust_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "rules": self._trust}
            self.trust_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    @staticmethod
    def pattern_of(tool: str, args: Dict[str, Any]) -> str:
        """参数模式：只取参数名，不取取值（连续 pip install 才算同一类）。"""
        return f"{tool}({','.join(sorted(args))})"

    def trust_rules(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._trust)

    def _bump(self, pattern: str) -> None:
        rec = self._trust.get(pattern) or {"count": 0, "trusted": False}
        rec["count"] = int(rec.get("count", 0)) + 1
        rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if rec["count"] >= TRUST_THRESHOLD:
            rec["trusted"] = True
        self._trust[pattern] = rec
        self._save_trust()

    # ---------------- registry.approver 接口 ----------------
    def needs(self, spec: "ToolSpec", args: Dict[str, Any]) -> bool:
        if self.auto_yes:
            return False
        if getattr(spec, "require_confirm", False):
            return True
        if self.confirm_writes and spec.name in _WRITE_TOOLS:
            return True
        # 电脑作用域：写工作区之外的文件必须弹确认（即使该工具平时免确认）
        if self.sandbox.scope == "computer" and spec.name in _WRITE_TOOLS:
            if self._writes_outside(args):
                return True
        return False

    def _writes_outside(self, args: Dict[str, Any]) -> bool:
        """写路径解析后是否落在工作区外（电脑作用域专用）。"""
        raw = str(args.get("path") or args.get("dst") or "").strip()
        if not raw:
            return False
        try:
            target = self.sandbox.resolve(raw, for_write=True)
        except Exception:
            return False  # 沙箱会拦的路径不需要弹窗，直接让它报错
        return not self.sandbox.is_inside(target)

    def request(self, spec: "ToolSpec", args: Dict[str, Any]) -> Dict[str, Any]:
        tool = spec.name
        pattern = self.pattern_of(tool, args)

        if self.auto_yes or self._session_all or tool in self._session_tools:
            return {"decision": "approve"}

        # ---- 确认策略规则表（优先于旧版计数信任）----
        action, policy = self.policies.effective(tool, args)
        if policy is not None and action == "deny":
            note = f"：{policy.note}" if policy.note else ""
            self._emit(
                {
                    "event": "log",
                    "level": "warn",
                    "message": f"{tool} 命中确认策略（规则 {policy.id} → 拒绝）{note}",
                }
            )
            return {
                "decision": "deny",
                "reason": f"操作被信任策略拒绝（规则 {policy.id} 匹配 {policy.tool}）",
            }
        if policy is not None and action == "allow":
            self.policies.record_hit(policy.id)
            self._emit(
                {
                    "event": "log",
                    "level": "info",
                    "message": f"{tool} 命中确认策略（规则 {policy.id} → 放行，"
                    f"累计 {policy.hits} 次）",
                }
            )
            return {"decision": "approve", "rule": policy.id}

        # ---- 旧版计数信任（无策略命中时的兜底自动晋升）----
        rule = self._trust.get(pattern) or {}
        if rule.get("trusted"):
            self._emit(
                {
                    "event": "log",
                    "level": "info",
                    "message": f"{pattern} 命中信任规则（已确认 {rule.get('count')} 次），自动放行",
                }
            )
            return {"decision": "approve"}

        cid = uuid.uuid4().hex[:8]
        preview = build_preview(tool, args, self.sandbox)
        desc_lines = (spec.description or "").splitlines()
        with self._lock:
            self._pending = {"cid": cid, "tool": tool, "pattern": pattern}
            self._decision = None
        self._wake.clear()

        self._emit(
            {
                "event": "need_confirm",
                "cid": cid,
                "tool": tool,
                "args": args,
                "pattern": pattern,
                "danger": "high" if tool in _HIGH_RISK else "medium",
                "description": desc_lines[0][:160] if desc_lines else f"调用 {tool}",
                "preview": preview,
                "timeout_s": self.timeout_s,
                "trust_count": int(rule.get("count", 0)),
                "trust_threshold": TRUST_THRESHOLD,
            }
        )

        deadline = time.time() + self.timeout_s
        decision: Optional[Dict[str, Any]] = None
        while time.time() < deadline:
            if self._wake.wait(0.2):
                with self._lock:
                    decision = self._decision
                    self._pending = None
                    self._decision = None
                break
            if self.control is not None and self.control.should_stop():
                break

        if decision is None:
            stopped = self.control is not None and self.control.should_stop()
            reason = "任务已停止，操作未执行" if stopped else f"超过 {self.timeout_s}s 未确认，已自动拒绝"
            decision = {"decision": "deny", "reason": reason, "timeout": not stopped}
            with self._lock:
                self._pending = None

        verdict = str(decision.get("decision") or "deny")
        if verdict in ("approve", "approve_session"):
            self._bump(pattern)
            # 策略表命中 ask 规则且被人工批准：计数，达到阈值自动晋升 allow
            if policy is not None and policy.action == "ask":
                promoted_rule = self.policies.record_hit(policy.id)
                if promoted_rule is not None and promoted_rule.action == "allow":
                    self._emit(
                        {
                            "event": "log",
                            "level": "info",
                            "message": f"规则 {policy.id} 已自动晋升为放行"
                            f"（累计批准 {promoted_rule.hits} 次）",
                        }
                    )
            if verdict == "approve_session":
                if tool in ("__all__",):
                    self._session_all = True
                else:
                    self._session_tools.add(tool)

        self._emit(
            {
                "event": "confirm_result",
                "cid": cid,
                "tool": tool,
                "decision": verdict,
                "reason": decision.get("reason") or "",
                "timeout": bool(decision.get("timeout")),
                "pattern": pattern,
                "trust": self._trust.get(pattern, {}),
                "rule": (policy.id if policy is not None else None),  # 审计追责用
            }
        )
        out: Dict[str, Any] = {"decision": "approve" if verdict in ("approve", "approve_session") else "deny"}
        if decision.get("args"):
            out["args"] = decision["args"]
        if decision.get("reason"):
            out["reason"] = decision["reason"]
        return out

    # ---------------- HTTP 层回填 ----------------
    def resolve(
        self,
        cid: str,
        decision: str,
        remember: bool = False,
        args: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            if cid and cid != self._pending.get("cid"):
                return False
            if self._decision is not None:
                return False
            if remember and decision == "approve":
                decision = "approve_session"
            self._decision = {
                "decision": decision,
                "args": args or {},
                "reason": reason,
            }
        self._wake.set()
        return True

    @property
    def waiting(self) -> bool:
        with self._lock:
            return self._pending is not None


# ---------------- 变更预览 ----------------
def build_preview(tool: str, args: Dict[str, Any], sandbox: "Sandbox") -> Dict[str, Any]:
    """生成人类能看懂的「这一步会动什么」。diff / 命令 / 文件清单 / 纯文本。"""
    root = sandbox.root
    try:
        if tool == "write_file":
            rel = str(args.get("path") or "")
            new = str(args.get("content") or "")
            append = bool(args.get("append"))
            target = sandbox.resolve(rel)
            old = ""
            existed = target.is_file()
            if existed and target.stat().st_size <= 512 * 1024:
                old = target.read_text(encoding="utf-8", errors="replace")
            if append:
                body = _short(old, 2000) + "\n--- 以下为追加内容 ---\n" + _short(new, 4000)
                return {
                    "kind": "text",
                    "target": target.relative_to(root).as_posix(),
                    "title": f"{'追加' if append else '写入'} {target.relative_to(root).as_posix()}",
                    "body": body,
                    "stat": f"追加 {len(new)} 字符",
                }
            if not existed:
                lines = new.splitlines() or [""]
                diff = "\n".join(f"+ {ln}" for ln in lines[:200])
                return {
                    "kind": "diff",
                    "target": target.relative_to(root).as_posix(),
                    "title": f"新建文件 {target.relative_to(root).as_posix()}",
                    "body": diff,
                    "stat": f"新增 {len(lines)} 行",
                }
            diff_lines = list(
                difflib.unified_diff(
                    old.splitlines(),
                    new.splitlines(),
                    fromfile="当前",
                    tofile="写入后",
                    lineterm="",
                    n=3,
                )
            )
            added = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
            removed = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
            return {
                "kind": "diff",
                "target": target.relative_to(root).as_posix(),
                "title": f"修改 {target.relative_to(root).as_posix()}",
                "body": _short("\n".join(diff_lines) or "（内容无变化）", 6000),
                "stat": f"+{added} / -{removed} 行",
            }

        if tool == "delete_file":
            rel = str(args.get("path") or "")
            target = sandbox.resolve(rel)
            files, total = [], 0
            if target.is_file():
                files = [target]
                total = target.stat().st_size
            elif target.is_dir():
                for p in target.rglob("*"):
                    if p.is_file():
                        files.append(p)
                        total += p.stat().st_size
                        if len(files) > 500:
                            break
            listing = "\n".join(
                f"- {p.relative_to(root).as_posix()} ({p.stat().st_size} B)" for p in files[:40]
            )
            more = f"\n… 共 {len(files)} 个文件" if len(files) > 40 else ""
            return {
                "kind": "files",
                "target": target.relative_to(root).as_posix(),
                "title": f"删除 {target.relative_to(root).as_posix()}",
                "body": (listing or "（空）") + more,
                "stat": f"{len(files)} 个文件 / {total / 1024:.1f} KB",
                "warning": "删除不可撤销（除非有快照）",
            }

        if tool == "run_shell":
            cmd = str(args.get("cmd") or "")
            warning = ""
            try:
                sandbox.check_command(cmd)
            except Exception as e:
                warning = f"沙箱会拒绝这条命令：{e}"
            return {
                "kind": "command",
                "target": "",
                "title": "执行 shell 命令",
                "body": cmd,
                "stat": f"超时 {args.get('timeout', 30)}s",
                "warning": warning,
            }

        if tool == "run_python":
            return {
                "kind": "code",
                "target": "",
                "title": "执行 Python 片段",
                "body": _short(str(args.get("code") or ""), 6000),
                "stat": f"超时 {args.get('timeout', 15)}s",
            }

        if tool == "move_file":
            return {
                "kind": "files",
                "target": str(args.get("dst") or ""),
                "title": "移动 / 重命名",
                "body": f"{args.get('src')}  →  {args.get('dst')}",
                "stat": "",
            }

        if tool == "xhs_publish":
            body = (
                f"标题：{args.get('title') or ''}\n"
                f"正文：{_short(str(args.get('content') or ''), 1200)}\n"
                f"图片：{args.get('images') or ''}\n"
                f"是否真正提交：{args.get('submit')}"
            )
            return {
                "kind": "text",
                "target": "",
                "title": "小红书发布操作",
                "body": body,
                "stat": "发布后难以撤回",
            }
    except Exception as e:
        return {
            "kind": "args",
            "target": "",
            "title": f"{tool} 预览失败",
            "body": json.dumps(args, ensure_ascii=False, indent=2),
            "stat": f"{type(e).__name__}: {e}",
        }

    return {
        "kind": "args",
        "target": "",
        "title": f"调用 {tool}",
        "body": json.dumps(args, ensure_ascii=False, indent=2),
        "stat": "",
    }
