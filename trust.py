"""确认策略规则表：把「哪些操作要问、哪些直接放」变成用户可编辑的数据。

设计决策（为什么这样而不是那样）：

1. 规则是**有序 list** 而非 dict：优先级就是语义——「rm -rf 之类拒绝」必须排在
   「run_shell 询问」前面，dict 没有顺序就没法表达。
2. 匹配用 glob（fnmatch）而不是正则：够表达 read_*、pip install * 这类意图，
   且用户手滑也不会写出灾难性回溯。参数值按字符串做 glob 匹配。
3. condition 是受限 eval：只暴露参数键 + len/str/int/float/abs/min/max，
   builtins 清空。允许写 "timeout < 60"，不允许写 "import os"。
4. 命中即计数：ask 规则被用户批准 N 次后自动晋升 allow（默认 3 次），
   这是「从烦人到好用」的分水岭；每次命中都返回 rule_id，审计日志可追责。

与 confirm_gate.ConfirmBroker 的关系：规则表在弹窗**之前**生效——
deny 静默拒绝、allow 静默放行，只有 ask（或无匹配）才走到人工确认。
旧的 trust.json（按参数名集合计数）继续保留，作为「没有规则命中时」的
兜底自动晋升机制，两者不冲突。
"""
from __future__ import annotations

import fnmatch
import json
import time
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
RULES_FILE = "trust-rules.json"
DEFAULT_PROMOTE_AFTER = 3

# 受限求值的白名单（eval_condition 用）
_SAFE_BUILTINS = {"len": len, "str": str, "int": int, "float": float,
                  "abs": abs, "min": min, "max": max, "round": round}


def eval_condition(expr: str, args: Dict[str, Any]) -> bool:
    """受限条件求值：只允许参数键和少量内建函数。任何异常都按 False 处理。"""
    if not expr or not expr.strip():
        return True
    scope: Dict[str, Any] = {"__builtins__": {}}
    scope.update(_SAFE_BUILTINS)
    for k, v in args.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            scope.setdefault(k, v)
    try:
        return bool(eval(expr, scope, {}))  # noqa: S307 —— 表达式来自本机用户
    except Exception:
        return False


@dataclass
class TrustRule:
    id: str
    tool: str = "*"  # 工具名 glob，如 "run_shell"、"read_*"
    param_pattern: Dict[str, str] = field(default_factory=dict)
    # 参数级 glob，如 {"cmd": "pip install *"}
    action: str = "ask"  # allow | deny | ask
    enabled: bool = True
    scope: str = "global"  # global 持久化；session 只存在内存（重启失效）
    hits: int = 0
    auto_promote_after: int = DEFAULT_PROMOTE_AFTER  # ask 命中 N 次自动升 allow；0=关
    condition: str = ""  # 受限表达式，如 "timeout < 60"
    note: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # 临时授权：到期自动失效（Unix 秒）
    promoted: bool = False  # 是否因自动晋升变过 allow（界面提示用）

    # ---------------- 匹配 ----------------
    def matches(self, tool: str, args: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.expires_at is not None and time.time() > self.expires_at:
            return False
        if not fnmatch.fnmatch(tool, self.tool):
            return False
        for key, pattern in (self.param_pattern or {}).items():
            if key not in args:
                return False
            if not fnmatch.fnmatch(str(args[key]), str(pattern)):
                return False
        return eval_condition(self.condition, args or {})

    # ---------------- 序列化 ----------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustRule":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


# 内置默认规则：用户看到的不该是一张空表。顺序即优先级。
def default_rules() -> List[TrustRule]:
    now = time.time()
    return [
        TrustRule(
            id="r-deny-destructive",
            tool="run_shell",
            param_pattern={"cmd": "*rm -rf*"},
            action="deny",
            note="rm -rf 一律拒绝（要删东西请用 delete_file，走工作区沙箱）",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-deny-dangerous",
            tool="run_shell",
            param_pattern={"cmd": "*sudo*"},
            action="deny",
            note="提权命令拒绝",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-ask-shell",
            tool="run_shell",
            action="ask",
            note="shell 命令默认询问；批准 3 次后自动放行同类",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-ask-python",
            tool="run_python",
            action="ask",
            note="任意代码执行默认询问",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-ask-delete",
            tool="delete_file",
            action="ask",
            note="删除文件默认询问",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-allow-readonly",
            tool="read_*",
            action="allow",
            note="只读工具直接放行",
            created_at=now,
            updated_at=now,
        ),
        # ---- 电脑操控（--full-computer）----
        TrustRule(
            id="r-allow-sysread",
            tool="sys_*",
            action="allow",
            note="电脑作用域：系统查询/截屏/剪贴板读取等只读操作直接放行",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-allow-mousemove",
            tool="ui_mouse_move",
            action="allow",
            note="移动光标本身无副作用，放行（点击/输入仍然要确认）",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-ask-ui",
            tool="ui_*",
            action="ask",
            note="桌面动作（点击/输入/开程序/杀进程）默认询问；同类批准 3 次自动放行",
            created_at=now,
            updated_at=now,
        ),
        TrustRule(
            id="r-allow-search",
            tool="search*",
            action="allow",
            note="搜索类工具直接放行",
            created_at=now,
            updated_at=now,
        ),
    ]


class TrustStore:
    """规则仓库：加载 → 匹配 → 命中计数 → 原子写回。线程安全（一任务一 Broker，多线程 HTTP 会并发 CRUD）。"""

    def __init__(self, path: Path, with_defaults: bool = True) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.rules: List[TrustRule] = []
        self._sessions: List[TrustRule] = []  # scope=session 的规则，不落盘
        self.load(with_defaults=with_defaults)

    # ---------------- 持久化 ----------------
    def load(self, with_defaults: bool = True) -> None:
        self.rules = []
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("version") == SCHEMA_VERSION:
                    self.rules = [TrustRule.from_dict(r) for r in data.get("rules", [])]
            except Exception:
                self.rules = []
        if not self.rules and with_defaults:
            self.rules = default_rules()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": SCHEMA_VERSION, "rules": [r.to_dict() for r in self.rules]}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)  # 原子替换，防写坏
        except Exception:
            pass

    # ---------------- 匹配 ----------------
    def _all_rules(self) -> List[TrustRule]:
        return self.rules + self._sessions

    def first_match(self, tool: str, args: Dict[str, Any]) -> Optional[TrustRule]:
        for rule in self._all_rules():
            if rule.matches(tool, args):
                return rule
        return None

    def effective(self, tool: str, args: Dict[str, Any]) -> Tuple[str, Optional[TrustRule]]:
        """返回 (action, rule)。action: allow | deny | ask；无匹配 → ask。"""
        rule = self.first_match(tool, args)
        return (rule.action if rule else "ask", rule)

    # ---------------- 命中计数 / 自动晋升 ----------------
    def record_hit(self, rule_id: str) -> Optional[TrustRule]:
        """命中一次：计数；ask 规则到达阈值自动晋升 allow（落盘，重启仍生效）。"""
        with self._lock:
            for rule in self._all_rules():
                if rule.id == rule_id:
                    rule.hits += 1
                    rule.updated_at = time.time()
                    if (
                        rule.action == "ask"
                        and rule.auto_promote_after
                        and rule.hits >= rule.auto_promote_after
                    ):
                        rule.action = "allow"
                        rule.promoted = True
                        rule.note = (rule.note + "；" if rule.note else "") + \
                            f"已自动晋升（累计批准 {rule.hits} 次）"
                    if rule in self.rules:
                        self.save()
                    return rule
        return None

    # ---------------- CRUD（HTTP 层用）----------------
    def add(self, data: Dict[str, Any]) -> TrustRule:
        """新增规则。持久规则插到**最前面**（优先级最高，保证「加完立刻生效」，
        否则会被已有的宽泛规则挡住永远匹配不到）；顺序可在界面里再调。"""
        with self._lock:
            rule = TrustRule(
                id=f"r{int(time.time() * 1000) % 10**9:09d}",
                **{k: v for k, v in data.items() if k != "id" and k in TrustRule.__dataclass_fields__},
            )
            if rule.scope == "session":
                self._sessions.append(rule)
                return rule
            self.rules.insert(0, rule)
            self.save()
            return rule

    def update(self, rule_id: str, data: Dict[str, Any]) -> Optional[TrustRule]:
        with self._lock:
            for rule in self._all_rules():
                if rule.id == rule_id:
                    for k, v in data.items():
                        if k in ("id", "created_at"):
                            continue
                        if k in TrustRule.__dataclass_fields__:
                            setattr(rule, k, v)
                    rule.updated_at = time.time()
                    if rule in self.rules:
                        self.save()
                    return rule
        return None

    def delete(self, rule_id: str) -> bool:
        with self._lock:
            before = len(self._all_rules())
            self.rules = [r for r in self.rules if r.id != rule_id]
            self._sessions = [r for r in self._sessions if r.id != rule_id]
            removed = before != len(self._all_rules())
            if removed:
                self.save()
            return removed

    def reorder(self, ids: List[str]) -> None:
        """拖拽排序：给定新的规则 id 顺序（只对持久规则生效）。"""
        with self._lock:
            order = {rid: i for i, rid in enumerate(ids)}
            self.rules.sort(key=lambda r: order.get(r.id, len(ids) + 999))
            self.save()

    def reset(self) -> None:
        """清空所有规则（含会话），恢复内置默认。"""
        with self._lock:
            self.rules = default_rules()
            self._sessions = []
            self.save()

    # ---------------- 模拟匹配（前端「测试」按钮）----------------
    def dry_run(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        rule = self.first_match(tool, args or {})
        return {
            "tool": tool,
            "args": args,
            "matched": rule is not None,
            "rule": rule.to_dict() if rule else None,
            "action": rule.action if rule else "ask（默认：无匹配时人工确认）",
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ---------------- 导入 / 导出 ----------------
    def export_payload(self) -> Dict[str, Any]:
        return {"version": SCHEMA_VERSION, "rules": [r.to_dict() for r in self.rules]}

    def import_payload(self, payload: Dict[str, Any]) -> int:
        with self._lock:
            try:
                rules = payload.get("rules") or []
                self.rules = [TrustRule.from_dict(r) for r in rules if isinstance(r, dict)]
            except Exception:
                return 0
            self.save()
            return len(self.rules)

    def list_rules(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._all_rules()]
