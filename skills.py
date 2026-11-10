"""技能库（T3）：把成功的操作序列固化成可复用的工具。

核心约束：
1. 技能 = 「工具序列 + 参数占位」，不是可执行代码。模型只能填参数，不能改流程；
2. 技能执行时仍然走 ToolRegistry.invoke，所以 schema 校验、高危确认、审计
   全部保留，技能不能成为绕过安全层的后门；
3. 不允许技能嵌套调用技能（避免无限递归）。
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from engine import ToolRegistry, ToolSpec

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,40}$")
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_QUOTED_RE = re.compile(r"[「『“\"']([^」』”\"'\n]{2,40})[」』”\"']")
_FILE_RE = re.compile(r"[\w./\\-]+\.(?:py|md|txt|json|csv|ya?ml|html|js|ts|log)")

_local = threading.local()


class SkillError(ValueError):
    """技能定义或调用不合法。"""


@dataclass
class Skill:
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source: str = "manual"
    uses: int = 0
    successes: int = 0
    created_at: str = ""
    goal: str = ""

    @property
    def tool_name(self) -> str:
        return f"skill_{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "steps": self.steps,
            "source": self.source,
            "uses": self.uses,
            "successes": self.successes,
            "created_at": self.created_at,
            "goal": self.goal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        name = str(data.get("name") or "").strip()
        if not _NAME_RE.match(name):
            raise SkillError(
                f"技能名 {name!r} 不合法：只能用小写字母开头、字母数字下划线，2-41 字符"
            )
        steps = data.get("steps") or []
        if not isinstance(steps, list) or not steps:
            raise SkillError(f"技能 {name} 没有任何步骤")
        parsed: List[Dict[str, Any]] = []
        for i, st in enumerate(steps, 1):
            if not isinstance(st, dict) or not st.get("tool"):
                raise SkillError(f"技能 {name} 第 {i} 步缺少 tool")
            parsed.append({"tool": str(st["tool"]), "args": st.get("args") or {}})
        return cls(
            name=name,
            description=str(data.get("description") or f"技能 {name}"),
            steps=parsed,
            params=data.get("params") or {},
            source=str(data.get("source") or "manual"),
            uses=int(data.get("uses") or 0),
            successes=int(data.get("successes") or 0),
            created_at=str(data.get("created_at") or ""),
            goal=str(data.get("goal") or ""),
        )

    def schema(self) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        required: List[str] = []
        for key, spec in self.params.items():
            spec = spec or {}
            props[key] = {
                "type": spec.get("type", "string"),
                "description": spec.get("description", ""),
            }
            if "default" in spec:
                props[key]["default"] = spec["default"]
            if spec.get("required"):
                required.append(key)
        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }

    def describe(self) -> str:
        head = f"{self.description}"
        if self.steps:
            flow = " → ".join(st["tool"] for st in self.steps[:8])
            head += f"\n  固定流程：{flow}"
        if self.params:
            head += "\n  参数：" + ", ".join(
                f"{k}({'必填' if v.get('required') else '可选'})" for k, v in self.params.items()
            )
        head += f"\n  历史：调用 {self.uses} 次，成功 {self.successes} 次"
        return head


# ---------------- 文件读写 ----------------
def skill_dir(root: str | Path) -> Path:
    d = Path(root).expanduser() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_skills(root: str | Path) -> List[Skill]:
    out: List[Skill] = []
    for path in sorted(skill_dir(root).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            skill = Skill.from_dict(data)
        except Exception as e:
            print(f"[skills] 跳过 {path.name}：{type(e).__name__}: {e}")
            continue
        skill._path = path  # type: ignore[attr-defined]
        out.append(skill)
    return out


def save_skill(root: str | Path, skill: Skill) -> Path:
    path = skill_dir(root) / f"{skill.name}.json"
    path.write_text(
        json.dumps(skill.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    skill._path = path  # type: ignore[attr-defined]
    return path


def delete_skill(root: str | Path, name: str) -> bool:
    path = skill_dir(root) / f"{name}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


def _persist_usage(skill: Skill, ok: bool, root: Optional[str | Path]) -> None:
    skill.uses += 1
    if ok:
        skill.successes += 1
    if root is not None:
        try:
            save_skill(root, skill)
        except Exception:
            pass


# ---------------- 执行 ----------------
def _substitute(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in params:
                raise SkillError(f"技能参数缺失：{key}（可用：{sorted(params)}）")
            return str(params[key])

        return _PLACEHOLDER_RE.sub(repl, value)
    if isinstance(value, list):
        return [_substitute(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, params) for k, v in value.items()}
    return value


def run_skill(
    skill: Skill,
    params: Dict[str, Any],
    registry: ToolRegistry,
    root: Optional[str | Path] = None,
    max_output: int = 600,
) -> Dict[str, Any]:
    """按定义顺序执行技能；任何一步失败默认中止，避免半途状态。"""
    if getattr(_local, "depth", 0) > 0:
        return {"ok": False, "output": "", "error": "技能内不允许再调用其他技能"}
    _local.depth = getattr(_local, "depth", 0) + 1

    lines: List[str] = []
    ok_all = True
    try:
        for i, st in enumerate(skill.steps, 1):
            tool = st["tool"]
            if str(tool).startswith("skill_"):
                return {"ok": False, "output": "", "error": "技能步骤里不能包含 skill_* 调用"}
            try:
                args = _substitute(st.get("args") or {}, params)
            except SkillError as e:
                return {"ok": False, "output": "\n".join(lines), "error": str(e)}

            res = registry.invoke(tool, args)
            ok = bool(res.get("ok"))
            lines.append(
                f"步骤 {i}/{len(skill.steps)} {tool} {'成功' if ok else '失败'}"
            )
            body = (res.get("output") or "").strip()
            if body:
                lines.append("  " + body[:max_output].replace("\n", "\n  "))
            if not ok:
                ok_all = False
                lines.append(f"  错误：{res.get('error')}")
                lines.append("已中止后续步骤（技能默认遇错即停）")
                break
        _persist_usage(skill, ok_all, root)
        return {"ok": ok_all, "output": "\n".join(lines), "error": None if ok_all else "技能执行失败"}
    finally:
        _local.depth = getattr(_local, "depth", 0) - 1


def register_skills(
    registry: ToolRegistry,
    skills: List[Skill],
    root: Optional[str | Path] = None,
) -> Dict[str, str]:
    """把技能挂进注册表，返回 {技能名: 状态}。"""
    report: Dict[str, str] = {}
    for skill in skills:
        missing = [st["tool"] for st in skill.steps if registry.get(st["tool"]) is None]
        if missing:
            report[skill.name] = f"跳过：依赖的工具不存在 {sorted(set(missing))}"
            continue
        if registry.get(skill.tool_name) is not None:
            report[skill.name] = "跳过：同名工具已存在"
            continue

        dangerous = any(
            registry.get(st["tool"]).dangerous for st in skill.steps  # type: ignore[union-attr]
        )
        spec = ToolSpec(
            name=skill.tool_name,
            description=skill.describe(),
            parameters=skill.schema(),
            fn=(lambda _s=skill, **kw: run_skill(_s, kw, registry, root)),
            dangerous=dangerous,
            require_confirm=False,  # 内部高危工具自己会触发确认
        )
        try:
            registry.add_spec(spec)
        except ValueError as e:
            report[skill.name] = f"跳过：{e}"
            continue
        report[skill.name] = "已注册"
    return report


# ---------------- 从轨迹结晶 ----------------
def _detect_input(goal: str) -> Optional[str]:
    m = _QUOTED_RE.search(goal or "")
    if m:
        return m.group(1).strip()
    m = _FILE_RE.search(goal or "")
    return m.group(0) if m else None


def crystallize(
    name: str,
    goal: str,
    trace: List[Dict[str, Any]],
    description: str = "",
    source: str = "crystallized",
) -> Skill:
    """把成功轨迹变成技能。只收录成功且参数可复现的步骤。"""
    steps: List[Dict[str, Any]] = []
    prev_sig = None
    for item in trace:
        if not item.get("ok"):
            continue
        tool = str(item.get("tool") or "")
        if not tool or tool.startswith("(") or tool.startswith("skill_"):
            continue
        args = item.get("args") or {}
        sig = json.dumps([tool, args], ensure_ascii=False, sort_keys=True)
        if sig == prev_sig:
            continue
        prev_sig = sig
        steps.append({"tool": tool, "args": args})
        if len(steps) >= 12:
            break

    if not steps:
        raise SkillError("轨迹里没有可固化的成功步骤")

    input_value = _detect_input(goal)
    params: Dict[str, Dict[str, Any]] = {}
    if input_value:
        replaced = False
        for st in steps:
            for k, v in list((st.get("args") or {}).items()):
                if isinstance(v, str) and v == input_value:
                    st["args"][k] = "{input}"
                    replaced = True
                elif isinstance(v, str) and input_value in v and len(v) < 300:
                    st["args"][k] = v.replace(input_value, "{input}")
                    replaced = True
        if replaced:
            params["input"] = {
                "type": "string",
                "required": True,
                "description": f"目标里要处理的对象，例如 {input_value!r}",
            }

    return Skill(
        name=name,
        description=description or f"固化自任务：{goal[:60]}",
        steps=steps,
        params=params,
        source=source,
        created_at=datetime.now().isoformat(timespec="seconds"),
        goal=goal[:300],
    )
