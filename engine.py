"""工具注册与执行引擎（T1）。

核心原则：
1. 模型只负责“选工具 + 填参数”，不能自己拼命令字符串；
2. 每个工具带 JSON Schema，调用前强制校验/类型转换，参数错了把错误回灌让模型重试；
3. 所有工具统一返回 {ok, output, error, meta}，模型才能看懂执行结果。
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_STR_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "Any": "string",
}


class ToolArgError(ValueError):
    """工具参数不合法（会被回灌给模型用于自我修正）。"""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]
    dangerous: bool = False
    require_confirm: bool = False


def _annotation_to_type(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return "string"
    if not isinstance(ann, str):
        return _TYPE_MAP.get(ann, "string")

    s = ann.strip()
    if s.startswith("Optional[") and s.endswith("]"):
        s = s[9:-1].strip()
    if s.startswith(("List[", "list[", "Sequence[")):
        return "array"
    if s.startswith(("Dict[", "dict[", "Mapping[")):
        return "object"
    return _STR_TYPE_MAP.get(s, "string")


def _schema_from_signature(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []

    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name.startswith("_"):
            continue  # 内部形参（如 _cancel 取消令牌），不进给模型的 Schema
        type_name = _annotation_to_type(p.annotation)
        prop: Dict[str, Any] = {"type": type_name}
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        elif isinstance(p.default, (str, int, float, bool)) or p.default is None:
            prop["default"] = p.default
        props[p.name] = prop

    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _coerce(value: Any, type_name: str, key: str) -> Any:
    """按 schema 做宽松类型转换（模型常把 20 写成 "20"）。"""
    if type_name == "string":
        return value if isinstance(value, str) else str(value)
    if type_name == "integer":
        if isinstance(value, bool):
            raise ToolArgError(f"参数 {key} 应为整数，收到布尔值")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                raise ToolArgError(f"参数 {key} 应为整数，收到 {value!r}")
        raise ToolArgError(f"参数 {key} 应为整数，收到 {type(value).__name__}")
    if type_name == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise ToolArgError(f"参数 {key} 应为数字，收到 {value!r}")
        raise ToolArgError(f"参数 {key} 应为数字，收到 {type(value).__name__}")
    if type_name == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "y"):
                return True
            if low in ("false", "0", "no", "n"):
                return False
        raise ToolArgError(f"参数 {key} 应为布尔值，收到 {value!r}")
    if type_name == "array":
        if isinstance(value, list):
            return value
        raise ToolArgError(f"参数 {key} 应为数组，收到 {type(value).__name__}")
    if type_name == "object":
        if isinstance(value, dict):
            return value
        raise ToolArgError(f"参数 {key} 应为对象，收到 {type(value).__name__}")
    return value


class ToolRegistry:
    """工具注册表：注册、描述、校验、执行、审计。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        # 终端确认回调：(spec, args) -> bool；为 None 时高危工具一律拒绝
        self.confirmer: Optional[Callable[[ToolSpec, Dict[str, Any]], bool]] = None
        # 图形界面确认闸门（优先级高于 confirmer）：
        #   .needs(spec, args) -> bool
        #   .request(spec, args) -> {"decision": approve|deny, "args": {...}?, "reason": str?}
        self.approver: Optional[Any] = None
        # 审计回调：每调用一次工具记录一条
        self.audit: Optional[Callable[[Dict[str, Any]], None]] = None
        # 取消令牌：声明了 _cancel 形参的工具会收到它（点「停止」时立刻生效）
        self.cancel_token: Optional[Any] = None

    # ---------- 注册 ----------
    def register(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dangerous: bool = False,
        require_confirm: bool = False,
    ):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            if tool_name in self._tools:
                raise ValueError(f"工具重名: {tool_name}")
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=(description or inspect.getdoc(fn) or "").strip(),
                parameters=_schema_from_signature(fn),
                fn=fn,
                dangerous=dangerous,
                require_confirm=require_confirm,
            )
            return fn

        return deco

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def add_spec(self, spec: ToolSpec) -> None:
        """直接挂载一个已构造好的工具（技能库等动态来源用）。"""
        if spec.name in self._tools:
            raise ValueError(f"工具重名: {spec.name}")
        self._tools[spec.name] = spec

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def names(self) -> List[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "dangerous": t.dangerous,
            }
            for t in self._tools.values()
        ]

    def to_prompt(self) -> str:
        """渲染成模型可读的工具清单（含参数 schema）。"""
        lines = ["可用工具（只能从中选择，参数必须符合 schema）："]
        for t in self._tools.values():
            flag = " [高危]" if t.dangerous else ""
            lines.append(f"- {t.name}{flag}: {t.description}")
            lines.append(f"  参数: {json.dumps(t.parameters, ensure_ascii=False)}")
        return "\n".join(lines)

    def to_brief(self, max_desc: int = 60) -> str:
        """精简清单（只给名字+一句说明），规划阶段用，避免撑爆上下文。"""
        lines = []
        for t in self._tools.values():
            first = (t.description or "").splitlines()[0][:max_desc]
            lines.append(f"- {t.name}: {first}")
        return "\n".join(lines)

    # ---------- 校验 ----------
    def validate_args(self, spec: ToolSpec, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolArgError("args 必须是 JSON 对象")

        schema = spec.parameters
        props: Dict[str, Any] = schema.get("properties", {})
        allowed = set(props)
        unknown = set(args) - allowed
        if unknown:
            raise ToolArgError(
                f"存在未定义参数 {sorted(unknown)}，本工具只接受 {sorted(allowed)}"
            )

        missing = [k for k in schema.get("required", []) if k not in args]
        if missing:
            raise ToolArgError(f"缺少必填参数 {missing}")

        cleaned: Dict[str, Any] = {}
        for key, value in args.items():
            cleaned[key] = _coerce(value, props[key]["type"], key)

        # 补默认值
        for key, prop in props.items():
            if key not in cleaned and "default" in prop:
                cleaned[key] = prop["default"]
        return cleaned

    # ---------- 执行 ----------
    def invoke(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        spec = self._tools.get(name)
        if spec is None:
            return self._result(
                False,
                error=f"工具 {name} 不存在。可用工具：{self.names()}",
                meta={"tool": name},
            )

        try:
            cleaned = self.validate_args(spec, args)
        except ToolArgError as e:
            return self._result(
                False, error=f"参数错误：{e}", meta={"tool": name, "args": args}
            )

        deny_reason = ""
        gate = self.approver
        if gate is not None and (spec.require_confirm or gate.needs(spec, args)):
            verdict = gate.request(spec, cleaned) or {}
            if str(verdict.get("decision")) == "approve":
                # 用户可能在弹窗里手动改了参数（例如路径写错了），改完要重新校验
                if verdict.get("args"):
                    try:
                        cleaned = self.validate_args(spec, verdict["args"])
                    except ToolArgError as e:
                        return self._result(
                            False,
                            error=f"用户修改后的参数不合法：{e}",
                            meta={"tool": name, "args": verdict.get("args"), "denied": True},
                        )
            else:
                deny_reason = str(verdict.get("reason") or "用户拒绝执行该操作")
        elif spec.require_confirm:
            if self.confirmer is None:
                deny_reason = "该操作需要人工确认，但当前为非交互模式，已拒绝执行"
            elif not self.confirmer(spec, cleaned):
                deny_reason = "用户拒绝执行该操作"

        if deny_reason:
            result = self._result(
                False,
                error=(
                    f"{deny_reason}。请换一种不需要该操作的做法；"
                    "如果确实必须这么做，请在最终答复里说明原因，不要重复同样的调用。"
                ),
                meta={"tool": name, "args": cleaned, "denied": True},
            )
            if self.audit is not None:
                try:
                    self.audit(
                        {
                            "tool": name,
                            "args": cleaned,
                            "ok": False,
                            "error": deny_reason,
                            "output_len": 0,
                            "dangerous": spec.dangerous,
                            "denied": True,
                        }
                    )
                except Exception:
                    pass
            return result

        try:
            call_kwargs = dict(cleaned)
            if self.cancel_token is not None:
                from cancel import accepts_cancel

                if accepts_cancel(spec.fn):
                    call_kwargs["_cancel"] = self.cancel_token
            raw = spec.fn(**call_kwargs)
            result = self._normalize(raw)
            if isinstance(raw, dict) and raw.get("_cancelled"):
                result["meta"]["cancelled"] = True
        except ToolArgError as e:
            result = self._result(False, error=f"参数错误：{e}")
        except Exception as e:  # 工具内部异常不能打断整个任务
            name_e = type(e).__name__
            if name_e == "CancelledError":
                result = self._result(False, error=f"已取消：{e}", meta={"cancelled": True})
            else:
                result = self._result(False, error=f"{name_e}: {e}")

        result.setdefault("meta", {})
        result["meta"].update({"tool": name, "args": cleaned})

        if self.audit is not None:
            try:
                self.audit(
                    {
                        "tool": name,
                        "args": cleaned,
                        "ok": result["ok"],
                        "error": result.get("error"),
                        "output_len": len(result.get("output") or ""),
                        "dangerous": spec.dangerous,
                    }
                )
            except Exception:
                pass  # 审计失败不影响主流程
        return result

    @staticmethod
    def _result(ok: bool, output: str = "", error: Optional[str] = None, meta=None) -> Dict[str, Any]:
        return {"ok": ok, "output": output, "error": error, "meta": meta or {}}

    @classmethod
    def _normalize(cls, raw: Any) -> Dict[str, Any]:
        """把工具返回值统一成 {ok, output, error}。"""
        if isinstance(raw, dict):
            if "ok" in raw or "output" in raw or "error" in raw:
                return {
                    "ok": bool(raw.get("ok", raw.get("error") in (None, ""))),
                    "output": raw.get("output", ""),
                    "error": raw.get("error"),
                    "meta": {k: v for k, v in raw.items() if k not in ("ok", "output", "error")},
                }
            return {"ok": True, "output": json.dumps(raw, ensure_ascii=False), "error": None, "meta": {}}
        if isinstance(raw, str):
            return {"ok": True, "output": raw, "error": None, "meta": {}}
        if hasattr(raw, "ok"):  # 兼容旧的 ToolResult
            return {
                "ok": bool(raw.ok),
                "output": "" if raw.data is None else str(raw.data),
                "error": getattr(raw, "error", None),
                "meta": {},
            }
        return {"ok": True, "output": str(raw), "error": None, "meta": {}}
