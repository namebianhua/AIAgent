from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api_clients import ModelRouter
from control import RunCancelled
from engine import ToolRegistry
from gateway import ModelGateway
from web_router import WebModelRouter
from tools import AVAILABLE_TOOLS

TOOL_CALL_MARK = "TOOL_CALL:"

_FORMAT_RULES = """
【工作方式】
- 你只能调用下面注册表里的工具，禁止编造工具名或参数。
- 需要执行动作时，只输出一行：TOOL_CALL: {"tool":"工具名","args":{...}}
- 一次只调用一个工具，等待执行结果后再决定下一步。
- 信息足够时，直接用自然语言给出最终答案，不要再输出 TOOL_CALL。
- 工具报错时先修正参数或换一种做法，不要重复同样的失败调用。
""".strip()


@dataclass
class AgentConfig:
    system_prompt: str = "你是一个能操作电脑的 AI 助手。"
    max_turns: int = 10
    tools: Optional[List[Any]] = None
    max_steps: int = 15
    max_observation_chars: int = 4000
    # ---------------- RAG 被动注入（第二阶段）----------------
    use_rag: bool = True          # 任务开始时自动检索知识库并注入系统提示词
    rag_top_k: int = 3            # 注入几条知识片段
    # 相关度门槛：None = 沿用该知识库标定出的门槛（推荐）。
    # 余弦分数的绝对值不可跨模型比较，写死一个数字会在换 embedding 模型后失效
    # （要么噪音全过线，要么有效结果全被滤掉），所以默认交给库内标定值。
    rag_min_score: Optional[float] = None
    rag_max_chars: int = 2400     # 注入内容总长度上限
    rag_collections: Optional[List[str]] = None  # 留空用引擎默认库
    # ---------------- 经验召回注入（第四阶段）----------------
    use_experience: bool = True
    exp_top_k: int = 2
    exp_min_score: Optional[float] = None  # None = 沿用经验库标定门槛
    exp_max_chars: int = 1600


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "…[已截断]"


_QUOTE_MAP = str.maketrans(
    {
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u201e": '"',  # „
        "\u201f": '"',  # ‟
        "\u2033": '"',  # ″
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\uff5b": "{",  # ｛
        "\uff5d": "}",  # ｝
        "\uff3b": "[",  # ［
        "\uff3d": "]",  # ］
    }
)
# 仅在引号修复仍失败时使用（会改动字符串内容里的全角标点，属于最后手段）
_STRUCT_MAP = str.maketrans({"\uff1a": ":", "\uff0c": ",", "\u3000": " "})


def _try_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _iter_json_candidates(text: str):
    """按花括号配对枚举文本中的候选 JSON 片段（跳过字符串内的括号）。"""
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c in ('"', "\u201d"):
                    in_str = False
                continue
            if c in ('"', "\u201c"):
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i: j + 1]
                    break


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从任意模型输出里抠出第一个可解析的 JSON 对象。

    网页模型常见问题：前后夹说明文字、用中文全角引号、代码块包裹。
    这里做多级修复，供 Planner / Critic / 工具调用共用。
    """
    if not text:
        return None
    tries = 0
    for raw in _iter_json_candidates(text):
        tries += 1
        if tries > 20:
            break
        for fixed in (
            raw,
            raw.translate(_QUOTE_MAP),
            raw.translate(_QUOTE_MAP).translate(_STRUCT_MAP),
        ):
            obj = _try_json(fixed)
            if obj is not None:
                return obj
    return None


def _extract_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里抠出工具调用 JSON。

    真实网页模型常把引号输出成中文全角引号（“”），这里做多级修复后再解析。
    """
    idx = text.find(TOOL_CALL_MARK)
    if idx == -1:
        return None
    tail = text[idx + len(TOOL_CALL_MARK):].strip()

    candidates: List[str] = [tail]
    start, end = tail.find("{"), tail.rfind("}")
    if start != -1 and end > start:
        candidates.append(tail[start: end + 1])
    fenced = re.search(r"\{.*\}", tail, re.S)
    if fenced:
        candidates.append(fenced.group(0))

    for raw in candidates:
        for fixed in (raw, raw.translate(_QUOTE_MAP), raw.translate(_QUOTE_MAP).translate(_STRUCT_MAP)):
            obj = _try_json(fixed)
            if obj is not None:
                return obj
    return None


def _legacy_registry() -> ToolRegistry:
    """没提供注册表时，用旧的 AVAILABLE_TOOLS 兜底，保持向后兼容。"""
    reg = ToolRegistry()

    @reg.register()
    def search(query: str, max_results: int = 3) -> Any:
        """联网搜索关键词，返回若干条结果。"""
        return AVAILABLE_TOOLS["search"](query, max_results=max_results)

    @reg.register(dangerous=True)
    def run_python(code: str) -> Any:
        """执行一段 Python 代码并返回输出。"""
        return AVAILABLE_TOOLS["run_python"](code)

    return reg


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        api_router: Optional[ModelRouter] = None,
        web_router: Optional[WebModelRouter] = None,
        registry: Optional[ToolRegistry] = None,
        gateway: Optional[ModelGateway] = None,
        control: Optional[Any] = None,
        rag: Optional[Any] = None,
        experience: Optional[Any] = None,
    ):
        self.config = config
        self.api_router = api_router
        self.web_router = web_router
        self.gateway = gateway or ModelGateway(api_router, web_router)
        self.registry = registry or _legacy_registry()
        # 交互层控制：停止 / 暂停 / 失败干预（None 时退化为原来的无人值守行为）
        self.control = control
        # 知识库（第二阶段被动注入）与经验库（第四阶段自我进化）
        # 二者都是可选能力：为 None 时 Agent 行为与未接入 RAG 时完全一致
        self.rag = rag
        self.experience = experience
        self.last_rag_context = ""      # 本次任务实际注入了什么（可观测/可复盘）
        self.last_experience_context = ""

        # config.tools 可做工具白名单：["read_file", ...] 或 [{"name": ...}]
        if config.tools:
            names = {
                t if isinstance(t, str) else (t or {}).get("name") for t in config.tools
            }
            names.discard(None)
            for name in list(self.registry.names()):
                if name not in names:
                    self.registry._tools.pop(name, None)

    # ---------------- RAG 上下文构造（第二阶段 / 第四阶段）----------------
    def _build_rag_context(self, goal: str) -> str:
        """检索知识库，渲染成可注入系统提示词的「参考资料」块。

        失败一律返回空串：RAG 是增强项，不能因为它挂了就让任务起不来。
        """
        cfg = self.config
        if self.rag is None or not cfg.use_rag or not (goal or "").strip():
            return ""
        try:
            colls = cfg.rag_collections or []
            if len(colls) > 1:
                hits = self.rag.search_multi(goal, colls, top_k=cfg.rag_top_k,
                                             min_score=cfg.rag_min_score)
            else:
                hits = self.rag.search(
                    goal,
                    top_k=max(1, int(cfg.rag_top_k)),
                    collection_name=(colls[0] if colls else None),
                    min_score=cfg.rag_min_score,
                )
        except Exception:
            return ""
        if not hits:
            return ""
        body = self.rag.format_context(hits, max_chars=cfg.rag_max_chars)
        if not body:
            return ""
        return (
            "【本地知识库检索结果】以下是与本任务相关的资料片段，回答时优先据此作答，"
            "并在涉及具体结论时说明来源文件；片段不足以支撑的部分要明确指出缺口，"
            "禁止编造资料里没有的内容。若需要更多细节，可调用 rag_search 主动查询。\n"
            + _clip(body, cfg.rag_max_chars)
        )

    def _build_experience_context(self, goal: str) -> str:
        """召回历史任务经验（语义版），提示 Agent「以前类似的事是怎么做成的」。"""
        cfg = self.config
        if self.experience is None or not cfg.use_experience or not (goal or "").strip():
            return ""
        try:
            hits = self.experience.recall(goal, k=max(1, int(cfg.exp_top_k)),
                                          min_score=cfg.exp_min_score)
        except Exception:
            return ""
        if not hits:
            return ""
        cards = [h.to_card(i) for i, h in enumerate(hits, 1)]
        body = _clip("\n\n".join(cards), cfg.exp_max_chars)
        return (
            "【历史任务经验】以下是过去做过的相似任务（含当时的做法与踩过的坑），"
            "可作参考但不要照抄——当前环境可能已变化，优先依据实际观测决策。\n"
            + body
        )

    # ---------------- 模型调用 ----------------
    def _call_model(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self.gateway.chat(messages, **kwargs)

    def _note_parse(self, resp: Dict[str, Any], ok: bool) -> None:
        """把「模型产出能否解析」反馈给网关——本地小模型连续出错时自动重路由到大模型。"""
        hook = getattr(self.gateway, "note_parse_error" if not ok else "note_parse_success", None)
        if hook is None:
            return
        try:
            hook(resp.get("_source", "")) if not ok else hook()
        except Exception:
            pass

    # ---------------- 主循环 ----------------
    def run_task(
        self,
        goal: str,
        max_steps: Optional[int] = None,
        on_event: Optional[Any] = None,
        rag_context: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """目标驱动的执行循环：规划 → 选工具 → 执行 → 观察 → 反思 → 下一步。

        rag_context: 调用方预先算好的知识/经验注入块（自主模式整个任务只算一次，
                     避免每个计划步骤都重复检索）。传 None 时本函数自己检索。
        """
        steps_limit = int(max_steps or self.config.max_steps)

        # ---- 第二阶段：知识注入（带着知识回答）----
        if rag_context is None:
            knowledge = self._build_rag_context(goal)
            experience = self._build_experience_context(goal)
        else:
            knowledge, experience = rag_context, ""
        self.last_rag_context = knowledge
        self.last_experience_context = experience

        system_parts = [
            self.config.system_prompt,
            self.registry.to_prompt(),
            _FORMAT_RULES,
        ]
        if knowledge:
            system_parts.append(knowledge)
        if experience:
            system_parts.append(experience)
        system = "\n\n".join(p for p in system_parts if p)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": goal},
        ]

        if on_event is not None and (knowledge or experience):
            self._notify(
                on_event,
                {
                    "event": "rag_inject",
                    "knowledge_chars": len(knowledge),
                    "experience_chars": len(experience),
                    "sources": self._rag_sources(knowledge),
                },
            )

        trace: List[Dict[str, Any]] = []
        last_sig: Optional[str] = None
        last_ok = True
        repeats = 0
        bad_format = 0
        consecutive_failures = 0
        stop_reason = "max_steps"
        final_answer = ""
        model_name = "unknown"

        for step in range(1, steps_limit + 1):
            # 可中断点：停止则立刻收尾；暂停则原地等待（当前步已完成的观测已进 trace）
            if self.control is not None:
                try:
                    self.control.checkpoint()
                except RunCancelled:
                    stop_reason = "cancelled"
                    final_answer = "任务已被用户停止，已保留执行轨迹。"
                    break

            self._notify(on_event, {"event": "thinking", "turn": step, "model": model_name})
            resp = self._call_model(messages, **kwargs)
            model_name = resp.get("model", model_name)
            content = resp.get("content") or ""
            if not isinstance(content, str):
                content = str(content)

            chars = sum(len(str(m.get("content") or "")) for m in messages)
            self._notify(
                on_event,
                {
                    "event": "usage",
                    "turn": step,
                    "chars": chars,
                    "est_tokens": int(chars / 1.6),  # 中文场景的粗估，不追求精确
                    "model": model_name,
                },
            )

            if TOOL_CALL_MARK not in content:
                final_answer = content
                stop_reason = "finished"
                break

            action = _extract_tool_json(content)
            if action is None:
                # 模型想调工具但 JSON 不合法：把格式错误回灌，让它自己修
                bad_format += 1
                self._note_parse(resp, ok=False)
                trace.append(
                    {
                        "step": step,
                        "tool": "(格式错误)",
                        "args": {},
                        "ok": False,
                        "elapsed_ms": 0,
                        "output": content[:200],
                        "error": "工具调用 JSON 无法解析",
                    }
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[格式错误] 你的工具调用不是合法 JSON（检查是否用了中文引号、"
                            '缺少引号或花括号）。请只输出一行：'
                            'TOOL_CALL: {"tool":"工具名","args":{...}}，必须使用英文双引号。'
                        ),
                    }
                )
                if bad_format >= 3:
                    stop_reason = "bad_tool_format"
                    final_answer = "连续 3 次无法生成合法的工具调用 JSON，已中止。"
                    break
                continue

            tool_name = action.get("tool")
            tool_args = action.get("args") or {}
            self._note_parse(resp, ok=True)

            sig = json.dumps([tool_name, tool_args], ensure_ascii=False, sort_keys=True)
            # 只有“上一次同样的调用是成功的”才累计重复，否则交给失败计数器处理
            if sig == last_sig and last_ok:
                repeats += 1
            elif sig != last_sig:
                repeats = 0
            if repeats >= 2:
                stop_reason = "loop_detected"
                final_answer = (
                    f"检测到重复执行同一动作（{tool_name}），已中止以免死循环。"
                )
                trace.append(
                    {"step": step, "tool": tool_name, "args": tool_args,
                     "ok": False, "output": "重复动作，已中止"}
                )
                break
            last_sig = sig

            spec = self.registry.get(tool_name)
            self._notify(
                on_event,
                {
                    "event": "tool_start",
                    "turn": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "dangerous": bool(getattr(spec, "dangerous", False)),
                    "require_confirm": bool(getattr(spec, "require_confirm", False)),
                },
            )

            try:
                observation, elapsed_ms, tool_args, attempt, skipped = self._run_tool(
                    tool_name, tool_args, step, on_event
                )
            except RunCancelled:
                stop_reason = "cancelled"
                final_answer = "任务已被用户停止，已保留执行轨迹。"
                break

            ok = bool(observation.get("ok"))
            consecutive_failures = 0 if ok else consecutive_failures + 1
            last_ok = ok

            trace.append(
                {
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "ok": ok,
                    "elapsed_ms": elapsed_ms,
                    "output": (observation.get("output") or "")[:500],
                    "error": observation.get("error"),
                    "attempt": attempt,
                    "skipped": skipped,
                }
            )
            self._notify(
                on_event,
                {
                    "event": "tool_end",
                    "turn": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "ok": ok,
                    "elapsed_ms": elapsed_ms,
                    "error": observation.get("error"),
                    "denied": bool((observation.get("meta") or {}).get("denied")),
                    "cancelled": bool((observation.get("meta") or {}).get("cancelled")),
                    "skipped": skipped,
                    "attempt": attempt,
                    "output": self._truncate(observation.get("output") or "", 4000),
                    "output_len": len(observation.get("output") or ""),
                },
            )

            obs_text = observation.get("output") or ""
            if observation.get("error"):
                obs_text += f"\n[错误] {observation['error']}"
            obs_text = self._truncate(obs_text, self.config.max_observation_chars)

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[工具执行结果] tool={tool_name} ok={ok}\n{obs_text}\n"
                        f"请根据结果继续；已完成则直接给出最终答案。"
                    ),
                }
            )
            self._compact(messages)

            if consecutive_failures >= 3:
                stop_reason = "too_many_failures"
                final_answer = (
                    f"连续 {consecutive_failures} 次工具调用失败，已中止。"
                    f"最后错误：{observation.get('error')}"
                )
                break
        else:
            final_answer = final_answer or "达到最大步数，任务未完成。"

        return {
            "goal": goal,
            "answer": final_answer,
            "model": model_name,
            "steps": len(trace),
            "stop_reason": stop_reason,
            "trace": trace,
            "messages": messages,
            "rag_sources": self._rag_sources(self.last_rag_context),
            "rag_chars": len(self.last_rag_context) + len(self.last_experience_context),
        }

    # 兼容旧接口
    def chat(self, user_input: str, **kwargs) -> Dict[str, Any]:
        return self.run_task(user_input, **kwargs)

    # ---------------- 单个工具的执行（含人工干预） ----------------
    def _run_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        turn: int,
        on_event: Optional[Any] = None,
    ):
        """执行一次工具调用；失败且用户开启「失败时暂停」时等三选一。

        返回 (observation, elapsed_ms, 实际使用的 args, 尝试次数, 是否被跳过)。
        """
        attempt = 1
        elapsed_ms = 0
        skipped = False

        t0 = time.time()
        observation = self.registry.invoke(tool_name, tool_args)
        elapsed_ms += int((time.time() - t0) * 1000)

        while not observation.get("ok") and self.control is not None and self.control.ask_on_failure:
            # 用户可能在这期间点了停止/暂停
            self.control.checkpoint()
            decision = self.control.ask_failure(
                turn=turn,
                tool=tool_name,
                args=tool_args,
                error=str(observation.get("error") or ""),
            )
            action = str(decision.get("action") or "")

            if action == "retry":
                attempt += 1
                t0 = time.time()
                observation = self.registry.invoke(tool_name, tool_args)
                elapsed_ms += int((time.time() - t0) * 1000)
                continue

            if action == "fix":
                new_args = decision.get("args") or {}
                if not isinstance(new_args, dict) or not new_args:
                    break
                attempt += 1
                tool_args = new_args
                t0 = time.time()
                observation = self.registry.invoke(tool_name, tool_args)
                elapsed_ms += int((time.time() - t0) * 1000)
                continue

            if action == "skip":
                original = str(observation.get("error") or "")
                observation = {
                    "ok": True,
                    "output": "（用户选择跳过该步骤，继续后面的步骤）",
                    "error": None,
                    "meta": {"skipped": True, "original_error": original},
                }
                skipped = True
                break

            # abort / continue / 超时：按原失败结果继续，交给模型换方案
            break

        return observation, elapsed_ms, tool_args, attempt, skipped

    @staticmethod
    def _notify(cb: Optional[Any], event: Dict[str, Any]) -> None:
        if cb is None:
            return
        try:
            cb(event)
        except Exception:
            pass

    # ---------------- 上下文管理 ----------------
    @staticmethod
    def _rag_sources(knowledge: str) -> List[str]:
        """从注入块里抽出来源文件名，供事件/复盘用（不含正文，避免泄露到日志）。"""
        return re.findall(r"来源：([^\n]+)", knowledge or "")[:8]

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = text[: int(limit * 0.6)]
        tail = text[-int(limit * 0.3):]
        return f"{head}\n...[中间省略 {len(text) - len(head) - len(tail)} 字符]...\n{tail}"

    @staticmethod
    def _compact(messages: List[Dict[str, str]], keep_recent: int = 24) -> None:
        """上下文过长时丢掉最早的观察记录（保留 system、目标与最近若干步）。"""
        if len(messages) <= keep_recent + 4:
            return
        head = messages[:2]
        tail = messages[-keep_recent:]
        dropped = len(messages) - 2 - keep_recent
        note = {
            "role": "user",
            "content": f"[上下文压缩] 已省略最早的 {dropped} 条记录，如需回顾请重新读取文件。",
        }
        messages[:] = head + [note] + tail
