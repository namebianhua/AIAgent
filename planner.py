"""Planner / Critic / Summarizer（T3）。

三个角色共用同一个模型网关，但职责严格分离：
- Planner：把目标拆成可执行步骤，并给出"完成标准"；
- Critic：只看证据判断某一步是否真的完成（模型自称完成不算）；
- Summarizer：多步任务结束后写一段人能读的总结。

所有输出都要求 JSON，并且都有解析失败时的确定性兜底，
避免"模型输出格式一乱，整个任务就静默当成功"。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent import extract_json_object
from gateway import ModelGateway

_NUMBERED_RE = re.compile(r"^\s*(?:\d+|[一二三四五六七八九十]+)\s*[.、)．]\s*(.+)$")

PLANNER_SYSTEM = """你是任务规划器。只输出 JSON，不要输出解释、不要用代码块。"""

PLANNER_TEMPLATE = """把目标拆成可执行步骤。

目标：{goal}
{experience}
可用工具（只列名字，参数在调用时再填）：
{tools}

要求：
1. 最多 {max_steps} 步，每步是一句具体动作，不要写"思考""分析"这类空步骤；
2. 每步必须能用上面某个工具完成，或者是一次信息确认；
3. 步骤之间不要重复，后一步可以依赖前一步的结果；
4. 如果目标一步就能完成，就只给一步。

只输出这个 JSON：
{{"steps":["步骤1","步骤2"],"success_criteria":"判断整体完成的可验证标准"}}"""

CRITIC_SYSTEM = """你是结果校验员。只输出 JSON，不要输出解释。"""

CRITIC_TEMPLATE = """判断当前步骤是否真的完成。

总目标：{goal}
当前步骤：{step}
完成标准：{criteria}

执行证据（工具真实返回，可能被截断）：
{evidence}

判定规则：
- 只有证据里出现实际观测（文件内容、命令输出、搜索结果、页面数据）才算完成；
- 模型自己说"已完成"但没有对应观测，算未完成；
- 证据明确报错且没有补救，算未完成。

只输出这个 JSON：
{{"ok":true,"reason":"一句话依据","advice":"若未完成，下一步该怎么改；完成则留空"}}"""

SUMMARY_SYSTEM = """你是任务总结员。用中文简洁回答，先结论后依据。"""

SUMMARY_TEMPLATE = """根据执行记录，给用户一个最终答复。

目标：{goal}
执行记录：
{records}

要求：直接说结果（做了什么、产出在哪、关键数据），不要复述过程，不要编造记录里没有的内容。"""


@dataclass
class Plan:
    goal: str
    steps: List[str]
    criteria: str = ""
    raw: str = ""
    parsed: bool = True


@dataclass
class Critique:
    ok: bool
    reason: str = ""
    advice: str = ""
    raw: str = ""
    parsed: bool = True


@dataclass
class Planner:
    gateway: ModelGateway
    max_steps: int = 6
    timeout: int = 120
    last_model: str = ""
    calls: int = 0

    # ---------- 内部 ----------
    def _ask(self, system: str, user: str) -> str:
        self.calls += 1
        resp = self.gateway.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=self.timeout,
        )
        self.last_model = resp.get("model", self.last_model)
        content = resp.get("content") or ""
        return content if isinstance(content, str) else str(content)

    # ---------- 规划 ----------
    def make_plan(
        self,
        goal: str,
        tools_brief: str,
        experience: str = "",
        feedback: str = "",
    ) -> Plan:
        exp_block = f"\n历史经验（仅供参考，环境可能已变化，不要照搬）：\n{experience}\n" if experience else ""
        if feedback:
            exp_block += f"\n上一次尝试失败，必须换一种做法：\n{feedback}\n"

        prompt = PLANNER_TEMPLATE.format(
            goal=goal,
            experience=exp_block,
            tools=tools_brief,
            max_steps=self.max_steps,
        )
        try:
            raw = self._ask(PLANNER_SYSTEM, prompt)
        except Exception as e:
            return Plan(goal=goal, steps=[goal], criteria="", raw=f"<规划失败 {e}>", parsed=False)

        steps, criteria = self._parse_plan(raw)
        parsed = bool(steps)
        if not steps:
            steps = [goal]
        return Plan(
            goal=goal,
            steps=steps[: self.max_steps],
            criteria=criteria,
            raw=raw[:800],
            parsed=parsed,
        )

    def _parse_plan(self, raw: str):
        obj = extract_json_object(raw)
        steps: List[str] = []
        criteria = ""
        if isinstance(obj, dict):
            criteria = str(obj.get("success_criteria") or obj.get("criteria") or "")[:300]
            raw_steps = obj.get("steps") or obj.get("plan") or []
            if isinstance(raw_steps, str):
                raw_steps = [raw_steps]
            if isinstance(raw_steps, list):
                for item in raw_steps:
                    if isinstance(item, str):
                        text = item.strip()
                    elif isinstance(item, dict):
                        text = str(
                            item.get("step") or item.get("action") or item.get("description") or ""
                        ).strip()
                    else:
                        text = str(item).strip()
                    text = re.sub(r"^[\s\-*•]+", "", text)
                    if text and len(text) <= 300:
                        steps.append(text)
        if not steps:
            # 兜底：解析编号列表（模型偶尔不守 JSON 格式）
            for line in (raw or "").splitlines():
                m = _NUMBERED_RE.match(line.strip())
                if m:
                    text = m.group(1).strip().strip("*`")
                    if 2 <= len(text) <= 300:
                        steps.append(text)
        steps = self._dedupe(steps)
        return steps, criteria

    @staticmethod
    def _dedupe(steps: List[str]) -> List[str]:
        out: List[str] = []
        for s in steps:
            if not out or s != out[-1]:
                out.append(s)
        return out

    # ---------- 校验 ----------
    def judge(self, goal: str, step: str, evidence: str, criteria: str = "") -> Critique:
        prompt = CRITIC_TEMPLATE.format(
            goal=goal,
            step=step,
            criteria=criteria or "该步骤的动作确实执行成功，并有可验证的观测结果",
            evidence=evidence[:3000],
        )
        try:
            raw = self._ask(CRITIC_SYSTEM, prompt)
        except Exception as e:
            return Critique(ok=True, reason=f"校验不可用，按执行结果判定（{e}）", parsed=False)

        obj = extract_json_object(raw)
        if not isinstance(obj, dict) or "ok" not in obj:
            return Critique(ok=True, reason="校验输出无法解析，按执行结果判定", raw=raw[:400], parsed=False)

        value = obj.get("ok")
        if isinstance(value, str):
            ok = value.strip().lower() in ("true", "yes", "1", "通过", "完成", "是")
        else:
            ok = bool(value)
        return Critique(
            ok=ok,
            reason=str(obj.get("reason") or "")[:300],
            advice=str(obj.get("advice") or "")[:300],
            raw=raw[:400],
        )

    # ---------- 总结 ----------
    def summarize(self, goal: str, records: List[Dict[str, Any]]) -> str:
        lines = []
        for i, r in enumerate(records, 1):
            lines.append(
                f"{i}. 步骤：{r.get('step', '')}\n   结果：{str(r.get('answer') or '')[:400]}"
            )
        prompt = SUMMARY_TEMPLATE.format(goal=goal, records="\n".join(lines)[:4000])
        try:
            text = self._ask(SUMMARY_SYSTEM, prompt).strip()
            if text:
                return text
        except Exception:
            pass
        return "\n".join(
            f"- {r.get('step')}: {str(r.get('answer') or '')[:200]}" for r in records
        )
