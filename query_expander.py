"""查询改写（第五阶段：跨词汇鸿沟）。

要解决的问题（实测数据，不是推测）：
    严格评测集里 6/20 条查询失败，全部是「用户口语」与「文档术语」字面不重叠：
      「最多能跑多快，加机器有用吗」 → 文档写的是「吞吐上限 12,000 条/秒」「硬件规格」
      「防火墙要放行哪些号」        → 文档写的是「端口清单」「8443/9092」
      「买哪几个行业比较好」        → 文档写的是「板块配置与标的精选」
    纯向量检索跨不过这道鸿沟（embedding 知道意思相近，但分数不足以压过其他块）。
    受控实验：把这 6 条查询人工扩展成含文档术语的形式后，top1 命中 6/6 全部修复。

做法：用本地小模型把用户查询改写成 2~3 条「可能出现在文档里的说法」，
多条查询各自检索后用 RRF 融合（融合逻辑在 rag_engine.search_hybrid）。

设计约束：
1. 纯增益、可关闭：改写失败/超时/模型不在，一律回落成「只用原查询」，绝不抛异常；
2. 低延迟：temperature=0、限制输出长度、进程内缓存同一条查询；
3. 不泄露：改写只发生在本地 Ollama，查询不出机器。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_PROMPT = """你是检索查询扩展器。针对用户问题，输出两部分内容，供两种不同的检索通道使用。

第一部分「改写」：把口语问题换成 2 条书面问法（用于语义向量检索）。
- 仍是完整问句，用文档里可能出现的书面词替换口语词；
- 保留原问题中的专有名词、编号、数字原样不动；
- 每条不超过 30 字。

第二部分「术语」：列出该问题在文档里最可能出现的术语词（用于关键词检索）。
- 只输出名词性术语/字段名/指标名/章节名，用空格分隔，**不要写成问句**；
- 把口语词换成书面词，例如：
  「能跑多快」→ 吞吐 性能 上限 硬件规格
  「放行哪些号」→ 端口 防火墙 网络配置
  「买哪几个行业」→ 板块 行业配置 标的精选
- 输出 2 行，每行 4~8 个词，每行不超过 40 字。

严格按下面格式输出，不要编号、不要解释、不要多余空行：
改写：
<问法1>
<问法2>
术语：
<术语行1>
<术语行2>

用户问题：{query}

"""


@dataclass
class ExpanderConfig:
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 25          # 改写是检索的前置步骤，必须快；超时就放弃
    max_variants: int = 3      # 向量通道：最多取几条书面问句改写
    max_terms: int = 2         # 词法通道：最多取几行术语词袋
    temperature: float = 0.0   # 改写要稳定可复现，不要创造性
    num_ctx: int = 1024
    enabled: bool = True


@dataclass
class Expansion:
    """一次查询扩展的结果，按检索通道分开存放。

    为什么必须分开（实测依据，不是设计偏好）：
    - paraphrases（书面问句）→ 向量通道。语义检索需要完整句式，
      把「吞吐 性能 上限 硬件规格」这类词袋编码成向量会得到语义质心，
      同时匹配多个块、精度被稀释（实测 top1 命中 3/6 → 1/6）。
    - terms（术语词袋）→ 词法通道。BM25 做的是词项匹配，要的就是关键词，
      完整句子反而会引入大量低 IDF 的虚词噪音。
    """
    query: str
    paraphrases: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    ok: bool = False          # 模型改写是否成功（False 时只用原查询）

    @property
    def vector_queries(self) -> List[str]:
        """向量通道用：原查询 + 书面改写。"""
        out = [self.query]
        for p in self.paraphrases:
            if p and p not in out:
                out.append(p)
        return out

    @property
    def lexical_queries(self) -> List[str]:
        """词法通道用：原查询 + 术语词袋。"""
        out = [self.query]
        for t in self.terms:
            if t and t not in out:
                out.append(t)
        return out


class QueryExpander:
    """把一条查询扩展成「书面问句 + 术语词袋」两组候选。

    任何失败都退化为只用原查询，绝不抛异常——改写是增益，不是主链路依赖。
    """

    def __init__(self, cfg: Optional[ExpanderConfig] = None, client: Any = None) -> None:
        self.cfg = cfg or ExpanderConfig()
        self._client = client           # 可注入 OllamaClient，便于复用连接与测试
        self._cache: Dict[str, Expansion] = {}
        self._lock = threading.RLock()
        self.last_latency_ms = 0
        self.last_error = ""
        self._disabled = not self.cfg.enabled

    # ---------- 输出解析 ----------
    @staticmethod
    def _clean_line(raw: str) -> str:
        line = (raw or "").strip()
        if not line:
            return ""
        # 去掉常见前缀：「1.」「-」「*」以及包裹的引号
        line = line.lstrip("0123456789.、)）-*·# \t").strip()
        line = line.strip("\"'“”‘’「」《》`")
        # 模型有时复述指令，这类行不是有效输出
        if any(k in line for k in ("要求", "以下是", "用户问题", "严格按")):
            return ""
        if len(line) > 120:
            return ""
        return line

    @staticmethod
    def _is_label(line: str) -> bool:
        """判断某行是否是「改写：」「术语：」这类分段标签（而非真实内容）。

        防御性过滤：正常情况下切块已排除标签，但模型偶尔会把标签写成
        「**术语**」「术语:」或在正文行里重复标签，这里兜底拦掉，
        避免标签文本被当成一条查询送进检索通道。
        """
        s = line.strip().strip("*").strip()
        if not s:
            return True
        # 去掉冒号后基本为空，说明就是个标签
        head = s.rstrip("：:").strip()
        return head in ("改写", "问句", "术语", "关键词", "关键术语")

    @classmethod
    def _parse(cls, text: str, original: str, max_paraphrase: int,
               max_terms: int) -> Expansion:
        """按「改写：/ 术语：」两段解析模型输出。

        分段失败时（模型没按格式走）退化为整体当问句用，保证不比原查询更差。
        """
        exp = Expansion(query=original)
        raw = text or ""

        # 定位两段。容忍模型写成「改写:」「**改写**」等变体。
        # 必须同时返回标签的**起始**与**结束**位置：
        #   起始 → 切块边界（上一段正文不能包含下一段的标签）
        #   结束 → 内容起点（跳过标签本身）
        # 早期版本只返回结束位置，却拿它当 para_block 的右边界，
        # 于是 raw[p_end:t_end] 把「术语：」这 3 个字切进了问句块，
        # 导致 '术语：' 被当成一条问句改写送进向量通道（实测污染 variants_used）。
        def _find(labels: List[str]) -> Tuple[int, int]:
            for lb in labels:
                i = raw.find(lb)
                if i >= 0:
                    return i, i + len(lb)
            return -1, -1

        p_begin, p_end = _find(["改写：", "改写:", "问句：", "问句:"])
        t_begin, t_end = _find(["术语：", "术语:", "关键词：", "关键词:"])

        para_block = ""
        term_block = ""
        if p_begin >= 0 and t_begin > p_begin:
            para_block = raw[p_end:t_begin]   # 右边界用「术语：」的起始位置
            term_block = raw[t_end:]
        elif p_begin >= 0:
            para_block = raw[p_end:]
        elif t_begin >= 0:
            term_block = raw[t_end:]
        else:
            para_block = raw  # 没有分段标记，整体当问句

        orig_norm = _norm(original)
        for ln in para_block.splitlines():
            c = cls._clean_line(ln)
            if not c or cls._is_label(c) or _norm(c) == orig_norm or c in exp.paraphrases:
                continue
            exp.paraphrases.append(c)
            if len(exp.paraphrases) >= max_paraphrase:
                break
        for ln in term_block.splitlines():
            c = cls._clean_line(ln)
            if not c or cls._is_label(c) or c in exp.terms:
                continue
            exp.terms.append(c)
            if len(exp.terms) >= max_terms:
                break

        exp.ok = bool(exp.paraphrases or exp.terms)
        return exp

    # ---------- 核心 ----------
    def expand_structured(self, query: str) -> Expansion:
        """返回分组后的扩展结果；不可用/失败时 ok=False 且只含原查询。"""
        query = (query or "").strip()
        if not query:
            return Expansion(query="")
        if self._disabled:
            return Expansion(query=query, ok=False)

        with self._lock:
            cached = self._cache.get(query)
        if cached is not None:
            return cached

        try:
            exp = self._call_model(query)
        except Exception as e:
            # 改写是可选增强：失败就只用原查询，不能让它拖垮检索
            self.last_error = f"{type(e).__name__}: {e}"
            exp = Expansion(query=query, ok=False)

        with self._lock:
            if len(self._cache) >= 256:
                self._cache.clear()
            self._cache[query] = exp
        return exp

    def expand(self, query: str) -> List[str]:
        """向后兼容接口：返回向量通道用的 [原查询, 改写...]。"""
        return self.expand_structured(query).vector_queries

    def _call_model(self, query: str) -> Expansion:
        cfg = self.cfg
        messages = [{"role": "user", "content": _PROMPT.format(query=query)}]
        t0 = time.perf_counter()

        if self._client is not None:
            resp = self._client.chat(
                messages,
                ollama_model=cfg.model,
                temperature=cfg.temperature,
                num_ctx=cfg.num_ctx,
                timeout=cfg.timeout,
            )
            content = str(resp.get("content") or "")
        else:
            import requests

            r = requests.post(
                f"{cfg.base_url.rstrip('/')}/api/chat",
                json={
                    "model": cfg.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
                },
                timeout=cfg.timeout,
            )
            r.raise_for_status()
            content = str((r.json().get("message") or {}).get("content") or "")

        self.last_latency_ms = round((time.perf_counter() - t0) * 1000)
        return self._parse(content, query, cfg.max_variants, cfg.max_terms)

    # ---------- 观测 ----------
    def probe(self) -> Dict[str, Any]:
        """检查改写通道是否可用（模型是否已拉取）。"""
        if self._disabled:
            return {"ok": False, "reason": "查询改写已关闭"}
        try:
            import requests

            r = requests.get(f"{self.cfg.base_url.rstrip('/')}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:
            self._disabled = True
            return {"ok": False, "reason": f"Ollama 不可达，已禁用改写：{type(e).__name__}"}
        hit = any(m == self.cfg.model or m.startswith(self.cfg.model + ":") for m in models)
        if not hit:
            self._disabled = True
            return {"ok": False, "reason": f"未找到改写模型 {self.cfg.model}，已禁用改写",
                    "available": models}
        return {"ok": True, "model": self.cfg.model}


def _norm(s: str) -> str:
    """归一化用于去重比较：去空白、转小写。"""
    return "".join((s or "").split()).lower()


_EXPANDER: Optional[QueryExpander] = None
_EXPANDER_LOCK = threading.Lock()


def get_expander(**kwargs: Any) -> QueryExpander:
    """进程内单例（缓存与连接复用）。"""
    global _EXPANDER
    with _EXPANDER_LOCK:
        if _EXPANDER is None:
            _EXPANDER = QueryExpander(ExpanderConfig(**kwargs))
        return _EXPANDER
