"""本地 RAG 引擎（第一阶段）：解析 → 分块 → Ollama 向量化 → ChromaDB 存储 → 相似度检索。

设计取舍：
1. 全程本地：向量化走 Ollama 的 embedding 模型，存储走 ChromaDB 持久化目录，
   文档内容不出机器；
2. chromadb 是**可选依赖**：没装时抛 RagUnavailable，由上层降级成「无知识库」运行，
   而不是让整个 Agent 起不来（项目其余部分刻意零重依赖）；
3. 幂等入库：同一文件内容没变就跳过，变了就先删旧块再写新块，
   避免反复入库把知识库灌成重复垃圾；
4. 维度守卫：每个 collection 记住首次入库用的 embedding 模型，
   换模型时报错而不是混入不同维度的向量（ChromaDB 会静默算出无意义的距离）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

# 支持的文本类文件（二进制/图片不入库）
TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".html", ".htm", ".css", ".csv", ".log",
    ".sql", ".sh", ".bat", ".xml", ".java", ".go", ".rs", ".c", ".h", ".cpp",
}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
SUPPORTED_EXT = TEXT_EXT | PDF_EXT | DOCX_EXT

# collection 的向量空间：cosine 下 distance = 1 - 余弦相似度，可直接换算相关度
_COSINE = "cosine"
_META_FILE = "embed_meta.json"

# 默认 embedding 模型（全项目唯一来源）。
# 选 bge-m3 而不是 nomic-embed-text 的依据见 RagConfig.embed_model 注释；
# 其他模块请统一 `from rag_engine import DEFAULT_EMBED_MODEL`，不要再写死字符串。
DEFAULT_EMBED_MODEL = "bge-m3"

# 默认相关度门槛（bge-m3 中文场景经验起点，请用 calibrate_scores 按语料标定）
DEFAULT_MIN_SCORE = 0.35

# 分块算法版本。参与内容指纹计算（见 RagEngine._content_hash）。
# **每次修改分块逻辑都必须递增**，否则重新入库会被幂等检查判为 unchanged 而静默跳过，
# 库里仍是旧切法的向量，改动看起来"没生效"且极难排查。
# v1 纯字符数聚合分块
# v2 结构感知：按标题切节、表格不跨块、长表格重复表头、块带【标题路径】前缀
# v3 标题路径始终保留根标题（修排序倒挂）；同小节片段贪心合并（消除碎片块）；过滤 --- 分隔线
CHUNKER_VERSION = 3


class RagUnavailable(RuntimeError):
    """RAG 依赖缺失或本地服务不可用（上层应降级，而不是崩）。"""


@dataclass
class RagConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    # bge-m3：多语言向量模型，中文语义区分度显著优于 nomic-embed-text（英文主导）。
    # 实测对照：中文「强相关 vs 不相关」nomic 只有 0.705/0.725（倒挂），bge-m3 能正常拉开。
    # 注意：不同模型的余弦分数分布不同，换模型时 min_score 需要重新标定（见 calibrate_scores）。
    embed_model: str = DEFAULT_EMBED_MODEL
    db_path: str = "./agent_knowledge_base"
    chunk_size: int = 600        # 每块最大字符数（bge-m3 上下文 8192 token，余量充足）
    chunk_overlap: int = 80      # 相邻块重叠字符数，避免句子被切断丢语义（约 13%，行业常规）
    embed_timeout: int = 120     # bge-m3 比 nomic 重，首token与长文本编码更慢，放宽超时
    embed_workers: int = 4       # 本地 embedding 并发数；CPU 推理不宜过高
    max_file_bytes: int = 20 * 1024 * 1024
    # 相关度门槛：低于此值的检索结果视为「未命中」，不注入、不返回。
    # 0.35 是 bge-m3 中文场景的经验起点，请用 calibrate_scores() 按自己的语料标定。
    min_score: float = DEFAULT_MIN_SCORE
    # 混合检索的向量高置信短路：当原查询的向量通道 top1 已过门槛、
    # 且与 top2 的分差 ≥ shortcut_gap 时，直接采信向量结果，不做 RRF 融合。
    # 为什么需要：RRF 只看名次、丢弃分数量纲，会把「向量已明确做对」的结果
    # 被「多个改写变体都排第 2」的块翻盘（实测退化 3 条）。详见 search_hybrid。
    hybrid_shortcut: bool = True
    # 短路所需的最小分差。取 0.02 的依据（_diag_gap.py 实测）：
    #   向量做对组 gap 中位 0.040（退化组 0.027~0.040 需被短路保住）；
    #   向量做错组 gap 中位 0.017（修复组多为 0.001~0.030，需照常进融合）。
    #   0.02 落在两组之间，偏向让「不确定」的情况走融合。
    shortcut_gap: float = 0.02
    # 混合检索每个通道的候选池大小。**必须与 top_k 解耦**（固定常量）：
    # 若随 top_k 变化，候选池大小就会变，RRF 名次随之漂移，
    # 导致同一个问题在 --top-k=1 / --top-k=3 下返回不同答案（实测发生过）。
    # 取 40 的理由：要显著大于常见 top_k（1~10）以覆盖候选，又不能大到
    # 拖慢检索（每条查询变体都要对候选池做一次打分）。
    hybrid_fetch: int = 40



def _norm_source_id(path: str | os.PathLike[str]) -> str:
    """把文件路径归一化成稳定的来源键，用于 ChromaDB 的精确匹配删除。

    为什么必须归一化：ChromaDB 的 where={"source_id": ...} 是**精确字符串匹配**，
    而 Windows 路径大小写不敏感。同一个文件可能先后以
    `c:\\Users\\Lenovo\\Downloads\\x.md`（小写盘符）和
    `C:\\Users\\Lenovo\\Downloads\\x.md`（大写盘符）入库，
    删除旧块时匹配不到 → 旧向量残留 → 每次重建都累积过期块，检索质量持续劣化。

    归一化规则（仅影响 source_id 这个匹配键，不影响展示用的 source_file）：
    1. resolve() 拿到绝对规范路径（消除 .. 、相对路径、正斜杠混用）；
    2. 分隔符统一为 \\；
    3. 在 Windows 上整体转小写（NTFS 大小写不敏感，Linux/macOS 保留原样）。
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    s = str(p).replace("/", "\\") if os.name == "nt" else str(p)
    return s.lower() if os.name == "nt" else s


def _strip_title_prefix(text: str) -> str:
    """去掉块首的【标题路径】前缀，只留正文（构造伪查询用）。"""
    head = (text or "").lstrip()
    if head.startswith("【"):
        end = head.find("】")
        if end > 0:
            return head[end + 1:].strip()
    return head.strip()


def _make_pseudo_query(body: str, limit: int = 48) -> str:
    """从块正文造一个「伪查询」：取首个句子片段，模拟真实用户提问的长度与信息量。

    刻意不用整块正文当查询——那样与块本身几乎重合，余弦会逼近 1.0，
    严重高估相关命中分数，标出的门槛同样偏高。
    """
    body = (body or "").strip()
    if not body:
        return ""
    # 表格块：取表头 + 首行数据，等价于用户问"某项的取值"
    if body.lstrip().startswith("|"):
        rows = [r.strip() for r in body.split("\n") if r.strip().startswith("|")]
        cells = [c.strip(" `") for r in rows[:3] for c in r.strip("|").split("|")]
        cells = [c for c in cells if c and not set(c) <= set("-: ")]
        return " ".join(cells[:6])[:limit]
    # 普通文本：取第一句
    m = re.split(r"(?<=[。！？；.!?])", body, maxsplit=1)
    first = (m[0] if m else body).strip()
    first = re.sub(r"^[>*#\-\s]+", "", first)          # 去掉 Markdown 引用/列表符号
    first = first.replace("**", "").replace("`", "")    # 去掉强调与代码标记
    return first[:limit].strip()


def _tokenize(text: str) -> List[str]:
    """中英文混合分词，零依赖。

    - 英文/数字/标识符：按 ASCII 连续串切（保留 E4201、8443、batch_size 这类整体）；
    - 中文：按字切（bigram 效果更好但内存翻倍，单字 + IDF 已足够区分）。
    刻意不引 jieba：项目其余部分保持零重依赖，且中文单字 + BM25 的 IDF 权重
    对「端口/吞吐/归因」这类术语词的区分已经够用。
    """
    if not text:
        return []
    tokens: List[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text):
        tokens.append(m.group(0).lower())
    return tokens


def _identifier_tokens(query: str) -> List[str]:
    """从查询里抽出「精确标识符」：错误码、端口号、参数名、版本号等。

    这些是纯向量检索最容易漏的（embedding 会把 E4201 和 E4202 编码得很接近），
    必须靠词法通道精确匹配。判定标准：含数字的字母数字串，或带下划线的标识符。
    """
    if not query:
        return []
    out: List[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+", query):
        tok = m.group(0)
        if "_" in tok or re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok):
            out.append(tok)
    return out


class _BM25Index:
    """极简 BM25（Okapi），零依赖。只为混合检索提供词法通道名次。

    实现的是标准 BM25 公式：
        score(D,Q) = Σ IDF(qi) · (f(qi,D)·(k1+1)) / (f(qi,D) + k1·(1-b+b·|D|/avgdl))
        IDF(qi) = ln( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
    k1=1.5、b=0.75 是 Okapi 的经典默认值。
    """

    def __init__(self, documents: Sequence[str], ids: Sequence[str],
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids = list(ids)
        self.docs = list(documents)
        self.n = len(self.docs)
        self.doc_tokens: List[List[str]] = [_tokenize(d) for d in self.docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        # 词频表：doc_index -> {token: count}
        self.tf: List[Dict[str, int]] = []
        for toks in self.doc_tokens:
            counter: Dict[str, int] = {}
            for t in toks:
                counter[t] = counter.get(t, 0) + 1
            self.tf.append(counter)
        # 文档频率：token -> 含该词的文档数
        self.df: Dict[str, int] = {}
        for counter in self.tf:
            for t in counter:
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, token: str) -> float:
        n_t = self.df.get(token, 0)
        if n_t == 0:
            return 0.0
        import math
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def search(self, query: str, top_n: int) -> List[tuple]:
        """返回 [(doc_id, score, content), ...]，按分数降序，score<=0 不返回。"""
        if self.n == 0:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        # 查询去重，避免重复词放大分数
        q_set = list(dict.fromkeys(q_tokens))
        scores: List[float] = [0.0] * self.n
        for token in q_set:
            idf = self._idf(token)
            if idf <= 0:
                continue  # 查询词不在任何文档里，跳过
            for i in range(self.n):
                f = self.tf[i].get(token, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = [(i, s) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        out: List[tuple] = []
        for i, s in ranked[:top_n]:
            out.append((self.ids[i], s, self.docs[i]))
        return out


def _clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB 的 metadata 只接受 str/int/float/bool，且不能有 None。"""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class RagEngine:
    """本地知识库：文档入库与相似度检索。线程安全（锁保护 embedding 缓存与元数据文件）。"""

    def __init__(self, config: Optional[RagConfig] = None):
        self.cfg = config or RagConfig()
        self._lock = threading.RLock()
        self._embed_cache: Dict[str, List[float]] = {}
        self._cache_limit = 512
        self._collections: Dict[str, Any] = {}
        self._space: Dict[str, str] = {}

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:  # 可选依赖：给出可执行的修复提示
            raise RagUnavailable(
                "未安装 chromadb，知识库功能不可用。修复：pip install chromadb"
            ) from e

        db_path = Path(self.cfg.db_path).expanduser()
        db_path.mkdir(parents=True, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(
                path=str(db_path), settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            raise RagUnavailable(f"ChromaDB 初始化失败（{db_path}）：{e}") from e

        self.db_path = db_path
        self.meta_path = db_path / _META_FILE
        self.default_collection = "agent_general_knowledge"
        # 构造时给定的门槛，作为「该库没有标定值」时的回落基线。
        # 不能用 DEFAULT_MIN_SCORE 兜底，否则会覆盖调用方显式传入的配置。
        self._base_min_score = float(self.cfg.min_score)
        # 检索链路的异常记录：混合检索内部有多处 except 会静默返回空结果，
        # 没有这个字段就完全无法排查「为什么检索不到东西」。
        self.last_error: str = ""

    # ===================== 向量化 =====================

    def _embed_one(self, text: str) -> List[float]:
        """单条向量化：优先新版 /api/embed（批量接口），回落到 /api/embeddings。"""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._embed_cache:
                return self._embed_cache[key]

        url = self.cfg.ollama_url.rstrip("/")
        last_err = ""
        # 新版 Ollama：POST /api/embed {"model":..,"input":[..]} -> {"embeddings":[[..]]}
        try:
            r = requests.post(
                f"{url}/api/embed",
                json={"model": self.cfg.embed_model, "input": [text]},
                timeout=self.cfg.embed_timeout,
            )
            r.raise_for_status()
            embs = r.json().get("embeddings") or []
            if embs and isinstance(embs[0], list):
                return self._cache_put(key, embs[0])
            last_err = "响应缺少 embeddings"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        # 旧版 Ollama：POST /api/embeddings {"model":..,"prompt":".."} -> {"embedding":[..]}
        try:
            r = requests.post(
                f"{url}/api/embeddings",
                json={"model": self.cfg.embed_model, "prompt": text},
                timeout=self.cfg.embed_timeout,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if isinstance(emb, list) and emb:
                return self._cache_put(key, emb)
            embs = data.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                return self._cache_put(key, embs[0])
            last_err = f"响应格式异常：{str(data)[:120]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        raise RagUnavailable(
            f"Ollama 向量化失败（模型 {self.cfg.embed_model}）：{last_err}。"
            f"请确认 Ollama 正在运行且已拉取该模型：ollama pull {self.cfg.embed_model}"
        )

    def _cache_put(self, key: str, vec: List[float]) -> List[float]:
        with self._lock:
            if len(self._embed_cache) >= self._cache_limit:
                self._embed_cache.clear()  # 简单全清，避免引入 LRU 依赖
            self._embed_cache[key] = vec
        return vec

    def _embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        """批量向量化：并发调用单条接口（Ollama 的 /api/embeddings 不支持真正的批处理）。"""
        if not texts:
            return []
        if len(texts) == 1:
            return [self._embed_one(texts[0])]
        workers = max(1, min(self.cfg.embed_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._embed_one, texts))

    def probe(self) -> Dict[str, Any]:
        """探活：Ollama 在不在、embedding 模型拉没拉、维度多少。"""
        url = self.cfg.ollama_url.rstrip("/")
        try:
            r = requests.get(f"{url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:
            return {"ok": False, "reason": f"Ollama 不可达：{type(e).__name__}: {e}",
                    "ollama_url": url, "embed_model": self.cfg.embed_model}
        hit = any(
            m == self.cfg.embed_model or m.startswith(self.cfg.embed_model + ":")
            for m in models
        )
        if not hit:
            return {"ok": False, "reason": f"未找到 embedding 模型 {self.cfg.embed_model}，"
                                          f"请先执行：ollama pull {self.cfg.embed_model}",
                    "ollama_url": url, "embed_model": self.cfg.embed_model,
                    "available": models}
        try:
            dim = len(self._embed_one("dim-probe"))
        except Exception as e:
            return {"ok": False, "reason": f"embedding 调用失败：{e}",
                    "ollama_url": url, "embed_model": self.cfg.embed_model}
        return {"ok": True, "ollama_url": url, "embed_model": self.cfg.embed_model,
                "dim": dim, "db_path": str(self.db_path)}

    # ===================== 分块 =====================

    # Markdown 标题行：# ~ ######
    # 必须带 MULTILINE：既要能 match 单行（切节用），也要能 search 全文（判断是否 Markdown）
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    # Markdown 表格行：以 | 开头
    _TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
    # 表格分隔行：|---|---|
    _TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
    # Markdown 水平分隔线：--- / *** / ___（3 个及以上，独占一行）
    # 必须过滤：这类行无语义，混进块里只会稀释向量、拉低命中率
    _HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

    def _split_long(self, para: str) -> List[str]:
        """超长段落：先按句子切，单句仍超长则按固定长度硬切。"""
        parts = re.split(r"(?<=[。！？；.!?;\n])", para)
        out: List[str] = []
        buf = ""
        for sent in parts:
            if not sent:
                continue
            while len(sent) > self.cfg.chunk_size:  # 单句超长，硬切
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(sent[: self.cfg.chunk_size])
                sent = sent[self.cfg.chunk_size:]
            if len(buf) + len(sent) > self.cfg.chunk_size:
                if buf:
                    out.append(buf)
                buf = sent
            else:
                buf += sent
        if buf:
            out.append(buf)
        return [p.strip() for p in out if p.strip()]

    def _split_table(self, rows: List[str], budget: int) -> List[str]:
        """长表格按行切分，每片都重复表头，保证单独一片也可读。

        为什么必须重复表头：表格行 `| 13 | 技术背离 | 2.5 |` 脱离表头后
        完全无法理解，向量化出来就是噪音。带上表头才有语义。
        """
        header: List[str] = []
        body: List[str] = []
        for i, row in enumerate(rows):
            if i <= 1 or self._TABLE_SEP_RE.match(row):
                header.append(row)
            else:
                body.append(row)
        if not header:  # 不像标准表格，退化为整块返回
            return ["\n".join(rows)]

        head_text = "\n".join(header)
        # 留给数据行的空间
        room = max(80, budget - len(head_text) - 1)
        out: List[str] = []
        buf: List[str] = []
        size = 0
        for row in body:
            if size + len(row) + 1 > room and buf:
                out.append(head_text + "\n" + "\n".join(buf))
                buf, size = [], 0
            buf.append(row)
            size += len(row) + 1
        if buf:
            out.append(head_text + "\n" + "\n".join(buf))
        return out or [head_text]

    def _split_sections(self, lines: List[str]) -> List[Tuple[str, List[str]]]:
        """按 Markdown 标题切节，返回 [(标题路径, 该节正文行), ...]。

        标题路径带上层级（如「4. 十五维状态向量评分表」），
        后续会作为前缀拼进每个块——标题是文档里最强的语义信号，
        丢了它，块就变成一堆无上下文的表格碎片。
        """
        sections: List[Tuple[str, List[str]]] = []
        heading_stack: List[str] = []   # 各级标题，用于构造路径
        current_body: List[str] = []
        current_title = ""

        for line in lines:
            m = self._HEADING_RE.match(line)
            if m:
                # 先把上一节收掉
                if current_title or current_body:
                    sections.append((current_title, current_body))
                level = len(m.group(1))
                title = m.group(2).strip()
                # 维护层级：同级替换，降级截断
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                # 路径构造：**根标题必须始终保留**，最多再接最近两级小节。
                # 根标题承载文档主题/产品名/系统名，是检索最强的区分信号。
                # 早期版本写 heading_stack[-2:]（只取最近两级），深层小节因此丢掉
                # 根标题，导致查询带产品名时「浅层章节压过真正含答案的深层章节」。
                # 受控实测（块内容不变，仅补回根标题）：
                #   「部署 HaloSync 的硬件规格」 硬件规格 0.584 → 0.763，
                #   产品概述（对照，前缀本就完整）0.713 不变 → 排序倒挂被修正。
                if len(heading_stack) <= 3:
                    path_parts = heading_stack
                else:
                    path_parts = [heading_stack[0]] + heading_stack[-2:]
                current_title = " > ".join(path_parts)
                current_body = []
            else:
                current_body.append(line)
        if current_title or current_body:
            sections.append((current_title, current_body))
        return sections

    def _chunk_markdown(self, text: str) -> List[str]:
        """结构感知分块：按标题切节 → 表格整体保留 → 节内再按长度切。

        解决的核心问题：纯字符数聚合会把「上一节表格的尾巴 + 下一节的标题」
        揉进同一块，导致含答案的块语义被稀释、检索分数被相邻内容拉低。

        同小节内的相邻片段会贪心合并装箱：表格与紧跟其后的补充说明
        （如「| CPU | 4核 | 8核 |」+「> 吞吐上限由软件硬限决定」）
        本属同一语义单元，拆成两块会让短块语义偏弱、还白占检索名额。
        """
        lines = text.split("\n")
        sections = self._split_sections(lines)
        if not sections:
            return self._chunk_plain(text)

        chunks: List[str] = []
        for title, body in sections:
            prefix = f"【{title}】\n" if title else ""
            # 前缀占用的预算要从正文里扣掉，否则块会超出 chunk_size
            budget = max(120, self.cfg.chunk_size - len(prefix))

            pieces = self._section_pieces(body, budget)
            if not pieces:
                continue

            # 贪心装箱：在不超预算的前提下把同小节的片段合并成一个块
            buf = ""
            for piece in pieces:
                if buf and len(buf) + 2 + len(piece) > budget:
                    chunks.append(prefix + buf if prefix else buf)
                    buf = piece
                else:
                    buf = f"{buf}\n\n{piece}" if buf else piece
            if buf:
                chunks.append(prefix + buf if prefix else buf)

        # 去掉过短的碎块（<40 字符基本没有语义价值，只会当噪音被检索出来）
        return [c for c in chunks if len(c.strip()) >= 40]

    def _section_pieces(self, body: List[str], budget: int) -> List[str]:
        """把一个小节的正文切成片段：表格整体保留（超长则按行切并重复表头）。

        表格与普通文本先分段，是因为表格行脱离表头就没有意义；
        但分段之后仍会交给上层的装箱逻辑合并，不会各自成块。
        """
        segments: List[Tuple[bool, List[str]]] = []  # (是否表格, 行列表)
        cur_is_table = False
        cur: List[str] = []
        for line in body:
            # 过滤 Markdown 分隔线（--- / *** / ___）：无语义，混进块里只会稀释向量
            if self._HR_RE.match(line):
                continue
            is_table = bool(self._TABLE_ROW_RE.match(line))
            if cur and is_table != cur_is_table:
                segments.append((cur_is_table, cur))
                cur = []
            cur_is_table = is_table
            cur.append(line)
        if cur:
            segments.append((cur_is_table, cur))

        pieces: List[str] = []
        for is_table, seg_lines in segments:
            seg_text = "\n".join(l for l in seg_lines if l.strip()).strip()
            if not seg_text:
                continue
            if is_table:
                pieces.extend(p.strip() for p in self._split_table(seg_lines, budget) if p.strip())
            elif len(seg_text) <= budget:
                pieces.append(seg_text)
            else:
                pieces.extend(p for p in self._split_long(seg_text) if p)
        return pieces

    def _chunk_plain(self, text: str) -> List[str]:
        """无结构文本（非 Markdown）：按段落聚合到 chunk_size。"""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        raw: List[str] = []
        buf = ""
        for para in paragraphs:
            if len(para) > self.cfg.chunk_size:
                if buf:
                    raw.append(buf.strip())
                    buf = ""
                raw.extend(self._split_long(para))
                continue
            if len(buf) + len(para) + 1 <= self.cfg.chunk_size:
                buf += para + "\n"
            else:
                if buf:
                    raw.append(buf.strip())
                buf = para + "\n"
        if buf.strip():
            raw.append(buf.strip())
        return [c for c in raw if c]

    def _chunk_text(self, text: str) -> List[str]:
        """分块入口：Markdown 走结构感知分块，其余按段落聚合。

        overlap 只对无结构文本生效。结构化分块**不做 overlap**：
        每块已带【标题】前缀，若再把上一块尾部拼到开头，块就以
        「上一节的表格残渣」开头，标题这个最强语义信号反而被污染。
        而结构分块本身已按标题分节、表格不跨块切，不需要靠重叠来救断句。
        """
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return []

        has_heading = bool(self._HEADING_RE.search(text)) or "\n#" in text
        if has_heading:
            chunks = self._chunk_markdown(text)
            if chunks:
                return chunks
            # 标题识别失败时兜底走无结构分块（下面会继续做 overlap）

        chunks = self._chunk_plain(text)
        overlap = max(0, min(self.cfg.chunk_overlap, self.cfg.chunk_size // 2))
        if overlap == 0 or len(chunks) <= 1:
            return chunks

        out: List[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                head = chunks[i - 1][-overlap:]
                out.append(f"{head}\n{chunk}")
            else:
                out.append(chunk)
        return out

    # ===================== 文档解析 =====================

    def parse_file(self, filepath: str | os.PathLike[str]) -> str:
        """解析常见格式为纯文本。不支持的格式抛 ValueError。"""
        path = Path(filepath).expanduser()
        if not path.is_file():
            raise IOError(f"文件不存在：{path}")
        ext = path.suffix.lower()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > self.cfg.max_file_bytes:
            raise ValueError(f"文件过大（{size} B > {self.cfg.max_file_bytes} B）：{path}")

        try:
            if ext in TEXT_EXT:
                return path.read_text(encoding="utf-8", errors="replace")
            if ext in PDF_EXT:
                try:
                    import pypdf
                except ImportError as e:
                    raise ValueError("解析 PDF 需要 pypdf：pip install pypdf") from e
                reader = pypdf.PdfReader(str(path))
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n\n".join(p for p in pages if p.strip())
            if ext in DOCX_EXT:
                try:
                    import docx
                except ImportError as e:
                    raise ValueError("解析 docx 需要 python-docx：pip install python-docx") from e
                d = docx.Document(str(path))
                parts = [p.text for p in d.paragraphs if p.text.strip()]
                for table in d.tables:  # 表格里的信息常被漏掉，一并抽出
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            parts.append(" | ".join(cells))
                return "\n".join(parts)
        except (ValueError, IOError):
            raise
        except Exception as e:
            raise IOError(f"解析文件失败 {path}：{type(e).__name__}: {e}") from e
        raise ValueError(f"不支持的文件格式：{ext or '(无扩展名)'}")

    def _content_hash(self, text: str, filepath: str) -> str:
        """内容指纹 = 正文 + 分块算法版本 + 分块参数 + embedding 模型。

        为什么不能只哈希正文：幂等入库靠这个指纹判断「是否已存在完全相同的版本」。
        若只算正文，那么改了 chunk_size / 分块策略 / 换了 embedding 模型之后重新入库，
        会被误判为 unchanged 而跳过——旧向量（错误的切分方式）继续留在库里，
        改动的分块逻辑等于没生效。把参数一起哈希，配置一变就强制重建。

        为什么还要带 CHUNKER_VERSION：只改分块**算法**、不改任何**参数**时，
        指纹不会变化，重建会静默失效。这是最难排查的一类问题——
        代码改了、测试过了，但库里躺着的还是旧切法的向量。
        所以每次调整分块逻辑都必须递增该版本号。
        """
        h = hashlib.sha256()
        h.update(text.encode("utf-8", errors="replace"))
        h.update(
            f"|chunker={CHUNKER_VERSION}"
            f"|chunk={self.cfg.chunk_size},{self.cfg.chunk_overlap}"
            f"|embed={self.cfg.embed_model}".encode("utf-8")
        )
        return h.hexdigest()[:32]


    # ===================== collection 管理 =====================

    def _load_meta(self) -> Dict[str, Any]:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_meta(self, meta: Dict[str, Any]) -> None:
        try:
            with self._lock:
                self.meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception:
            pass  # 元数据写失败不影响入库主流程

    def _collection(self, name: str):
        """取（或建）collection，并记住它的向量空间与 embedding 模型。"""
        with self._lock:
            if name in self._collections:
                return self._collections[name]
            coll = None
            try:
                coll = self.client.get_collection(name=name)
            except Exception:
                try:
                    coll = self.client.create_collection(
                        name=name, metadata={"hnsw:space": _COSINE}
                    )
                except Exception:
                    # 并发建库或旧版 API 差异：退回 get_or_create（不带 metadata）
                    coll = self.client.get_or_create_collection(name=name)
            space = _COSINE
            try:
                space = str((coll.metadata or {}).get("hnsw:space") or _COSINE)
            except Exception:
                pass
            self._space[name] = space
            self._collections[name] = coll
            return coll

    def _check_embed_model(self, name: str) -> None:
        """维度守卫：同一 collection 混入不同 embedding 模型的向量会让检索彻底失真。"""
        meta = self._load_meta()
        entry = meta.get(name) or {}
        recorded = entry.get("embed_model")
        if recorded and recorded != self.cfg.embed_model:
            raise RagUnavailable(
                f"知识库 {name} 已用 {recorded} 建立向量，当前配置是 {self.cfg.embed_model}。"
                f"混用会导致检索失真。请二选一：改回原模型，或用 --rag-collection 指定一个新库名重建。"
            )
        if not recorded:
            meta[name] = {**entry, "embed_model": self.cfg.embed_model}
            self._save_meta(meta)

    def _min_score_for(self, name: str) -> float:
        """取该 collection 应使用的相关度门槛。

        余弦分数分布随 embedding 模型而变，所以门槛必须跟库走，不能全局共用一个值
        （否则换模型后阈值失效、噪音全部过线——这正是 nomic 时期 min_score=0.30
        形同虚设的原因）。该库没有标定值时回落到构造时的基线门槛。

        刻意做成「只读取、不改状态」：检索是多线程会走的路径，
        若在这里改写 cfg.min_score，A 库的门槛会串到并发检索的 B 库。
        """
        try:
            entry = (self._load_meta().get(name) or {})
        except Exception:
            return self._base_min_score
        saved = entry.get("min_score")
        if isinstance(saved, (int, float)) and not isinstance(saved, bool) and saved > 0:
            return float(saved)
        return self._base_min_score

    def _relevance(self, name: str, distance: float) -> float:
        """把 ChromaDB 的距离换算成 0-1 相关度。

        cosine 空间：distance = 1 - 余弦相似度，直接 1 - d。
        l2 空间（旧库）：对归一化向量 d² = 2(1 - cos)，故 cos = 1 - d²/2。
        """
        try:
            d = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if self._space.get(name, _COSINE) == _COSINE:
            score = 1.0 - d
        else:
            score = 1.0 - (d * d) / 2.0
        return max(0.0, min(1.0, score))

    def list_collections(self) -> List[str]:
        try:
            raw = self.client.list_collections()
        except Exception:
            return []
        out: List[str] = []
        for c in raw or []:
            out.append(getattr(c, "name", None) or str(c))
        return out

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            return False
        with self._lock:
            self._collections.pop(collection_name, None)
            self._space.pop(collection_name, None)
            cache = getattr(self, "_bm25_cache", None)
            if isinstance(cache, dict):
                cache.pop(collection_name, None)  # 库已删除，词法索引一并丢弃
        meta = self._load_meta()
        if meta.pop(collection_name, None) is not None:
            self._save_meta(meta)
        return True

    # ===================== 入库 =====================

    def add_text(
        self,
        text: str,
        *,
        source_id: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_name: Optional[str] = None,
        replace: bool = True,
    ) -> Dict[str, Any]:
        """把一段纯文本入库（经验库/剪贴板/接口数据等非文件来源用）。

        source_id 是幂等键：同 id 再次入库会先删旧块（replace=True）。
        """
        name = collection_name or self.default_collection
        text = (text or "").strip()
        if not text:
            return {"status": "skipped", "reason": "内容为空", "collection": name,
                    "source_id": source_id, "chunk_count": 0}

        coll = self._collection(name)
        self._check_embed_model(name)

        content_hash = self._content_hash(text, source_id)
        doc_id = f"{source_id}-{content_hash}"

        # 已有完全相同的内容 -> 跳过，避免重复灌库
        try:
            existed = coll.get(where={"doc_id": doc_id}, include=[])
            if existed and existed.get("ids"):
                return {"status": "unchanged", "collection": name, "source_id": source_id,
                        "doc_id": doc_id, "chunk_count": len(existed["ids"])}
        except Exception:
            pass

        if replace:
            self.delete_by_source(source_id, collection_name=name)

        chunks = self._chunk_text(text)
        if not chunks:
            return {"status": "skipped", "reason": "分块后无有效内容",
                    "collection": name, "source_id": source_id, "chunk_count": 0}

        embeddings = self._embed_many(chunks)
        if len(embeddings) != len(chunks):
            raise RagUnavailable(
                f"向量化数量不匹配（{len(embeddings)} != {len(chunks)}）"
            )

        base = {
            "source_id": source_id,
            "source_name": source_name or source_id,
            "doc_id": doc_id,
            "content_hash": content_hash,
            "collection": name,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if metadata:
            base.update(metadata)
        base = _clean_meta(base)

        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metas = [{**base, "chunk_index": i, "chunk_total": len(chunks)} for i in range(len(chunks))]
        coll.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=chunks)
        # 正文已变，BM25 词法索引必须失效重建。
        # 不能只靠缓存 key 里的 len(docs)：add_text 的流程是「先删旧块再加新块」，
        # 若新旧块数恰好相同，key 不变 → 缓存命中 → 继续用旧正文索引（静默 stale）。
        self._bump_bm25_stamp(name)

        meta = self._load_meta()
        entry = meta.get(name) or {}
        entry["embed_model"] = self.cfg.embed_model
        entry["dim"] = len(embeddings[0]) if embeddings else entry.get("dim")
        entry["updated_at"] = base["indexed_at"]
        meta[name] = entry
        self._save_meta(meta)

        return {"status": "success", "collection": name, "source_id": source_id,
                "doc_id": doc_id, "chunk_count": len(chunks),
                "chars": len(text), "dim": len(embeddings[0]) if embeddings else 0}

    def add_document(
        self,
        filepath: str | os.PathLike[str],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """入库单个文件。同一文件内容未变时返回 status=unchanged。"""
        path = Path(filepath).expanduser()
        text = self.parse_file(path)
        # source_id 用归一化键（匹配/删除用），source_file 保留原样（展示用）
        source_key = _norm_source_id(path)
        if not text.strip():
            return {"status": "skipped", "reason": "文件内容为空",
                    "collection": collection_name or self.default_collection,
                    "source_id": source_key, "chunk_count": 0}
        try:
            st = path.stat()
            extra = {"source_file": str(path), "source_filename": path.name,
                     "file_size": st.st_size, "file_mtime": st.st_mtime}
        except OSError:
            extra = {"source_file": str(path), "source_filename": path.name}
        merged = {**extra, **(metadata or {})}
        res = self.add_text(
            text,
            source_id=source_key,
            collection_name=collection_name,
            metadata=merged,
            source_name=path.name,
        )
        res["source_file"] = str(path)
        return res

    def add_documents(
        self,
        paths: Iterable[str | os.PathLike[str]],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """批量入库；单个文件失败不中断整批（失败原因收在 errors 里）。"""
        ok: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        for p in paths:
            try:
                r = self.add_document(p, collection_name=collection_name, metadata=metadata)
                if r.get("status") in ("success", "unchanged"):
                    ok.append(r)
                else:
                    errors.append({"file": str(p), "reason": str(r.get("reason") or r.get("status"))})
            except Exception as e:
                errors.append({"file": str(p), "reason": f"{type(e).__name__}: {e}"})
        return {
            "status": "success" if ok and not errors else ("partial" if ok else "failed"),
            "collection": collection_name or self.default_collection,
            "added": len(ok),
            "chunks": sum(int(r.get("chunk_count") or 0) for r in ok),
            "unchanged": sum(1 for r in ok if r.get("status") == "unchanged"),
            "errors": errors,
            "results": ok,
        }

    def add_directory(
        self,
        dirpath: str | os.PathLike[str],
        collection_name: Optional[str] = None,
        patterns: Optional[Sequence[str]] = None,
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """入库整个目录（只收 SUPPORTED_EXT，自动跳过 .git/.venv 等）。"""
        root = Path(dirpath).expanduser()
        if not root.is_dir():
            raise IOError(f"不是目录：{root}")
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".agent",
                ".idea", ".vscode", "dist", "build"}
        pats = [p.lower() for p in (patterns or [])]
        files: List[Path] = []
        walker = root.rglob("*") if recursive else root.glob("*")
        for p in walker:
            if not p.is_file():
                continue
            if any(part in skip for part in p.relative_to(root).parts[:-1]):
                continue
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if pats and not any(p.match(pat) for pat in pats):
                continue
            files.append(p)
        files.sort()
        res = self.add_documents(files, collection_name=collection_name, metadata=metadata)
        res["scanned"] = len(files)
        res["dir"] = str(root)
        return res

    # ===================== 删除 =====================

    def delete_by_source(self, source_id: str, collection_name: Optional[str] = None) -> int:
        """按来源删除全部块（重新入库前清理旧版本）。

        先走精确匹配（快，覆盖归一化键入库的新数据）；若删不到任何块，
        再扫描全部 source_id 做**归一化比对**兜底——用于清理历史遗留数据：
        归一化之前入库的块，source_id 是 `c:\\...` / `C:\\...` 这类原始写法，
        与新键字面不等，精确匹配删不掉，会一直残留在库里污染检索。
        """
        name = collection_name or self.default_collection
        try:
            coll = self._collection(name)
        except Exception:
            return 0

        target = _norm_source_id(source_id)
        deleted = 0
        # 依次尝试：归一化键（覆盖新数据）→ 原始入参（防御性兼容）
        for key in dict.fromkeys([target, source_id]):
            if not key:
                continue
            try:
                before = coll.count()
                coll.delete(where={"source_id": key})
                deleted += max(0, before - coll.count())
            except Exception:
                continue
        if deleted:
            self._bump_bm25_stamp(name)  # 正文已删，词法索引必须失效
            return deleted

        # 兜底：扫描历史遗留的非归一化 source_id
        try:
            res = coll.get(include=["metadatas"], limit=min(coll.count(), 5000) or 1)
        except Exception:
            return 0
        stale_ids = set()
        for m in (res.get("metadatas") or []):
            m = m or {}
            sid = m.get("source_id")
            if isinstance(sid, str) and sid and _norm_source_id(sid) == target:
                stale_ids.add(sid)
        if not stale_ids:
            return 0
        removed = 0
        for sid in stale_ids:
            try:
                before = coll.count()
                coll.delete(where={"source_id": sid})
                removed += max(0, before - coll.count())
            except Exception:
                continue
        if removed:
            self._bump_bm25_stamp(name)  # 兜底删除同样会让正文变化，词法索引必须失效
        return removed

    def delete_document(self, filepath: str | os.PathLike[str],
                        collection_name: Optional[str] = None) -> int:
        return self.delete_by_source(str(Path(filepath).expanduser()), collection_name)

    # ===================== 检索 =====================

    def search(
        self,
        query: str,
        top_k: int = 3,
        collection_name: Optional[str] = None,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """相似度检索，返回按相关度降序的结果列表。

        min_score: 相关度门槛，None 表示用 cfg.min_score。
                   低于门槛的结果会被丢弃（纯向量检索在中文下常有「全都 0.7 左右」
                   的假高分问题，门槛是过滤噪音的主要手段）。
        collection 不存在、query 为空、embedding 不可用时返回 []（不抛异常，
        检索是增强能力，不该打断主流程）。
        """
        name = collection_name or self.default_collection
        query = (query or "").strip()
        if not query:
            return []
        # 门槛按库取值（每个库的标定结果不同），未显式传参时用该库标定值或基线值
        threshold = self._min_score_for(name) if min_score is None else float(min_score)
        if name not in self.list_collections():
            return []
        try:
            coll = self._collection(name)
            if coll.count() == 0:
                return []
            qvec = self._embed_one(query)
            # 门槛会丢弃结果，所以向库多要一些再筛，避免 top_k 被过滤到不足
            fetch = max(int(top_k), 1) * 3 if threshold > 0 else max(1, int(top_k))
            res = coll.query(
                query_embeddings=[qvec],
                n_results=max(1, min(fetch, max(1, coll.count()))),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            dist = dists[i] if i < len(dists) else 0.0
            score = self._relevance(name, dist)
            if score < threshold:
                continue
            out.append({
                "content": doc or "",
                "metadata": dict(meta),
                "distance": float(dist) if isinstance(dist, (int, float)) else 0.0,
                "relevance_score": round(score, 4),
                "source": meta.get("source_name") or meta.get("source_file") or "",
                "collection": name,
            })
        out.sort(key=lambda x: x["relevance_score"], reverse=True)
        return out[: max(1, int(top_k))]

    def search_multi(
        self,
        query: str,
        collections: Sequence[str],
        top_k: int = 3,
        min_score: Optional[float] = None,
        hybrid: bool = False,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """跨多个 collection 检索后统一排序（知识库 + 经验库联检用）。

        hybrid=False（纯向量）：各库分数同量纲（同一 embedding 模型 + cosine 空间），
        直接按 relevance_score 归并即可。

        hybrid=True：**不能**直接按 relevance_score / rrf_score 排序。
          search_hybrid 有两条出口，rrf_score 的量纲不同：
            · 向量高置信短路 → rrf_score 被设成余弦分（约 0.6）
            · 正常 RRF 融合   → rrf_score 是名次倒数之和（约 0.07）
          跨库比大小会让「走了短路的库」无条件压过「走了融合的库」，
          与内容相关性完全无关。所以把每个库的结果当成一个排名，
          再做一次 RRF，量纲统一后才可跨库比较。
        """
        k = max(1, int(top_k))
        rankings: List[List[Dict[str, Any]]] = []
        for name in collections:
            if not name:
                continue
            if hybrid:
                rl = self.search_hybrid(query, top_k=k, collection_name=name,
                                        min_score=min_score, rrf_k=rrf_k)
            else:
                rl = self.search(query, top_k=k, collection_name=name,
                                 min_score=min_score)
            if rl:
                rankings.append(rl)
        if not rankings:
            return []

        if not hybrid:
            merged = [x for r in rankings for x in r]
            merged.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            return merged[:k]

        if len(rankings) == 1:
            return rankings[0][:k]

        fused: Dict[str, Dict[str, Any]] = {}
        for rl in rankings:
            for rank, item in enumerate(rl, 1):
                # 不同库的 id 可能同形，用 (collection, id) 做唯一键
                key = f"{item.get('collection') or ''}|{item.get('id') or ''}"
                slot = fused.get(key)
                if slot is None:
                    slot = dict(item)
                    slot["_x_rrf"] = 0.0
                    fused[key] = slot
                slot["_x_rrf"] += 1.0 / (rrf_k + rank)
        out = list(fused.values())
        for it in out:
            it["rrf_score"] = round(it.pop("_x_rrf"), 6)
        out.sort(key=lambda x: (x["rrf_score"], x.get("relevance_score", 0.0)),
                 reverse=True)
        return out[:k]

    # ===================== 混合检索（第五阶段）=====================
    #
    # 为什么需要：纯向量检索跨不过「词汇鸿沟」。实测严格评测集 20 条口语化查询失败 6 条，
    # 全是用户口语与文档术语字面不重叠（「最多能跑多快」vs「吞吐上限 12,000 条/秒」、
    # 「防火墙放行哪些号」vs「端口清单 8443」）。人工把查询改写成文档术语后，6/6 全部修复。
    #
    # 三条通道互补：
    #   1. 原查询向量检索     —— 语义相近即可命中（常规问法的主力）
    #   2. 改写查询向量检索   —— 跨词汇鸿沟（口语 → 术语）
    #   3. BM25 词法检索      —— 精确标识符（错误码 E4201、端口 8443、参数 batch_size）
    # 用 RRF（Reciprocal Rank Fusion）融合：只看各通道的**名次**不看分数量纲，
    # 因此不必担心「向量余弦」和「BM25 分数」不可比的问题。

    def _get_expander(self, expander: Any = None) -> Any:
        """取查询改写器（进程内复用）；不可用时返回 None。"""
        exp = expander if expander is not None else getattr(self, "expander", None)
        if exp is not None:
            return exp
        try:
            from query_expander import get_expander
            exp = get_expander()
            self.expander = exp
            return exp
        except Exception:
            return None

    def _expand_query(self, query: str, expander: Any = None) -> Any:
        """取查询扩展结果，按通道分组返回。

        返回对象带 .vector_queries（原查询 + 书面问句改写）与
        .lexical_queries（原查询 + 术语词袋）两个属性。
        改写器不可用/调用失败时返回一个只含原查询的降级对象，
        保证混合检索在任何情况下都能工作（只是少了扩展增益）。
        """
        from query_expander import Expansion

        exp = self._get_expander(expander)
        if exp is None:
            return Expansion(query=query, ok=False)
        try:
            result = exp.expand_structured(query)
        except Exception as e:
            self.last_error = f"_expand_query: {type(e).__name__}: {e}"
            return Expansion(query=query, ok=False)
        # 防御：改写器返回空查询时不要污染检索
        if not result or not (result.query or "").strip():
            return Expansion(query=query, ok=False)
        return result

    def _vector_ranked(self, query: str, name: str, fetch: int,
                       where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """单条查询的向量检索排序表（不做门槛过滤，融合后再统一判定）。"""
        try:
            coll = self._collection(name)
            res = coll.query(
                query_embeddings=[self._embed_one(query)],
                n_results=max(1, min(fetch, max(1, coll.count()))),
                where=where or None,
                # 注意：ids **不能**写进 include —— ChromaDB 只接受
                # documents/embeddings/metadatas/distances/uris/data，传 ids 会抛
                # ValueError（而 ids 本来就会默认返回）。
                # 这个错误曾被下面的 except 静默吞掉，导致混合检索恒返回空列表，
                # 所以这里把异常记进 self.last_error，便于排查。
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            self.last_error = f"_vector_ranked: {type(e).__name__}: {e}"
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            dist = dists[i] if i < len(dists) else 0.0
            meta = metas[i] if i < len(metas) and metas[i] else {}
            out.append({
                "id": ids[i] if i < len(ids) and ids[i] else f"{name}#{i}",
                "content": doc or "",
                "metadata": dict(meta),
                "distance": float(dist) if isinstance(dist, (int, float)) else 0.0,
                "relevance_score": self._relevance(name, dist),
                "source": meta.get("source_name") or meta.get("source_file") or "",
                "collection": name,
                "channel": "vector",
            })
        out.sort(key=lambda x: x["relevance_score"], reverse=True)
        return out

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        collection_name: Optional[str] = None,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        expand: bool = True,
        use_lexical: bool = True,
        rrf_k: int = 60,
        expander: Any = None,
    ) -> List[Dict[str, Any]]:
        """混合检索：查询改写 + 多路向量 + BM25 词法，RRF 融合排序。

        门槛语义（重要）：标定出的 min_score 是**向量余弦**口径，所以只用它过滤
        「向量通道」的候选，保证门槛的可解释性不被融合分数污染。
        纯词法命中的候选（向量分不达标但精确匹配到标识符）单独放行，
        且必须满足「查询里的标识符原样出现在块中」这一硬条件，避免引入噪音。

        返回结果按融合分排序，每项额外带：
          rrf_score     融合分（仅用于排序）
          matched_by    命中来源：vector / vector+lexical / lexical
          variants_used 实际参与检索的查询变体
        """
        name = collection_name or self.default_collection
        query = (query or "").strip()
        if not query or name not in self.list_collections():
            return []
        threshold = self._min_score_for(name) if min_score is None else float(min_score)
        try:
            total = self._collection(name).count()
        except Exception:
            return []
        if total == 0:
            return []

        k = max(1, int(top_k))
        # 候选池大小必须与 top_k **解耦**，固定为常量。
        # 原来写 fetch = max(k*4, 10)，后果是 top_k 不只改变返回条数，
        # 还改变每个通道的候选池 → RRF 名次随之变化 → 同一个问题在
        # --top-k=1 和 --top-k=3 下给出不同答案（实测 top_k=1 时 top1 是
        # 完全错误的块，top_k=3 才正确）。这让检索结果不可预期、指标不可比。
        # 固定候选池后，top_k 只做截断，排序保持稳定。
        fetch = min(self.cfg.hybrid_fetch, max(1, total))

        # ---- 向量高置信短路 ----
        # RRF 只看名次、丢弃分数量纲：向量通道里「0.625 vs 0.622」和
        # 「0.60 vs 0.40」都同样表现为 rank1/rank2（1/61 vs 1/62），
        # 于是「两个改写变体都排第 2」的块能击败「原查询向量排第 1」的块。
        # 实测后果：严格集修复 4 条却退化 3 条，净增益只有 +1。
        #
        # 所以当向量通道已经给出**高置信**答案时，直接采信它，不再融合。
        # 判据用实测数据（_diag_gap.py）：
        #   退化组「数据从源头到目标…」gap=0.027、「上证指数跌了多少」gap=0.040
        #   → top1 过门槛且分差明确，属于向量本已做对、不该被融合翻盘的情形；
        #   修复组「买哪几个行业」top1=0.492、「防火墙放行哪些号」top1=0.523
        #   → 都**低于**门槛 0.5284，短路条件不成立，照常进融合（这正是它们被修好的原因）。
        # 即：短路只拦「向量已确定」的情况，对「向量不确定」的情况完全不影响。
        if self.cfg.hybrid_shortcut:
            try:
                base = self._vector_ranked(query, name, fetch, where)
            except Exception:
                base = []
            passed = [b for b in base if b["relevance_score"] >= threshold]
            if len(passed) >= 2:
                gap = passed[0]["relevance_score"] - passed[1]["relevance_score"]
                if gap >= self.cfg.shortcut_gap:
                    out = passed[:k]
                    # rrf_score 必须与正常融合分支同量纲（名次倒数之和），
                    # 否则跨库 search_multi 比较时会失真——短路分支若填余弦分（~0.6），
                    # 会无条件压过融合分支的名次倒数（~0.07），与相关性无关。
                    # 短路结果是单一排名表，按 RRF 公式逐个赋分即可对齐量纲。
                    for rank, it in enumerate(out, 1):
                        it["matched_by"] = "vector_shortcut"
                        it["rrf_score"] = round(1.0 / (rrf_k + rank), 6)
                        it["variants_used"] = [query]
                    return out
            # 只有 1 个过门槛（或 0 个）时不短路：
            # 这正说明向量通道不确定，必须交给改写与词法通道补救。

        # ---- 查询扩展：两个通道用不同的查询形式 ----
        # 向量通道要「书面问句」（完整句式才有稳定语义方向）；
        # 词法通道要「术语词袋」（BM25 按词项匹配，整句会引入大量低 IDF 虚词噪音）。
        # 实测依据：把术语词袋喂给向量通道，严格集 top1 命中从 3/6 掉到 1/6。
        if expand:
            expansion = self._expand_query(query, expander)
            vec_queries = expansion.vector_queries
            lex_queries = expansion.lexical_queries
            expanded_ok = bool(getattr(expansion, "ok", False))
        else:
            vec_queries, lex_queries, expanded_ok = [query], [query], False

        def _dedup(items: List[str]) -> List[str]:
            out: List[str] = []
            for it in items:
                it = (it or "").strip()
                if it and it not in out:
                    out.append(it)
            return out

        vec_queries = _dedup(vec_queries)[:4] or [query]
        lex_queries = _dedup(lex_queries)[:4] or [query]
        variants = vec_queries if vec_queries == lex_queries else vec_queries + lex_queries[1:]

        rankings: List[List[Dict[str, Any]]] = []
        for v in vec_queries:
            rl = self._vector_ranked(v, name, fetch, where)
            if rl:
                rankings.append(rl)
        if not rankings:
            return []

        if use_lexical:
            # 多条术语查询各自出一个排序表，一起进 RRF；
            # BM25 索引在 _bm25_index 内按 collection 缓存，多次调用不会重复建索引
            for lq in lex_queries:
                rl = self._lexical_ranked(lq, name, fetch, where)
                if rl:
                    rankings.append(rl)

        # ---- RRF 融合 ----
        # 用 item["channel"] 判断来源，不用 `rank_list is lexical` 的身份比较：
        # 后者依赖变量引用不变，一旦将来对 rankings 做拷贝/过滤就会静默失效。
        fused: Dict[str, Dict[str, Any]] = {}
        for rank_list in rankings:
            for rank, item in enumerate(rank_list, 1):
                key = item["id"]
                slot = fused.get(key)
                if slot is None:
                    slot = {**item, "rrf_score": 0.0, "_vec": 0.0, "_lex": False}
                    fused[key] = slot
                slot["rrf_score"] += 1.0 / (rrf_k + rank)
                slot["_vec"] = max(slot["_vec"], float(item.get("relevance_score") or 0.0))
                if item.get("channel") == "lexical":
                    slot["_lex"] = True
                else:
                    # 向量通道带完整 metadata/source，用它补全词法通道的空缺
                    if not slot.get("metadata"):
                        slot["metadata"] = item.get("metadata") or {}
                    if not slot.get("source"):
                        slot["source"] = item.get("source") or ""

        # ---- 门槛判定 ----
        id_tokens = _identifier_tokens(query)
        admitted: List[Dict[str, Any]] = []
        need_backfill: List[Dict[str, Any]] = []
        for slot in fused.values():
            vec_score = slot["_vec"]
            passed_vec = vec_score >= threshold
            # 纯词法放行条件：查询含标识符（错误码/端口/参数名）且原样出现在块里。
            # 这个硬条件是刻意的——词法分数与余弦不同量纲，不能直接套门槛，
            # 所以改用「精确标识符必须出现在正文」作为更严格的可解释判据。
            passed_lex = bool(
                slot["_lex"] and not passed_vec and id_tokens
                and any(t.lower() in (slot["content"] or "").lower() for t in id_tokens)
            )
            if not (passed_vec or passed_lex):
                continue
            slot["relevance_score"] = round(vec_score, 4)
            slot["rrf_score"] = round(slot["rrf_score"], 6)
            slot["matched_by"] = ("vector+lexical" if (passed_vec and slot["_lex"])
                                  else ("lexical" if passed_lex else "vector"))
            slot.pop("_vec", None)
            slot.pop("_lex", None)
            slot.pop("channel", None)
            slot["variants_used"] = variants
            if not slot.get("metadata") and not slot.get("source"):
                need_backfill.append(slot)  # 仅被词法通道命中，出处待补
            admitted.append(slot)

        if need_backfill:
            self._backfill_meta(name, need_backfill)

        # 融合分优先；同分时用向量相关度兜底（融合分粒度粗，容易并列）
        admitted.sort(key=lambda x: (x["rrf_score"], x["relevance_score"]), reverse=True)
        return admitted[:k]

    # ---------- BM25 词法通道 ----------
    def _bm25_index(self, name: str) -> Optional["_BM25Index"]:
        """构造（或复用）某 collection 的 BM25 索引。

        缓存失效判定用 (块数, 变更戳)：块数变化能覆盖增删，
        变更戳覆盖「删 N 条同时加 N 条」这种块数不变的替换。
        """
        cache = getattr(self, "_bm25_cache", None)
        if cache is None:
            cache = self._bm25_cache = {}
        try:
            coll = self._collection(name)
            count = coll.count()
            res = coll.get(include=["documents"], limit=min(count, 20000) or 1)
        except Exception:
            return None
        docs = list(res.get("documents") or [])
        ids = list(res.get("ids") or [])
        if not docs:
            return None
        stamp = int(getattr(self, "_bm25_stamp", {}).get(name, 0))
        key = (name, len(docs), stamp)
        hit = cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        index = _BM25Index(docs, ids)
        cache[name] = (key, index)
        return index

    def _bump_bm25_stamp(self, name: str) -> None:
        stamps = getattr(self, "_bm25_stamp", None)
        if stamps is None:
            stamps = self._bm25_stamp = {}
        stamps[name] = int(stamps.get(name, 0)) + 1

    def _lexical_ranked(self, query: str, name: str, fetch: int,
                        where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """BM25 词法检索排序表。词法分数不进门槛判定，只提供名次参与 RRF。"""
        index = self._bm25_index(name)
        if index is None:
            return []
        try:
            scored = index.search(query, fetch)
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        # where 过滤：BM25 索引本身不认识 metadata 条件，必须在这里补一道，
        # 否则调用方传了 where 时，词法通道会返回违反过滤条件的块。
        allowed_ids: Optional[set] = None
        if where:
            try:
                coll = self._collection(name)
                res = coll.get(where=where, include=[], limit=min(coll.count(), 20000) or 1)
                allowed_ids = set(res.get("ids") or [])
            except Exception:
                allowed_ids = set()  # 条件非法时宁可返回空，也不返回越界结果
        for doc_id, bm25_score, content in scored:
            if bm25_score <= 0:
                continue
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            out.append({
                "id": doc_id,
                "content": content,
                "metadata": {},          # 词法通道不取 metadata，融合时由向量通道或回填补全
                "distance": 0.0,
                "relevance_score": 0.0,  # 词法分与余弦不同量纲，不参与门槛
                "bm25_score": round(bm25_score, 4),
                "source": "",
                "collection": name,
                "channel": "lexical",
            })
        return out

    def _backfill_meta(self, name: str, slots: List[Dict[str, Any]]) -> None:
        """为「仅被词法通道命中」的块补全 metadata / source。

        BM25 索引只存正文，不带元数据。若某块只在词法通道命中，
        不补全就会返回一条**没有出处**的检索结果——用户看到答案却不知来自哪个文件，
        引用与追责都无从谈起。这里按 id 批量取回，一次查询搞定。
        """
        if not slots:
            return
        ids = [s["id"] for s in slots if s.get("id")]
        if not ids:
            return
        try:
            res = self._collection(name).get(ids=ids, include=["metadatas"])
        except Exception:
            return
        metas = res.get("metadatas") or []
        got_ids = res.get("ids") or []
        lookup = {i: (metas[n] or {}) for n, i in enumerate(got_ids) if n < len(metas)}
        for slot in slots:
            meta = lookup.get(slot.get("id"))
            if not meta:
                continue
            slot["metadata"] = dict(meta)
            slot["source"] = (meta.get("source_name") or meta.get("source_file")
                              or slot.get("source") or "")

    def search_lexical(self, query: str, top_k: int = 5,
                       collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """纯 BM25 词法检索（调试/精确标识符查询用）。"""
        name = collection_name or self.default_collection
        hits = self._lexical_ranked((query or "").strip(), name, max(1, int(top_k)))
        self._backfill_meta(name, hits)  # 独立调用时同样要有出处
        return hits[: max(1, int(top_k))]

    @staticmethod
    def format_context(results: Sequence[Dict[str, Any]], max_chars: int = 3000) -> str:
        """把检索结果渲染成可注入系统提示词的参考资料块。"""
        if not results:
            return ""
        lines: List[str] = []
        used = 0
        for i, r in enumerate(results, 1):
            src = r.get("source") or "未知来源"
            body = (r.get("content") or "").strip()
            head = f"[{i}] 相关度 {r.get('relevance_score', 0):.2f} | 来源：{src}\n"
            room = max_chars - used - len(head) - 4
            if room <= 80:
                break
            if len(body) > room:
                body = body[:room] + "…[截断]"
            block = head + body
            lines.append(block)
            used += len(block) + 2
        return "\n\n".join(lines)

    # ===================== 观测 =====================

    def list_sources(self, collection_name: Optional[str] = None,
                     limit: int = 200) -> List[Dict[str, Any]]:
        name = collection_name or self.default_collection
        try:
            coll = self._collection(name)
            res = coll.get(include=["metadatas"], limit=limit)
        except Exception:
            return []
        grouped: Dict[str, Dict[str, Any]] = {}
        for meta in (res.get("metadatas") or []):
            meta = meta or {}
            sid = str(meta.get("source_id") or meta.get("source_file") or "?")
            g = grouped.setdefault(sid, {"source_id": sid, "chunks": 0,
                                         "source_name": meta.get("source_name") or "",
                                         "indexed_at": meta.get("indexed_at") or ""})
            g["chunks"] += 1
        return sorted(grouped.values(), key=lambda x: x["source_id"])

    def stats(self) -> Dict[str, Any]:
        meta = self._load_meta()
        colls: List[Dict[str, Any]] = []
        for name in self.list_collections():
            try:
                count = self._collection(name).count()
            except Exception:
                count = -1
            entry = meta.get(name) or {}
            colls.append({
                "name": name,
                "chunks": count,
                "sources": len(self.list_sources(name)),
                "embed_model": entry.get("embed_model", ""),
                "dim": entry.get("dim"),
                "space": self._space.get(name, _COSINE),
                "min_score": entry.get("min_score"),
            })
        return {
            "db_path": str(self.db_path),
            "embed_model": self.cfg.embed_model,
            "ollama_url": self.cfg.ollama_url,
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
            "default_collection": self.default_collection,
            "collections": colls,
            "total_chunks": sum(c["chunks"] for c in colls if c["chunks"] > 0),
            "min_score": self.cfg.min_score,
        }

    # ===================== 分数标定 =====================

    def calibrate_scores(
        self,
        positives: Sequence[Tuple[str, str]],
        negatives: Optional[Sequence[str]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """用「正例对 + 负例查询」标定相关度门槛，解决换模型后阈值失效的问题。

        为什么需要：余弦分数的绝对值**不可跨模型比较**。nomic-embed-text 在中文下
        连「天气晴朗」都能给到 0.68，导致 min_score=0.30 形同虚设（全部噪音过线）；
        bge-m3 的分布则完全不同。所以每换一个模型，门槛都必须重新实测标定。

        :param positives: [(查询, 期望命中的文档文本), ...] —— 期望检索到的对
        :param negatives: [查询, ...] —— 库里**不该**有答案的查询，用来测噪音地板
        :return: {"suggested_min_score": float, "margin": float, ...}
                 margin = 正例最低分 - 负例最高分；margin <= 0 说明模型区分不开，
                 此时任何阈值都救不了，应换模型而不是调阈值。
        """
        name = collection_name or self.default_collection
        pos_scores: List[float] = []
        pos_detail: List[Dict[str, Any]] = []
        for query, doc in positives:
            try:
                qv = self._embed_one(query)
                dv = self._embed_one(doc)
            except Exception as e:
                return {"ok": False, "error": f"向量化失败：{type(e).__name__}: {e}"}
            # 与检索链路保持一致：cosine 空间下 relevance = 1 - distance = 余弦
            score = self._relevance(name, 1.0 - _cosine(qv, dv))
            pos_scores.append(score)
            pos_detail.append({"query": query[:60], "score": round(score, 4)})

        neg_scores: List[float] = []
        neg_detail: List[Dict[str, Any]] = []
        for query in (negatives or []):
            hits = self.search(query, top_k=1, collection_name=name, min_score=0.0)
            if hits:
                neg_scores.append(hits[0]["relevance_score"])
                neg_detail.append({"query": query[:60],
                                   "score": hits[0]["relevance_score"],
                                   "matched": (hits[0].get("content") or "")[:60]})

        if not pos_scores:
            return {"ok": False, "error": "positives 不能为空"}

        pos_min = min(pos_scores)
        neg_max = max(neg_scores) if neg_scores else None
        margin = round(pos_min - neg_max, 4) if neg_max is not None else None

        if margin is None:
            # 没有负例：取正例最低分下方留 10% 余量作为保守门槛
            suggested = round(pos_min * 0.90, 4)
        elif margin <= 0:
            # 正负例分数重叠：阈值无法分开，问题在模型/语料而非门槛
            suggested = round((pos_min + neg_max) / 2, 4)
        else:
            # 落在正例最低分与负例最高分之间，偏向正例一侧留容错
            suggested = round(neg_max + margin * 0.65, 4)

        return {
            "ok": True,
            "embed_model": self.cfg.embed_model,
            "collection": name,
            "positives": pos_detail,
            "negatives": neg_detail,
            "pos_min": round(pos_min, 4),
            "pos_max": round(max(pos_scores), 4),
            "neg_max": round(neg_max, 4) if neg_max is not None else None,
            "margin": margin,
            "suggested_min_score": suggested,
            "separable": bool(margin is None or margin > 0.05),
            "verdict": (
                "模型可区分，按 suggested_min_score 设置门槛"
                if (margin is None or margin > 0.05)
                else "⚠️ 正负例分数重叠，换阈值无济于事——需换 embedding 模型或改进分块"
            ),
        }

    def auto_calibrate(
        self,
        collection_name: Optional[str] = None,
        sample_pairs: int = 120,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """从库内已有数据自动标定相关度门槛，无需人工标注正负例。

        取样走**真实检索路径**（query → chunk），而不是块与块之间的余弦：
        1. 从每个块的正文造一个「伪查询」（剥掉标题前缀，取首句或表格首行）；
        2. 用伪查询调 search()，记录它命中**自己那块**的分数 → 相关命中地板；
        3. 同一伪查询在**其他小节**块上的最高分 → 噪音天花板；
        4. 门槛取两者的中点：低于地板不会误杀有效结果，又能压住大部分噪音。

        为什么不按「同文档/跨文档」分组取样：加了【标题路径】前缀后，
        跨文档块对因标题不同而天然虚低，用它当噪音会低估天花板、把门槛标得偏高，
        实测会把大量有效查询误杀。真实 RAG 里最典型的噪音是「命中同一份文档的
        错误章节」，所以负例统一按跨小节统计。

        :param sample_pairs: 最多取多少个伪查询样本（固定随机种子，结果可复现）
        :param persist: 标定成功后是否写入库元数据（之后检索自动沿用）
        """
        name = collection_name or self.default_collection
        if name not in self.list_collections():
            return {"ok": False, "error": f"知识库 {name} 不存在"}
        try:
            coll = self._collection(name)
            total = coll.count()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if total < 2:
            return {"ok": False, "error": f"知识库 {name} 只有 {total} 个块，样本不足，无法标定"}

        # 取出全部块的向量、归属文档与正文（正文用于取小节标题前缀）
        try:
            res = coll.get(include=["metadatas", "embeddings", "documents"],
                           limit=min(total, 2000))
        except Exception as e:
            return {"ok": False, "error": f"读取向量失败：{type(e).__name__}: {e}"}
        metas = list(res.get("metadatas") or [])
        # embeddings 可能是 numpy 2D 数组：`x or []` 会触发 __bool__ 歧义报错，必须显式判 None
        raw_embs = res.get("embeddings")
        embs = [] if raw_embs is None else list(raw_embs)
        raw_docs = res.get("documents")
        docs_txt = [] if raw_docs is None else list(raw_docs)
        if len(embs) < 2:
            return {"ok": False, "error": "库内向量不足，无法标定"}

        # 小节身份：取块首的【标题路径】前缀。负例统一按「跨小节」判定，
        # 不再分单文档/多文档——跨文档块对因标题前缀不同而天然虚低，
        # 用它当噪音会低估天花板、把门槛标偏高（实测会误杀大量有效查询）。
        def _section_of(i: int) -> str:
            m = metas[i] if i < len(metas) and metas[i] else {}
            txt = str(docs_txt[i]) if i < len(docs_txt) and docs_txt[i] else ""
            head = txt.lstrip()
            if head.startswith("【"):
                end = head.find("】")
                if end > 0:
                    return head[1:end]
            # 没有标题前缀时退到「来源 + chunk 序号」，保证每个块各自成节
            return f"{m.get('source_id') or '?'}#chunk{m.get('chunk_index', i)}"

        group_key: List[str] = [_section_of(i) for i in range(len(embs))]
        n_sections = len(set(group_key))
        if n_sections < 2:
            return {"ok": False,
                    "error": "库内只有一个语义小节，无法构造跨小节噪音样本；"
                             "请入库内容更丰富的文档后重试"}

        # ---------- 取样：走真实检索路径，而不是块-块余弦 ----------
        # 为什么不用块-块相似度（旧实现的做法）：
        #   门槛卡的是「查询 → 块」的相关度，而块-块相似度是另一个量纲，
        #   两者分布不同、不可直接套用。更糟的是块内容里带【标题路径】前缀，
        #   同文档块对天然共享产品名/文档名而相似度虚高，跨文档块对因标题不同而虚低，
        #   于是算出的 margin 实际在测「是否同一篇文档」，与「查询是否相关」无关。
        #   实测后果：据此标出 0.637 的门槛，把 9/25 条有效查询直接误杀（P@1 100%→64%）。
        #
        # 正确做法：拿块正文造「伪查询」，走真实 search() 通道打分——
        #   正例 = 伪查询命中它自己那块时的分数（相关命中地板）
        #   负例 = 同一伪查询在其他小节块上的最高分（噪音天花板）
        #   两者都是 query→chunk，与门槛同量纲，可直接比较。
        rng = random.Random(20260101)  # 固定种子，标定结果可复现

        identities: List[Tuple[str, Any]] = []  # 每块的 (source_id, chunk_index) 身份
        for i, m in enumerate(metas):
            m = m or {}
            identities.append((str(m.get("source_id") or ""), m.get("chunk_index")))

        pos: List[float] = []   # 相关命中分数
        neg: List[float] = []   # 跨小节噪音分数
        cand_idx = list(range(len(embs)))
        rng.shuffle(cand_idx)

        for i in cand_idx:
            if len(pos) >= sample_pairs:
                break
            body = _strip_title_prefix(str(docs_txt[i]) if i < len(docs_txt) else "")
            pq = _make_pseudo_query(body)
            if len(pq) < 8:
                continue  # 伪查询太短（纯符号/表格残渣），无意义
            try:
                hits = self.search(pq, top_k=5, collection_name=name, min_score=0.0)
            except Exception:
                continue
            if not hits:
                continue

            self_id = identities[i]
            self_score = None
            other_best = 0.0
            for h in hits:
                hm = h.get("metadata") or {}
                hid = (str(hm.get("source_id") or ""), hm.get("chunk_index"))
                if hid == self_id:
                    self_score = float(h.get("relevance_score") or 0.0)
                else:
                    # 只统计「跨小节」的噪音；同小节其他块仍属相关内容，不算噪音
                    j = next((k for k, v in enumerate(identities) if v == hid), None)
                    if j is not None and group_key[j] != group_key[i]:
                        other_best = max(other_best, float(h.get("relevance_score") or 0.0))
            if self_score is not None:
                pos.append(self_score)
            if other_best > 0:
                neg.append(other_best)

        if len(pos) < 3 or len(neg) < 3:
            return {"ok": False,
                    "error": f"取样不足（相关 {len(pos)} / 噪音 {len(neg)}）；"
                             f"语料太少或块内容过短，请入库更多文档后重试"}

        pos_sorted = sorted(pos)
        neg_sorted = sorted(neg)
        # 相关命中地板（P10：容忍 10% 的正常波动，门槛必须低于它，否则会误杀有效结果）
        p10 = pos_sorted[max(0, int(len(pos_sorted) * 0.10) - 1)]
        # 噪音天花板（P90：高于它的门槛能滤掉约 90% 噪音）
        neg_max = neg_sorted[-1]
        neg_p90 = neg_sorted[min(len(neg_sorted) - 1, int(len(neg_sorted) * 0.90))]

        # 门槛取「噪音天花板」与「相关地板」的中点：
        # 既不会高于 p10 而误杀有效结果，又能尽量压住噪音。
        # 偏向召回是刻意的取舍——漏掉噪音只是多读几段，误杀正确答案则是知识库静默失效。
        margin = round(p10 - neg_p90, 4)
        suggested = round((neg_p90 + p10) / 2, 4) if margin > 0 else round(p10 * 0.9, 4)
        suggested = round(min(max(suggested, 0.05), p10), 4)

        src_count = len({str((m or {}).get("source_id") or "?") for m in metas})
        result = {
            "ok": True,
            "embed_model": self.cfg.embed_model,
            "collection": name,
            "sample_basis": "pseudo_query_via_search",
            "samples": {"docs": src_count, "chunks": len(embs),
                        "sections": n_sections,
                        "pos_pairs": len(pos), "neg_pairs": len(neg)},
            "pos_p10": round(p10, 4),
            "pos_min": round(pos_sorted[0], 4),
            "neg_max": round(neg_max, 4),
            "neg_p90": round(neg_p90, 4),
            "margin": margin,
            "suggested_min_score": suggested,
            "separable": margin > 0,
            "verdict": (
                "相关命中地板高于跨小节噪音天花板，门槛取两者中点，可分离"
                if margin > 0 else
                "⚠️ 相关命中与跨小节噪音分数重叠（margin≤0）：门槛已按「不高于相关地板」"
                "保守取值，可避免误杀，但滤噪能力有限。建议检查文档主题是否过于集中、"
                "或改用 calibrate_scores 以真实问句标定。"
            ),
        }
        if margin <= 0:
            result["caveat"] = (
                "库内各小节主题相近，跨小节噪音与相关命中分数重叠，"
                "本次门槛只能保证不误杀、无法保证滤噪；"
                "入库主题更分散的文档后重跑 --kb-calibrate 可获得更可靠的门槛。"
            )

        # 安全闸门：门槛必须**不高于相关命中地板 p10**，否则会把有效结果一起滤掉。
        # suggested 的计算式已用 min(..., p10) 保证这一点；这里再做一次显式钳制，
        # 防止将来改动取样逻辑时丢掉该不变量——
        # 历史上就因为标出高于相关分数的门槛，导致知识库静默返回 0 条（不报错、极难排查）。
        suggested = round(min(suggested, p10), 4)
        result["suggested_min_score"] = suggested
        if persist:
            self.save_min_score(suggested, collection_name=name)
            result["persisted"] = True
        else:
            result["persisted"] = False
        return result


    def save_min_score(self, min_score: float,
                       collection_name: Optional[str] = None) -> None:
        """把标定出的门槛写进库元数据，之后检索该库时自动沿用。

        刻意不改写 self.cfg.min_score：那是全局共享状态，server / scheduler 会并发检索
        多个库，在这里改会让本次标定值串到其他库。检索侧统一走 _min_score_for(name)。
        """
        name = collection_name or self.default_collection
        meta = self._load_meta()
        entry = meta.get(name) or {}
        entry["min_score"] = float(min_score)
        meta[name] = entry
        self._save_meta(meta)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（标定用；检索链路走 ChromaDB 的 distance，二者等价）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0



# ===================== 进程内单例 =====================

_ENGINE: Optional[RagEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine(
    db_path: Optional[str] = None,
    embed_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    **kwargs: Any,
) -> RagEngine:
    """取全局唯一引擎（避免每个任务都重开一次 ChromaDB 连接）。

    参数只在首次创建时生效；后续调用返回同一实例。
    """
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            cfg = RagConfig(
                db_path=db_path or os.getenv("RAG_DB_PATH", "./agent_knowledge_base"),
                embed_model=embed_model or os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
                ollama_url=ollama_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
                **kwargs,
            )
            _ENGINE = RagEngine(cfg)
        return _ENGINE


def reset_engine() -> None:
    """测试用：丢弃单例。"""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None
