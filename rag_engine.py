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
                # 路径只取最近两级，太长会稀释语义
                current_title = " > ".join(heading_stack[-2:])
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

            # 把正文按「表格块 / 普通文本块」分段，表格不与其他内容混切
            segments: List[Tuple[bool, List[str]]] = []  # (是否表格, 行列表)
            cur_is_table = False
            cur: List[str] = []
            for line in body:
                is_table = bool(self._TABLE_ROW_RE.match(line))
                if cur and is_table != cur_is_table:
                    segments.append((cur_is_table, cur))
                    cur = []
                cur_is_table = is_table
                cur.append(line)
            if cur:
                segments.append((cur_is_table, cur))

            for is_table, seg_lines in segments:
                seg_text = "\n".join(l for l in seg_lines if l.strip()).strip()
                if not seg_text:
                    continue
                if is_table:
                    for piece in self._split_table(seg_lines, budget):
                        piece = piece.strip()
                        if piece:
                            chunks.append(prefix + piece if prefix else piece)
                elif len(seg_text) <= budget:
                    chunks.append(prefix + seg_text if prefix else seg_text)
                else:
                    for piece in self._split_long(seg_text):
                        chunks.append(prefix + piece if prefix else piece)

        # 去掉过短的碎块（<40 字符基本没有语义价值，只会当噪音被检索出来）
        return [c for c in chunks if len(c.strip()) >= 40]

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
        """内容指纹 = 正文 + 分块参数 + embedding 模型。

        为什么不能只哈希正文：幂等入库靠这个指纹判断「是否已存在完全相同的版本」。
        若只算正文，那么改了 chunk_size / 分块策略 / 换了 embedding 模型之后重新入库，
        会被误判为 unchanged 而跳过——旧向量（错误的切分方式）继续留在库里，
        改动的分块逻辑等于没生效。把参数一起哈希，配置一变就强制重建。
        """
        h = hashlib.sha256()
        h.update(text.encode("utf-8", errors="replace"))
        h.update(
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
        if not text.strip():
            return {"status": "skipped", "reason": "文件内容为空",
                    "collection": collection_name or self.default_collection,
                    "source_id": str(path), "chunk_count": 0}
        try:
            st = path.stat()
            extra = {"source_file": str(path), "source_filename": path.name,
                     "file_size": st.st_size, "file_mtime": st.st_mtime}
        except OSError:
            extra = {"source_file": str(path), "source_filename": path.name}
        merged = {**extra, **(metadata or {})}
        res = self.add_text(
            text,
            source_id=str(path),
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
        """按来源删除全部块（重新入库前清理旧版本）。"""
        name = collection_name or self.default_collection
        try:
            coll = self._collection(name)
            before = coll.count()
            coll.delete(where={"source_id": source_id})
            return max(0, before - coll.count())
        except Exception:
            return 0

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
    ) -> List[Dict[str, Any]]:
        """跨多个 collection 检索后统一排序（知识库 + 经验库联检用）。"""
        merged: List[Dict[str, Any]] = []
        for name in collections:
            if not name:
                continue
            merged.extend(self.search(query, top_k=top_k, collection_name=name,
                                      min_score=min_score))
        merged.sort(key=lambda x: x["relevance_score"], reverse=True)
        return merged[: max(1, top_k)]


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

        取样思路（无监督），分两种情况：
        - 多篇文档：正例 = 同文档内块对；负例 = 跨文档块对（主题通常不同）。
        - 单篇文档：跨文档负例无从构造，降级为「按小节」取样——
          正例 = 同小节内的块对（标题前缀相同，语义本就接近）；
          负例 = 跨小节的块对（不同章节，相关性显著更低）。
          小节身份取自结构分块写入的【标题路径】前缀。

        取正例分布的低分位（P10）作为门槛：既容忍同组内的正常波动，
        又能滤掉组间噪音。块数不足 2 时无法标定，返回 ok=False。

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

        # 按来源文档分组（同文档内 = 正例，跨文档 = 负例）
        groups: Dict[str, List[int]] = {}
        group_key: List[str] = []
        for i, m in enumerate(metas):
            m = m or {}
            src = str(m.get("doc_id") or m.get("source_id") or m.get("source_file") or "?")
            groups.setdefault(src, []).append(i)
            group_key.append(src)

        # 小节身份：取块首的【标题路径】前缀，供单文档降级取样用
        def _section_of(i: int) -> str:
            m = metas[i] if i < len(metas) and metas[i] else {}
            txt = str(docs_txt[i]) if i < len(docs_txt) and docs_txt[i] else ""
            head = txt.lstrip()
            if head.startswith("【"):
                end = head.find("】")
                if end > 0:
                    return head[1:end]
            # 没有前缀时退到 chunk 序号段，至少能区分不同位置
            return f"{m.get('source_id') or '?'}#chunk{m.get('chunk_index', i)}"

        multi_doc = len(groups) >= 2
        docs_with_multi = [idxs for idxs in groups.values() if len(idxs) >= 2]

        if multi_doc:
            sample_basis = "cross_document"
        else:
            # 单篇文档：改按小节分组
            if not docs_with_multi:
                return {"ok": False,
                        "error": "库内只有单一文档且不足 2 块，无法构造正负例；"
                                 "请入库更多文档后重试，或用 calibrate_scores 手工指定正负例"}
            group_key = [_section_of(i) for i in range(len(embs))]
            sample_basis = "cross_section"
            # 重新按小节分组，正例要从「同小节」里取
            sec_groups: Dict[str, List[int]] = {}
            for i, k in enumerate(group_key):
                sec_groups.setdefault(k, []).append(i)
            docs_with_multi = [idxs for idxs in sec_groups.values() if len(idxs) >= 2]
            if not docs_with_multi:
                return {"ok": False,
                        "error": "单篇文档内每个小节都只有 1 块，无法构造同组正例；"
                                 "请入库更多文档，或用 calibrate_scores 手工指定正负例"}

        rng = random.Random(20260101)  # 固定种子，标定结果可复现
        pos: List[float] = []
        neg: List[float] = []

        # 正例：同组（多文档=同文档；单文档=同小节）内随机块对
        attempts = 0
        while len(pos) < sample_pairs and attempts < sample_pairs * 20:
            attempts += 1
            if not docs_with_multi:
                break
            idxs = rng.choice(docs_with_multi)
            a, b = rng.sample(idxs, 2)
            pos.append(_cosine(embs[a], embs[b]))

        # 负例：跨组随机块对
        all_idx = list(range(len(embs)))
        attempts = 0
        while len(neg) < sample_pairs and attempts < sample_pairs * 20:
            attempts += 1
            a, b = rng.sample(all_idx, 2)
            if group_key[a] == group_key[b]:
                continue  # 同组的留给正例，负例只要跨组的
            neg.append(_cosine(embs[a], embs[b]))

        if not pos or not neg:
            return {"ok": False,
                    "error": f"取样不足（正例 {len(pos)} / 负例 {len(neg)}）；"
                             f"语料太少，请入库更多文档后重试"}

        pos_sorted = sorted(pos)
        neg_sorted = sorted(neg)
        # 门槛取正例 P10：容忍同文档 10% 的正常波动，同时尽量高于负例地板
        p10 = pos_sorted[max(0, int(len(pos_sorted) * 0.10) - 1)]
        neg_max = neg_sorted[-1]
        neg_p90 = neg_sorted[min(len(neg_sorted) - 1, int(len(neg_sorted) * 0.90))]

        # 正例 P10 若低于负例 P90，说明两组重叠，取中点更稳
        suggested = p10 if p10 > neg_p90 else round((p10 + neg_p90) / 2, 4)
        suggested = round(min(max(suggested, 0.05), 0.90), 4)
        margin = round(p10 - neg_max, 4)

        result = {
            "ok": True,
            "embed_model": self.cfg.embed_model,
            "collection": name,
            "sample_basis": sample_basis,
            "samples": {"docs": len(groups), "chunks": len(embs),
                        "groups": len(set(group_key)),
                        "pos_pairs": len(pos), "neg_pairs": len(neg)},
            "pos_p10": round(p10, 4),
            "pos_min": round(pos_sorted[0], 4),
            "neg_max": round(neg_max, 4),
            "neg_p90": round(neg_p90, 4),
            "margin": margin,
            "suggested_min_score": suggested,
            "separable": margin > 0,
            "verdict": (
                "正负例可分离，门槛已按正例 P10 标定"
                if margin > 0 else (
                    # 单文档跨小节取样下 margin≤0 是预期结果，不代表模型有问题
                    "单篇文档内各章节主题相近，跨小节块对无法充当有效负例，"
                    "本次无法自动标定。请改用真实问句标定（calibrate_scores：文档内有答案的问题"
                    "作正例、无关问题作负例），或入库更多不同主题的文档后重试。"
                    if sample_basis == "cross_section" else
                    "⚠️ 正负例分数重叠（margin≤0）：调门槛无法根治，"
                    "需换 embedding 模型、增大 chunk_size 或检查文档是否主题过于集中"
                )
            ),
        }
        if sample_basis == "cross_section":
            # 单文档降级取样的可信度低于跨文档取样，必须显式告知，不能让人误以为是同等标定
            result["caveat"] = (
                "库内只有 1 篇文档，负例改用「跨小节块对」近似。"
                "同一篇报告的不同章节主题相近，噪音地板会被低估，标定门槛偏松；"
                "入库更多文档后重跑 --kb-calibrate 可获得更可靠的门槛。"
            )

        # 安全闸门：不可分离时**拒绝写入门槛**。
        # margin≤0 意味着「同组块对」并不比「跨组块对」更相似，此时算出的门槛
        # 往往高于真实问题的命中分数，写进去会让知识库静默返回 0 条（不报错、难排查）。
        # 宁可保持原门槛不变，也不能标出一个让检索整体失效的值。
        if persist and margin > 0:
            self.save_min_score(suggested, collection_name=name)
            result["persisted"] = True
        elif persist:
            result["persisted"] = False
            result["not_persisted_reason"] = (
                "正负例不可分离，已保留原有门槛不做修改"
                f"（当前生效值 {self._min_score_for(name)}）"
            )
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
