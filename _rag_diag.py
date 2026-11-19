"""One-off diagnostic: query-chunk scores vs persisted threshold. Not part of the product."""
from rag_engine import RagEngine, RagConfig, _cosine

e = RagEngine(RagConfig(db_path="./agent_knowledge_base"))
coll = e._collection("agent_general_knowledge")
res = coll.get(include=["documents", "metadatas", "embeddings"])
docs = res["documents"]
metas = res["metadatas"]
embs = list(res["embeddings"])
PREFIX = "为这个句子生成表示以用于检索相关文章："
THR = 0.5097

cases = [
    ("端口", "HaloSync 用了哪些端口", "HaloSync", "端口"),
    ("在途", "在途数据量怎么计算", "HaloSync", "在途"),
    ("错误码", "同步失败的错误码有哪些", "HaloSync", "错误"),
    ("硬件", "部署 HaloSync 需要什么硬件", "HaloSync", "硬件"),
    ("一致", "一致性模式有哪几种", "HaloSync", "一致性"),
    ("RSG", "RSG 三层门控结论是什么", "SRM", "RSG"),
    ("十五维", "十五维状态向量加权总分是多少", "SRM", "加权"),
    ("早盘", "次日早盘条件单预案", "SRM", "早盘"),
    ("长目标", "请根据知识库回答 HaloSync 技术手册里端口清单和硬件规格，并说明出处", "HaloSync", "端口"),
    ("无关", "今天北京天气怎么样", None, None),
]


def rank_of(qv, needle_src, needle_kw):
    order = sorted(range(len(embs)), key=lambda i: _cosine(qv, embs[i]), reverse=True)
    for r, i in enumerate(order, 1):
        src = (metas[i] or {}).get("source_name", "")
        body = docs[i] or ""
        if needle_src in src and (needle_kw is None or needle_kw in body):
            return r, round(_cosine(qv, embs[i]), 4)
    return 99, 0.0


print(f"chunks={len(docs)} threshold={THR}")
for name, q, src, kw in cases:
    qv = e._embed_one(q)
    pv = e._embed_one(PREFIX + q)
    raw_scores = sorted((_cosine(qv, embs[i]), i) for i in range(len(embs)))
    top_s, top_i = raw_scores[-1]
    passed = sum(1 for s, _ in raw_scores if s >= THR)
    head = (docs[top_i] or "").split("\n")[0][:48]
    src_name = ((metas[top_i] or {}).get("source_name") or "")[:12]
    if src:
        rr, rs = rank_of(qv, src, kw)
        pr, ps = rank_of(pv, src, kw)
        hit = "HIT" if rr == 1 else f"rank{rr}"
    else:
        rr = pr = rs = ps = "-"
        hit = "neg"
    print(
        f"\n[{name}] {hit} top={top_s:.3f} pass>={THR}:{passed} src={src_name}"
    )
    print(f"  q: {q}")
    print(f"  top1: {head}")
    print(f"  keyword rank raw={rr}@{rs}  prefix={pr}@{ps}")
    top3 = sorted(raw_scores, reverse=True)[:3]
    for s, i in top3:
        h = (docs[i] or "").split("\n")[0][:40]
        print(f"    {s:.3f} {h}")
