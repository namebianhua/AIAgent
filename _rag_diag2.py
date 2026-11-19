"""Print scores of the chunks that should have won."""
from rag_engine import RagEngine, RagConfig, _cosine

e = RagEngine(RagConfig(db_path="./agent_knowledge_base"))
coll = e._collection("agent_general_knowledge")
res = coll.get(include=["documents", "metadatas", "embeddings"])
docs = list(res["documents"])
embs = list(res["embeddings"])

needles = ["端口清单", "硬件规格", "3.2 端口", "附录 A"]
qs = [
    "HaloSync 用了哪些端口",
    "部署 HaloSync 需要什么硬件",
    "端口清单",
    "硬件规格",
]
qvs = {q: e._embed_one(q) for q in qs}

for i, d in enumerate(docs):
    head = (d or "").split("\n")[0]
    if not any(n in (d or "") for n in needles):
        continue
    print(f"\n#{i} len={len(d)} {head}")
    print((d or "")[:220].replace("\n", " | "))
    for q, qv in qvs.items():
        print(f"  { _cosine(qv, embs[i]):.3f}  {q}")
