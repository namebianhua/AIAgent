# -*- coding: utf-8 -*-
"""search_hybrid 的 A/B 验证：纯向量 / +BM25 / +查询改写，看各自贡献。只读。"""
from rag_engine import RagEngine, RagConfig

e = RagEngine(RagConfig(db_path="agent_knowledge_base", embed_model="bge-m3"))
COLL = "agent_general_knowledge"
TH = e._min_score_for(COLL)

FAILS = [
    ("明天开盘挂什么单", "13. 次日"),
    ("买哪几个行业比较好", "8. 板块配置与标的精选"),
    ("昨天那笔操作事后看做对了吗", "11. 盘后复盘归因"),
    ("市场情绪现在偏乐观还是偏悲观", "6. NLP 叙事引擎"),
    ("防火墙要放行哪些号", "3.2 端口清单"),
    ("最多能跑多快，加机器有用吗", "3.1 硬件规格"),
]
# 词法通道专项：含精确标识符的查询
LEX = [
    ("E4201 是什么错误", "7 错误码"),
    ("8443 端口是干什么的", "3.2 端口清单"),
    ("batch_size 默认值多少", "4 核心配置参数"),
]


def rank(hits, expect):
    for i, h in enumerate(hits, 1):
        if expect in (h["content"] or ""):
            return i
    return 0


print(f"门槛 {TH}\n")
print("=" * 88)
print("A/B 对比：严格集 6 个失败案例")
print("=" * 88)
print(f"{'查询':26s} {'纯向量':>8s} {'+BM25':>8s} {'+改写':>8s}")
print("-" * 88)
tally = {"vec": 0, "bm25": 0, "exp": 0}
for q, expect in FAILS:
    h_vec = e.search(q, top_k=3, collection_name=COLL)
    h_lex = e.search_hybrid(q, top_k=3, collection_name=COLL, expand=False, use_lexical=True)
    h_exp = e.search_hybrid(q, top_k=3, collection_name=COLL, expand=True, use_lexical=True)
    r = [rank(h_vec, expect), rank(h_lex, expect), rank(h_exp, expect)]
    for k, v in zip(("vec", "bm25", "exp"), r):
        if v == 1:
            tally[k] += 1
    fmt = lambda x: (f"top{x}" if x else "✗")
    print(f"{q[:24]:26s} {fmt(r[0]):>8s} {fmt(r[1]):>8s} {fmt(r[2]):>8s}")

print("-" * 88)
print(f"{'top1 命中':26s} {tally['vec']:>6}/6 {tally['bm25']:>6}/6 {tally['exp']:>6}/6")

print()
print("=" * 88)
print("词法通道专项：精确标识符查询（BM25 的主场）")
print("=" * 88)
for q, expect in LEX:
    h_vec = e.search(q, top_k=3, collection_name=COLL)
    h_lex = e.search_hybrid(q, top_k=3, collection_name=COLL, expand=False, use_lexical=True)
    rv, rl = rank(h_vec, expect), rank(h_lex, expect)
    mb = h_lex[0].get("matched_by") if h_lex else "-"
    print(f"  {q:28s} 纯向量={'top%d' % rv if rv else '✗':>6s}  混合={'top%d' % rl if rl else '✗':>6s}  通道={mb}")

print()
print("=" * 88)
print("改写通道可观测性（看模型改写出了什么）")
print("=" * 88)
h = e.search_hybrid("最多能跑多快，加机器有用吗", top_k=2, collection_name=COLL,
                    expand=True, use_lexical=True)
if h:
    print("  原查询：最多能跑多快，加机器有用吗")
    for i, v in enumerate(h[0].get("variants_used") or [], 1):
        print(f"    变体{i}: {v}")
    print(f"  命中来源：{h[0].get('source')}｜matched_by={h[0].get('matched_by')}"
          f"｜vec={h[0].get('relevance_score')}｜rrf={h[0].get('rrf_score')}")
    print(f"  出处是否完整：{'✓' if h[0].get('source') else '✗ 缺失'}")
