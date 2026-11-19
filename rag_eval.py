# -*- coding: utf-8 -*-
"""RAG 检索质量评测与失败模式诊断。

不再凭推断归因：先建带标注答案的评测集，量化 Recall@k / MRR，
再逐个失败案例打印实际命中的块，定位是哪一类问题（分块、向量、门槛、还是检索策略）。

用法：.venv\\Scripts\\python.exe rag_eval.py [--kb 库名] [--model 模型名]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag_engine import DEFAULT_EMBED_MODEL, RagConfig, RagEngine

# ---------------------------------------------------------------------------
# 评测集：每条含查询、期望命中的章节关键词（判定依据）、类别。
# 答案标注来自文档实际内容（块首【标题路径】即章节名），可客观判定命中与否。
# ---------------------------------------------------------------------------


@dataclass
class Case:
    query: str
    expect_section: str        # 期望命中的章节标识（出现在块的【标题路径】或正文中即算命中）
    category: str = "常规"     # 用于分组统计失败模式
    expect_terms: List[str] = field(default_factory=list)  # 正文必须包含的关键词（更严格判定）


CASES: List[Case] = [
    # ---- SRM 盘后报告：数值型查询 ----
    Case("2026年10月12日上证指数收盘点位和成交额是多少", "1. 数据获取与校验", "数值查询",
         ["3797.24"]),
    Case("十五维状态向量的加权总分和日内轨迹", "4. 十五维状态向量评分表", "数值查询",
         ["38.35", "35.5"]),
    Case("RSG三层门控R S G各自的结论是什么", "5. RSG", "结构化结论",
         ["下跌末段", "中性偏空"]),
    Case("反身性循环当前处于哪个阶段", "7. 反身性循环诊断", "概念查询"),
    Case("NLP叙事引擎输出的市场情绪是什么", "6. NLP 叙事引擎", "概念查询"),
    Case("宁德时代这笔仓位执行得怎么样，浮亏多少", "2. 上期配置复盘", "标的查询",
         ["宁德"]),
    Case("次日早盘的条件单预案有哪些", "13. 次日", "计划查询"),
    Case("新组合配置清单和上期有什么差异", "9. 新组合配置清单", "对比查询"),
    Case("双轨盘后结算的结果是什么", "10. 双轨盘后结算", "结果查询"),
    Case("模型自己最可能错在哪里", "14. 我最可能错", "元认知查询"),
    Case("板块配置怎么选标的", "8. 板块配置与标的精选", "策略查询"),
    Case("周期判定和参数迭代做了什么变更", "12. 周期判定", "变更查询"),
    Case("大盘中长期和本周的时间尺度判断", "3. 大盘三层时间尺度", "趋势查询"),
    Case("盘后复盘归因的主要结论", "11. 盘后复盘归因", "归因查询"),

    # ---- HaloSync 技术手册：典型技术文档查询 ----
    Case("HaloSync 部署需要开放哪些端口", "3.2 端口清单", "技术查询",
         ["端口"]),
    Case("部署 HaloSync 的硬件规格要求是什么", "3.1 硬件规格", "技术查询"),
    Case("同步延迟高怎么排查", "8 故障排查", "故障排查"),
    # 用文档中真实存在的错误码（E4201）。早期版本这里写 E4001 —— 该码在文档中
    # 根本不存在（实际只有 E4201/E4202/E4210/E4211/E4212/E4301），检索其实正确命中了
    # 「7 错误码」表（0.705），却因断言要求正文含 E4001 被判为未命中，属于评测集标注错误。
    Case("错误码 E4201 是什么意思", "7 错误码", "错误码查询", ["E4201"]),
    Case("E4211 表示什么原因", "7 错误码", "错误码查询", ["E4211"]),
    Case("在途数据量怎么计算", "4.1 在途数据量", "计算公式"),
    Case("重试等待时间的计算公式", "4.2 重试等待时间", "计算公式"),
    Case("软限和硬限的区别是什么", "6.1 软限与硬限", "概念对比"),
    Case("HaloSync 支持哪些一致性模式", "5 一致性模式", "功能查询"),
    Case("数据流路径是怎样的", "2.1 数据流路径", "架构查询"),
    Case("常见部署误区有哪些", "附录 A", "经验查询"),

    # ---- 无关查询：应当被门槛过滤（测假阳性）----
    Case("北京明天天气怎么样", "__NONE__", "噪音过滤"),
    Case("红烧肉的做法步骤", "__NONE__", "噪音过滤"),
    Case("如何安装 Python 的 pandas", "__NONE__", "噪音过滤"),
]

# ---------------------------------------------------------------------------
# 严格评测集：口语化改写，**刻意不含任何章节名与文档术语**。
#
# 为什么必须有这一组：上面的常规用例里查询词大量复用章节标题
# （如「数据流路径」「硬件规格」「端口清单」），而分块时已把标题写进块前缀，
# 等于拿标题去匹配标题，P@1 会虚高、掩盖真实检索能力。
# 这一组模拟真实用户：只描述需求、不知道文档怎么分章。
# 判定仍用客观依据（期望章节 + 正文关键词），不靠主观打分。
# ---------------------------------------------------------------------------

STRICT_CASES: List[Case] = [
    # SRM 报告：口语化、不含章节名
    Case("那天大盘最后收在多少点", "1. 数据获取与校验", "严格·口语化", ["3797.24"]),
    Case("给整个市场状态打个总分是多少", "4. 十五维状态向量评分表", "严格·口语化", ["35.5"]),
    Case("现在还能不能买新股票", "5. RSG", "严格·口语化", ["冻结"]),
    Case("那个电池股买完之后亏了多少", "2. 上期配置复盘", "严格·口语化", ["宁德"]),
    Case("明天开盘挂什么单", "13. 次日", "严格·口语化"),
    Case("这套模型自己承认哪里可能判断错", "14. 我最可能错", "严格·口语化"),
    Case("买哪几个行业比较好", "8. 板块配置与标的精选", "严格·口语化"),
    Case("昨天那笔操作事后看做对了吗", "11. 盘后复盘归因", "严格·口语化"),
    Case("接下来一两周大盘大概怎么走", "3. 大盘三层时间尺度", "严格·口语化"),
    Case("市场情绪现在偏乐观还是偏悲观", "6. NLP 叙事引擎", "严格·口语化"),

    # HaloSync 手册：口语化、不含章节名与产品参数名
    Case("装这个软件机器配置要多高", "3.1 硬件规格", "严格·口语化", ["CPU"]),
    Case("防火墙要放行哪些号", "3.2 端口清单", "严格·口语化", ["8443"]),
    Case("任务一直不动了怎么办", "8 故障排查", "严格·口语化"),
    Case("源端密码过期会报哪个号", "7 错误码", "严格·错误码", ["E4201"]),
    Case("最多能跑多快，加机器有用吗", "3.1 硬件规格", "严格·口语化", ["12,000"]),
    Case("数据从源头到目标中间经过哪些环节", "2.1 数据流路径", "严格·口语化"),
    Case("超过限制会丢数据吗", "6.1 软限与硬限", "严格·口语化"),
    Case("这个平台是用来干什么的", "1 产品概述", "严格·口语化", ["HaloSync"]),

    # 跨文档混淆：两个主题完全不同的库内文档，测是否串味
    Case("上证指数跌了多少", "1. 数据获取与校验", "严格·跨文档", ["0.43"]),
    Case("一致性模式有几种可选", "5 一致性模式", "严格·跨文档"),
]


def _hit(h: Dict[str, Any], case: Case) -> bool:
    """判定一个检索结果是否命中期望：章节路径 + 正文关键词都要对得上。"""
    content = h.get("content") or ""
    if case.expect_section != "__NONE__" and case.expect_section not in content:
        # 章节标识不在块里，再宽松地看正文关键词
        if not case.expect_terms or not all(t in content for t in case.expect_terms):
            return False
    if case.expect_terms and not all(t in content for t in case.expect_terms):
        return False
    return True


def evaluate(engine: RagEngine, collection: str, top_k: int = 3,
             verbose_fail: bool = True,
             cases: Optional[List[Case]] = None,
             hybrid: bool = False) -> Dict[str, Any]:
    """跑评测集，返回总体指标 + 分类别指标 + 失败案例明细。

    :param cases: 指定用例集；None 时用常规集 CASES。传 STRICT_CASES 可测
                  「不含章节术语」的真实检索能力，避免用标题匹配标题造成虚高。
    :param hybrid: True 时走 search_hybrid（查询改写 + BM25 + RRF 融合），
                   False 时走纯向量 search。两者共用同一套用例与判定，可直接对比。
    """
    case_set = CASES if cases is None else cases
    threshold = engine._min_score_for(collection)
    rows = []
    for case in case_set:
        if hybrid:
            hits = engine.search_hybrid(case.query, top_k=top_k, collection_name=collection)
        else:
            hits = engine.search(case.query, top_k=top_k, collection_name=collection)
        if case.expect_section == "__NONE__":
            # 噪音查询：命中 0 条才算正确
            ok = len(hits) == 0
            rank = 0 if ok else 1
        else:
            rank = 0
            for i, h in enumerate(hits, 1):
                if _hit(h, case):
                    rank = i
                    break
            ok = rank == 1
        rows.append({
            "case": case, "hits": hits, "rank": rank, "top1_ok": ok,
            "recall": rank > 0,
            "top_score": hits[0]["relevance_score"] if hits else 0.0,
        })

    scored = [r for r in rows if r["case"].expect_section != "__NONE__"]
    noise = [r for r in rows if r["case"].expect_section == "__NONE__"]

    n = len(scored)
    p1 = sum(1 for r in scored if r["top1_ok"])
    recall_at_k = sum(1 for r in scored if r["recall"])
    mrr = sum(1.0 / r["rank"] for r in scored if r["rank"] > 0) / n if n else 0.0
    noise_ok = sum(1 for r in noise if r["top1_ok"])

    by_cat: Dict[str, List[Any]] = {}
    for r in scored:
        by_cat.setdefault(r["case"].category, []).append(r)

    cat_stats = {
        c: {
            "n": len(v),
            "top1": sum(1 for x in v if x["top1_ok"]),
            "recall": sum(1 for x in v if x["recall"]),
        }
        for c, v in by_cat.items()
    }

    # 失败案例明细
    fails = []
    for r in rows:
        if r["top1_ok"]:
            continue
        item = {
            "query": r["case"].query,
            "category": r["case"].category,
            "expect": r["case"].expect_section,
            "rank": r["rank"],
            "n_hits": len(r["hits"]),
            "top_hits": [
                {"score": h["relevance_score"],
                 "head": " ".join((h["content"] or "").split())[:110]}
                for h in r["hits"][:2]
            ],
        }
        fails.append(item)

    return {
        "threshold": threshold,
        "top_k": top_k,
        "n_scored": n,
        "precision_at_1": round(p1 / n, 3) if n else 0.0,
        "recall_at_k": round(recall_at_k / n, 3) if n else 0.0,
        "mrr": round(mrr, 3),
        "noise_filtered": f"{noise_ok}/{len(noise)}",
        "by_category": cat_stats,
        "failures": fails if verbose_fail else [],
    }


def _print_report(title: str, results: Dict[int, Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    """打印一组用例集的指标、分类别表现与失败明细，返回基准 top_k 的结果。"""
    print("=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'top_k':>6} | {'P@1':>7} | {'Recall@k':>9} | {'MRR':>7} | 噪音过滤")
    for k, r in results.items():
        print(f"{k:>6} | {r['precision_at_1']:>7.1%} | {r['recall_at_k']:>8.1%} | "
              f"{r['mrr']:>7.3f} | {r['noise_filtered']}")

    base = results[top_k]
    print()
    print(f"分类别表现（top_k={top_k}）")
    for c, s in sorted(base["by_category"].items(),
                       key=lambda x: x[1]["top1"] / max(1, x[1]["n"])):
        rate = s["top1"] / max(1, s["n"])
        flag = "✓" if rate >= 0.8 else ("△" if rate >= 0.5 else "✗")
        print(f"  {flag} {c:12s} P@1 {s['top1']}/{s['n']} = {rate:>5.1%}"
              f"   Recall@k {s['recall']}/{s['n']}")

    if base["failures"]:
        print()
        print(f"失败案例明细（top_k={top_k}，共 {len(base['failures'])} 条）")
        for f in base["failures"]:
            print(f"\n✗ [{f['category']}] {f['query']}")
            print(f"   期望：{f['expect']}｜实际命中位次：{f['rank'] or '未命中'}"
                  f"｜返回 {f['n_hits']} 条")
            for i, h in enumerate(f["top_hits"], 1):
                print(f"   top{i} ({h['score']:.3f}): {h['head']}")
    else:
        print("\n（无失败案例）")
    return base


def _run_set(engine: RagEngine, kb: str, cases: List[Case], top_k: int,
             hybrid: bool = False) -> Dict[int, Dict[str, Any]]:
    return {k: evaluate(engine, kb, top_k=k, verbose_fail=(k == top_k),
                        cases=cases, hybrid=hybrid)
            for k in (1, 3, 5)}


def _print_compare(vec: Dict[int, Dict[str, Any]], hyb: Dict[int, Dict[str, Any]],
                   cases: List[Case], engine: RagEngine, kb: str, top_k: int) -> None:
    """逐条对比纯向量与混合检索的命中位次，直接暴露每个通道的贡献。"""
    print("=" * 76)
    print("逐条对比：纯向量 → 混合检索（✓=top1 命中）")
    print("=" * 76)
    v_hits = h_hits = 0
    improved = regressed = 0
    for case in cases:
        if case.expect_section == "__NONE__":
            continue
        v = engine.search(case.query, top_k=top_k, collection_name=kb)
        h = engine.search_hybrid(case.query, top_k=top_k, collection_name=kb)
        rv = next((i for i, x in enumerate(v, 1) if _hit(x, case)), 0)
        rh = next((i for i, x in enumerate(h, 1) if _hit(x, case)), 0)
        if rv == 1:
            v_hits += 1
        if rh == 1:
            h_hits += 1
        tag = ""
        if rv != 1 and rh == 1:
            improved += 1
            tag = "  ← 修复"
        elif rv == 1 and rh != 1:
            regressed += 1
            tag = "  ← 退化"
        fv = f"top{rv}" if rv else "✗"
        fh = f"top{rh}" if rh else "✗"
        print(f"  {fv:>5s} → {fh:<5s}  {case.query[:34]}{tag}")
    print("-" * 76)
    print(f"  top1 命中：纯向量 {v_hits} → 混合 {h_hits}｜"
          f"修复 {improved} 条，退化 {regressed} 条")
    if regressed:
        print("  ⚠️ 存在退化案例，混合检索不是纯增益，需检查融合权重或门槛")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 检索质量评测")
    ap.add_argument("--kb", default="agent_general_knowledge")
    ap.add_argument("--model", default="")
    ap.add_argument("--db", default="agent_knowledge_base")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict-only", action="store_true",
                    help="只跑严格集（口语化、不含章节术语）")
    ap.add_argument("--hybrid", action="store_true",
                    help="用混合检索（查询改写 + BM25 + RRF）代替纯向量检索")
    ap.add_argument("--compare", action="store_true",
                    help="严格集下对比纯向量与混合检索，并逐条列出修复/退化案例")
    ap.add_argument("--fail-under", type=float, default=0.0,
                    help="严格集 P@1 低于此值时退出码为 1（回归门禁用，0=不启用）")
    args = ap.parse_args()

    engine = RagEngine(RagConfig(
        db_path=args.db,
        embed_model=args.model or DEFAULT_EMBED_MODEL,
    ))

    print(f"评测库：{args.kb}｜模型：{engine.cfg.embed_model}｜"
          f"门槛：{engine._min_score_for(args.kb)}｜top_k={args.top_k}｜"
          f"模式：{'混合检索' if args.hybrid else '纯向量'}\n")

    if args.compare:
        _print_compare(
            _run_set(engine, args.kb, STRICT_CASES, args.top_k, hybrid=False),
            _run_set(engine, args.kb, STRICT_CASES, args.top_k, hybrid=True),
            STRICT_CASES, engine, args.kb, args.top_k)
        return 0

    strict_results = _run_set(engine, args.kb, STRICT_CASES, args.top_k, hybrid=args.hybrid)
    normal_results = (None if args.strict_only
                      else _run_set(engine, args.kb, CASES, args.top_k, hybrid=args.hybrid))

    if args.json:
        payload = {"strict": {str(k): {kk: vv for kk, vv in v.items() if kk != "failures"}
                              for k, v in strict_results.items()}}
        if normal_results is not None:
            payload["normal"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "failures"}
                                 for k, v in normal_results.items()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if normal_results is not None:
        _print_report("常规集：查询含章节术语（偏乐观，仅作回归对照）",
                      normal_results, args.top_k)
        print("\n")

    # 严格集是真实能力的度量，放在后面压轴，并以它作为最终结论依据
    strict_base = _print_report(
        "严格集：口语化改写，不含章节名与文档术语（真实检索能力）",
        strict_results, args.top_k)

    p1 = strict_base["precision_at_1"]
    print()
    print("=" * 76)
    print("结论（以严格集为准）")
    print("=" * 76)
    print(f"  严格集 P@1 = {p1:.1%}｜Recall@{args.top_k} = {strict_base['recall_at_k']:.1%}"
          f"｜MRR = {strict_base['mrr']:.3f}｜噪音过滤 {strict_base['noise_filtered']}")
    if normal_results is not None:
        n_p1 = normal_results[args.top_k]["precision_at_1"]
        gap = n_p1 - p1
        print(f"  常规集 P@1 = {n_p1:.1%}（含章节术语，比严格集高 {gap:+.1%}"
              f"{'，差距说明指标虚高来自标题匹配' if gap > 0.05 else '，两组接近'}）")

    if args.fail_under > 0 and p1 < args.fail_under:
        print(f"\n✗ 未达回归门禁：严格集 P@1 {p1:.1%} < {args.fail_under:.1%}")
        return 1
    print(f"\n{'✓ 通过' if p1 >= 0.9 else '△ 存在失败案例，见上方明细'}"
          f"（严格集 P@1 {p1:.1%}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
