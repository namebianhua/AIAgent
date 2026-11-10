"""小红书选择器自检（诊断用）。

小红书前端改版频繁，工具突然取不到数据时先跑这个：
    python xhs_probe.py --keyword 咖啡

它会报告每个候选选择器的命中数，以及页面上真实出现的 class 名，
用于快速定位是哪一层失效（卡片 / 标题 / 作者 / 详情 / 评论）。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import xhs_web as X

SEARCH_CANDIDATES = [
    "section.note-item",
    "div.note-item",
    'section[class*="note-item"]',
    'div[class*="note-item"]',
    'a[href*="/explore/"]',
    'div[class*="feeds-page"] section',
    ".note-list > *",
]

FIELD_CANDIDATES = {
    "title": ["a.title span", ".footer .title span", ".title", 'span[class*="title"]'],
    "author": [".author .name", ".author-wrapper .name", ".name", 'span[class*="name"]'],
    "likes": [".like-wrapper .count", ".count", 'span[class*="like"] .count'],
}

DETAIL_CANDIDATES = [
    "#detail-title",
    "#detail-desc",
    ".note-content",
    ".desc",
    '[class*="note-text"]',
    ".comment-item",
    '[class*="comment-item"]',
]


def _count(page, sel: str) -> int:
    try:
        return page.locator(sel).count()
    except Exception:
        return -1


def probe_search(page, keyword: str) -> Dict[str, Any]:
    from urllib.parse import quote

    url = X.XHS_SEARCH_URL.format(kw=quote(keyword))
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(6000)

    report: Dict[str, Any] = {
        "url": page.url,
        "cards": {s: _count(page, s) for s in SEARCH_CANDIDATES},
        "fields": {},
        "classes": [],
        "first_note": None,
    }

    # 字段级命中：以第一个 note-item 为样本
    for field, sels in FIELD_CANDIDATES.items():
        hits = {}
        for s in sels:
            try:
                el = page.locator(s).first
                txt = el.inner_text(timeout=1500).strip() if el.count() else ""
            except Exception:
                txt = ""
            hits[s] = {"count": _count(page, s), "sample": txt[:40]}
        report["fields"][field] = hits

    # 页面上真实出现的 class 关键词，便于人工比对
    try:
        classes = page.evaluate(
            """() => {
              const s = new Set();
              document.querySelectorAll('[class]').forEach(el => {
                const c = el.className;
                if (typeof c === 'string') c.split(/\\s+/).forEach(x => x && s.add(x));
              });
              return Array.from(s).filter(x => /note|feed|title|author|like|desc|comment/i.test(x)).slice(0, 60);
            }"""
        )
        report["classes"] = classes
    except Exception:
        pass

    notes = page.evaluate(X.EXTRACT_NOTES_JS) or []
    report["extracted"] = len(notes)
    if notes:
        report["first_note"] = notes[0]
    return report


def probe_detail(page, url: str) -> Dict[str, Any]:
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(5000)
    report: Dict[str, Any] = {
        "url": page.url,
        "selectors": {s: _count(page, s) for s in DETAIL_CANDIDATES},
    }
    try:
        data = page.evaluate(X.EXTRACT_DETAIL_JS) or {}
        report["extracted"] = {
            k: (v[:80] if isinstance(v, str) else (len(v) if isinstance(v, list) else v))
            for k, v in data.items()
        }
    except Exception as e:
        report["extracted_error"] = f"{type(e).__name__}: {e}"
    return report


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="小红书选择器自检")
    ap.add_argument("--keyword", default="咖啡", help="用于探测搜索页的关键词")
    ap.add_argument("--profile", default=".agent/browser/xhs", help="登录态目录")
    ap.add_argument(
        "--headless", action="store_true", help="无头模式（默认有头便于观察）"
    )
    args = ap.parse_args(argv)

    session = X.XhsSession(
        X.XhsConfig(profile_dir=args.profile, headless=args.headless)
    )
    try:
        session.start(offscreen=args.headless)
        if not session.is_logged_in(nav=True):
            print("未登录。先运行：python main.py --xhs-login", file=sys.stderr)
            return 2

        print("=== 搜索页探针 ===")
        rep = probe_search(session.page, args.keyword)
        print(json.dumps(rep, ensure_ascii=False, indent=2))

        if rep.get("first_note", {}).get("url"):
            print("\n=== 详情页探针 ===")
            det = probe_detail(session.page, rep["first_note"]["url"])
            print(json.dumps(det, ensure_ascii=False, indent=2))
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
