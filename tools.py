from __future__ import annotations

import html
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

MAX_OUTPUT_CHARS = 4000


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None


_BLOCK_RE = re.compile(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', re.S | re.I)
_BING_TITLE_RE = re.compile(r'<h2>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_BING_SNIPPET_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.S | re.I)
_LINK_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.S | re.I,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S | re.I
)
_TAG_RE = re.compile(r"<[^>]+>")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _search_bing(query: str, max_results: int) -> list:
    resp = requests.get(
        "https://cn.bing.com/search",
        params={"q": query, "setlang": "zh-CN"},
        headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"},
        timeout=12,
    )
    resp.raise_for_status()

    results = []
    for block in _BLOCK_RE.findall(resp.text):
        m = _BING_TITLE_RE.search(block)
        if not m:
            continue
        url, title = m.group(1), _clean(m.group(2))
        snip = _BING_SNIPPET_RE.search(block)
        results.append(
            {
                "title": title,
                "url": html.unescape(url),
                "snippet": _clean(snip.group(1)) if snip else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def _unwrap_ddg_link(href: str) -> str:
    """DuckDuckGo 的结果链接是跳转链接，真实地址在 uddg 参数里。"""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc:
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def _search_ddg(query: str, max_results: int) -> list:
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": UA},
        timeout=10,
    )
    resp.raise_for_status()

    links = _LINK_RE.findall(resp.text)
    snippets = [_clean(s) for s in _SNIPPET_RE.findall(resp.text)]
    results = []
    for i, (href, title_html) in enumerate(links[:max_results]):
        results.append(
            {
                "title": _clean(title_html),
                "url": _unwrap_ddg_link(href),
                "snippet": snippets[i] if i < len(snippets) else "",
            }
        )
    return results


def search_web(query: str, max_results: int = 3) -> ToolResult:
    """轻量网页搜索：Bing 优先（国内可直连），DuckDuckGo 兜底。"""
    errors = []
    for name, fn in (("bing", _search_bing), ("duckduckgo", _search_ddg)):
        try:
            results = fn(query, max_results)
            if results:
                return ToolResult(
                    ok=True,
                    data={"query": query, "engine": name, "results": results},
                )
            errors.append(f"{name}: 无结果")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__} {str(e)[:60]}")

    return ToolResult(ok=False, error="搜索失败（" + "; ".join(errors) + ")")


def run_python(code: str) -> ToolResult:
    """在子进程里执行 Python 代码片段。

    用 sys.executable 而不是写死 "python"，保证跑在与当前项目相同的解释器/依赖里。
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n...[输出过长已截断]"
        if result.returncode != 0 and not output:
            output = f"进程退出码 {result.returncode}，无输出"
        return ToolResult(ok=result.returncode == 0, data=output)
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error="代码执行超时（15 秒）")
    except Exception as e:
        return ToolResult(ok=False, error=str(e))


AVAILABLE_TOOLS = {
    "search": search_web,
    "run_python": run_python,
}
