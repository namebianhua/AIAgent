"""T1 内置工具集：文件系统 + Shell + Python + 网页搜索。

所有文件路径都经 sandbox.resolve() 锁在工作区内；所有写入都受只读模式约束。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine import ToolRegistry
from sandbox import Sandbox, SandboxError
from tools import search_web as _search_web

MAX_OUTPUT = 8000
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".agent", ".idea"}
_BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz",
    ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".pyc", ".mp3", ".mp4",
    ".woff", ".woff2", ".ttf", ".db", ".sqlite",
}


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[内容过长，已截断，原始长度 {len(text)} 字符]"


def _looks_binary(path: Path) -> bool:
    if path.suffix.lower() in _BINARY_EXT:
        return True
    try:
        with path.open("rb") as f:
            return b"\x00" in f.read(1024)
    except Exception:
        return True


def _rel(sandbox: Sandbox, p: Path) -> str:
    """显示用路径：工作区内显示相对路径，工作区外（电脑作用域）显示绝对路径。"""
    try:
        return p.relative_to(sandbox.root).as_posix()
    except ValueError:
        return str(p)


def _skip_parts(sandbox: Sandbox, p: Path) -> bool:
    """跳过 .git/node_modules 等目录；工作区外的路径看末尾几段。"""
    try:
        parts = p.relative_to(sandbox.root).parts
    except ValueError:
        parts = p.parts[-4:]
    return any(part in _SKIP_DIRS for part in parts)


def build_registry(sandbox: Sandbox) -> ToolRegistry:
    reg = ToolRegistry()
    reg.audit = lambda ev: sandbox.audit({**ev, "kind": "tool_call"})

    # ---------------- 文件系统 ----------------
    @reg.register()
    def list_dir(path: str = ".", recursive: bool = False, max_entries: int = 200) -> Dict[str, Any]:
        """列出工作区目录内容。

        path: 相对工作区的路径，默认当前目录
        recursive: 是否递归列出子目录
        max_entries: 最多返回多少条
        """
        target = sandbox.resolve(path)
        if not target.is_dir():
            return {"ok": False, "output": "", "error": f"不是目录：{path}"}

        items: List[str] = []
        walker = target.rglob("*") if recursive else target.iterdir()
        for p in walker:
            rel = _rel(sandbox, p)
            if _skip_parts(sandbox, p):
                continue
            if p.is_dir():
                items.append(f"[DIR ] {rel}/")
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                items.append(f"[FILE] {rel}  ({size} B)")
            if len(items) >= max_entries:
                items.append(f"...[超过 {max_entries} 条，已截断]")
                break

        if not items:
            return {"ok": True, "output": "（空目录）", "error": None}
        return {"ok": True, "output": "\n".join(items), "error": None}

    @reg.register()
    def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> Dict[str, Any]:
        """读取工作区内文本文件的内容（带行号）。

        path: 相对工作区的文件路径
        start_line: 从第几行开始（1 起）
        max_lines: 最多读多少行
        """
        target = sandbox.resolve(path)
        if not target.is_file():
            return {"ok": False, "output": "", "error": f"文件不存在：{path}"}
        if _looks_binary(target):
            return {"ok": False, "output": "", "error": f"二进制文件不支持读取：{path}"}

        start = max(1, int(start_line))
        limit = max(1, min(int(max_lines), 2000))
        try:
            with target.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}

        total = len(lines)
        chunk = lines[start - 1: start - 1 + limit]
        numbered = [f"{start + i:>5} | {ln.rstrip(chr(10))}" for i, ln in enumerate(chunk)]
        body = "\n".join(numbered)
        if start - 1 + limit < total:
            body += f"\n...[共 {total} 行，仅显示 {start}-{start + len(chunk) - 1} 行]"
        return {"ok": True, "output": _truncate(body), "error": None}

    @reg.register(dangerous=True)
    def write_file(path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """写入或追加文本文件（工作区内）。

        path: 相对工作区的文件路径，父目录会自动创建
        content: 要写入的完整文本
        append: True 时追加到文件末尾，False 覆盖
        """
        sandbox.check_writable()
        target = sandbox.resolve(path, for_write=True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with target.open(mode, encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}
        action = "追加" if append else "写入"
        return {
            "ok": True,
            "output": f"已{action} {_rel(sandbox, target)}（{len(content)} 字符）",
            "error": None,
        }

    @reg.register()
    def make_dir(path: str) -> Dict[str, Any]:
        """创建目录（含多级父目录）。path 为相对工作区的路径或绝对路径。"""
        sandbox.check_writable()
        target = sandbox.resolve(path, for_write=True)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "output": f"目录已就绪：{path}", "error": None}

    @reg.register()
    def move_file(src: str, dst: str) -> Dict[str, Any]:
        """移动或重命名文件/目录（源和目标必须在同一作用域内）。"""
        sandbox.check_writable()
        s = sandbox.resolve(src)
        d = sandbox.resolve(dst, for_write=True)
        if not s.exists():
            return {"ok": False, "output": "", "error": f"源不存在：{src}"}
        try:
            d.parent.mkdir(parents=True, exist_ok=True)
            s.replace(d)
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "output": f"已移动：{src} -> {dst}", "error": None}

    @reg.register(dangerous=True, require_confirm=True)
    def delete_file(path: str, recursive: bool = False) -> Dict[str, Any]:
        """删除工作区内的文件或目录（高危，需要人工确认）。

        path: 相对工作区的路径
        recursive: 删除目录时必须为 True
        """
        import shutil

        sandbox.check_writable()
        target = sandbox.resolve(path, for_write=True)
        if target == sandbox.root:
            return {"ok": False, "output": "", "error": "拒绝删除工作区根目录"}
        if not target.exists():
            return {"ok": False, "output": "", "error": f"不存在：{path}"}
        try:
            if target.is_dir():
                if not recursive:
                    return {"ok": False, "output": "", "error": "删除目录需显式传 recursive=true"}
                sandbox.check_recursive_delete(target)  # 工作区外禁止递归删除
                shutil.rmtree(target)
            else:
                target.unlink()
        except SandboxError as e:
            return {"ok": False, "output": "", "error": str(e)}
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "output": f"已删除：{path}", "error": None}

    @reg.register()
    def search_text(
        pattern: str,
        path: str = ".",
        max_results: int = 30,
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """在工作区文本文件里按正则搜索内容，返回 "文件:行号: 内容"。

        pattern: 正则表达式
        path: 搜索起始目录（相对工作区）
        max_results: 最多返回多少条命中
        """
        target = sandbox.resolve(path)
        if not target.exists():
            return {"ok": False, "output": "", "error": f"路径不存在：{path}"}
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return {"ok": False, "output": "", "error": f"正则非法：{e}"}

        hits: List[str] = []
        files = [target] if target.is_file() else target.rglob("*")
        for p in files:
            if not p.is_file() or _looks_binary(p):
                continue
            if _skip_parts(sandbox, p):
                continue
            if p.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for no, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = _rel(sandbox, p)
                            hits.append(f"{rel}:{no}: {line.strip()[:200]}")
                            if len(hits) >= max_results:
                                return {"ok": True, "output": _truncate("\n".join(hits)), "error": None}
            except Exception:
                continue

        if not hits:
            return {"ok": True, "output": "（无匹配）", "error": None}
        return {"ok": True, "output": _truncate("\n".join(hits)), "error": None}

    # ---------------- 命令 ----------------
    @reg.register(dangerous=True, require_confirm=True)
    def run_shell(cmd: str, timeout: int = 30, _cancel=None) -> Dict[str, Any]:
        """在工作区内执行单条命令（白名单限制，不支持管道/链式/重定向）。

        cmd: 例如 "python -V"、"git status"、"ls -la"
        timeout: 超时秒数（1-600）
        """
        result = sandbox.safe_run(cmd, timeout=timeout, _cancel=_cancel)
        if result.get("output"):
            result["output"] = _truncate(result["output"])
        return result

    @reg.register(dangerous=True, require_confirm=True)
    def run_python(code: str, timeout: int = 15, _cancel=None) -> Dict[str, Any]:
        """在子进程中执行 Python 代码片段（工作目录=工作区，默认超时 15 秒）。

        code: 要执行的 Python 代码
        """
        timeout = max(1, min(int(timeout), 120))
        if _cancel is not None and getattr(_cancel, "is_cancelled", False):
            return {"ok": False, "output": "", "error": f"已取消：{_cancel.reason}",
                    "_cancelled": True}

        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=str(sandbox.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
            )
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}

        interval = 0.2
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if _cancel is not None and getattr(_cancel, "is_cancelled", False):
                    Sandbox._kill_tree(proc)
                    return {"ok": False, "output": "", "error": f"已取消：{_cancel.reason}",
                            "_cancelled": True}
                if time.monotonic() >= deadline:
                    Sandbox._kill_tree(proc)
                    return {"ok": False, "output": "",
                            "error": f"执行超时（{timeout}s），已终止", "_killed": True}
                if _cancel is not None:
                    _cancel.wait(interval)
                else:
                    time.sleep(interval)
            out, err = proc.communicate(timeout=5)
        except Exception as e:
            Sandbox._kill_tree(proc)
            return {"ok": False, "output": "", "error": f"已取消（{type(e).__name__}）",
                    "_cancelled": True}

        output = ((out or "") + (err or "")).strip()
        return {
            "ok": proc.returncode == 0,
            "output": _truncate(output) if output else "（无输出）",
            "error": None if proc.returncode == 0 else f"退出码 {proc.returncode}",
        }

    # ---------------- 网络 ----------------
    @reg.register()
    def search(query: str, max_results: int = 3) -> Dict[str, Any]:
        """联网搜索（Bing 优先），用于获取工作区之外的最新信息。

        query: 搜索关键词
        max_results: 返回条数（1-10）
        """
        result = _search_web(query, max_results=max(1, min(int(max_results), 10)))
        if not result.ok:
            return {"ok": False, "output": "", "error": result.error}
        data = result.data or {}
        lines = [f"引擎：{data.get('engine', '-')}"]
        for i, item in enumerate(data.get("results", []), 1):
            lines.append(f"{i}. {item['title']}\n   {item['url']}\n   {item['snippet']}")
        return {"ok": True, "output": _truncate("\n".join(lines)), "error": None}

    # 供调试/自检使用
    reg.sandbox = sandbox  # type: ignore[attr-defined]
    return reg


def describe(reg: ToolRegistry) -> str:
    return json.dumps(reg.list_tools(), ensure_ascii=False, indent=2)
