"""电脑操控工具集：把 agent 的能力从「工作区沙箱」扩展到整台电脑。

设计决策（为什么这样而不是那样）：

1. **零新增依赖**：本机 pip 的 CDN 不稳（历史教训），所以截图走 GDI BitBlt、
   键鼠走 keybd_event/mouse_event、剪贴板走 Win32 API——全部用 ctypes 手写，
   PNG 编码只用标准库 zlib。pyautogui/Pillow 一个都不装。
2. **工具名分两组前缀**：sys_*（只读查询：系统信息/磁盘/进程/窗口/剪贴板/截屏/全盘找文件）
   与 ui_*（真实动作：鼠标/键盘/输入/聚焦窗口/打开程序/结束进程）。
   这样默认确认策略规则表只需要两条 glob（sys_* 放行、ui_* 询问）就能覆盖全部。
3. **动作工具全部 require_confirm**：真实桌面上点一下是不可撤销的，必须经过
   确认策略（用户可以在设置页为 ui_* 配 allow 或依赖 3 次批准自动晋升）。
4. **非 Windows 平台优雅降级**：ctypes.windll 只在 Windows 存在，其它平台
   返回明确错误而不是崩溃；sys_info/sys_find_file 天然跨平台。

只有在 Sandbox(scope="computer") 时才注册这批工具（见 runner.execute）。
"""
from __future__ import annotations

import ctypes
import fnmatch
import json
import os
import platform
import shutil
import struct
import subprocess
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine import ToolRegistry
from sandbox import Sandbox, SandboxError

_IS_WINDOWS = os.name == "nt"


# =====================================================================
# Windows API 基础（全部惰性取，避免非 Windows 平台 import 崩）
# =====================================================================
def _user32():
    if not _IS_WINDOWS:
        raise OSError("仅支持 Windows（当前平台没有 user32）")
    return ctypes.windll.user32


def _kernel32():
    if not _IS_WINDOWS:
        raise OSError("仅支持 Windows（当前平台没有 kernel32）")
    return ctypes.windll.kernel32


def _gdi32():
    if not _IS_WINDOWS:
        raise OSError("仅支持 Windows（当前平台没有 gdi32）")
    return ctypes.windll.gdi32


# =====================================================================
# 截图：GDI BitBlt 抓屏 + 标准库 zlib 写 PNG
# =====================================================================
def _capture_screen() -> Tuple[int, int, List[bytes]]:
    """抓主屏，返回 (宽, 高, 自上而下的 RGB 行列表)。"""
    user32, gdi32 = _user32(), _gdi32()
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    if width <= 0 or height <= 0:
        raise OSError("无法获取屏幕尺寸")

    hwnd = user32.GetDesktopWindow()
    hdc = user32.GetDC(hwnd)
    memdc = gdi32.CreateCompatibleDC(hdc)
    bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(memdc, bitmap)
    try:
        if not gdi32.BitBlt(memdc, 0, 0, width, height, hdc, 0, 0, 0x00CC0020):  # SRCCOPY
            raise OSError("BitBlt 失败")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height  # 负值 = 自上而下，省掉手动翻转
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        buf = ctypes.create_string_buffer(width * height * 4)
        if not gdi32.GetDIBits(memdc, bitmap, 0, height, buf, ctypes.byref(bmi), 0):
            raise OSError("GetDIBits 失败")
    finally:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(hwnd, hdc)

    rows: List[bytes] = []
    for y in range(height):
        row_bgrx = bytes(buf[y * width * 4: (y + 1) * width * 4])
        # BGRX → RGB
        rows.append(bytes(v for i in range(0, len(row_bgrx), 4) for v in (row_bgrx[i + 2], row_bgrx[i + 1], row_bgrx[i])))
    return width, height, rows


def _png_bytes(width: int, height: int, rows: List[bytes]) -> bytes:
    """把 RGB 行编码成 PNG（只依赖 zlib + crc32，不装 Pillow）。"""
    raw = b"".join(b"\x00" + row for row in rows)  # 每行前置 filter=0
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit 真彩

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


# =====================================================================
# 键盘 / 鼠标
# =====================================================================
_VK_MAP = {
    "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09,
    "space": 0x20, "backspace": 0x08, "bs": 0x08, "delete": 0x2E, "del": 0x2E,
    "insert": 0x2D, "ins": 0x2D,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pgup": 0x21, "pageup": 0x21,
    "pgdn": 0x22, "pagedown": 0x22,
    "lwin": 0x5B, "rwin": 0x5C, "win": 0x5B, "super": 0x5B,
    "shift": 0x10, "lshift": 0xA0, "rshift": 0xA1,
    "ctrl": 0x11, "control": 0x11, "lctrl": 0xA2, "rctrl": 0xA3,
    "alt": 0x12, "menu": 0x12, "lalt": 0xA4, "ralt": 0xA5,
    "capslock": 0x14, "printscreen": 0x2C, "prtsc": 0x2C,
    "scrolllock": 0x91, "pause": 0x13, "apps": 0x5D,
}
for _i in range(1, 25):
    _VK_MAP[f"f{_i}"] = 0x70 + _i - 1  # F1=0x70 … F24=0x87

_MODIFIERS = {"shift", "lshift", "rshift", "ctrl", "control", "lctrl", "rctrl",
             "alt", "menu", "lalt", "ralt", "win", "lwin", "rwin", "super"}

_KEYEVENTF_KEYUP = 0x0002


def _vk_of(name: str) -> int:
    """名字 → 虚拟键码；支持 a-z / 0-9 / 常用键名。"""
    key = name.strip().lower()
    if not key:
        raise ValueError("键名为空")
    if key in _VK_MAP:
        return _VK_MAP[key]
    if len(key) == 1 and ("a" <= key <= "z"):
        return ord(key.upper())
    if len(key) == 1 and key.isdigit():
        return 0x30 + int(key)
    raise ValueError(f"不认识的键名：{name}（示例：enter/esc/tab/win/ctrl/alt+f4/f1…f24）")


def _key_tap(vk: int) -> None:
    u = _user32()
    u.keybd_event(vk, 0, 0, 0)
    time.sleep(0.01)
    u.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)


def _key_combo(combo: str) -> List[int]:
    """解析 "ctrl+s" 这类组合，按下修饰键 → 敲主键 → 松开修饰键。"""
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("组合键为空")
    mods = [_vk_of(p) for p in parts[:-1]]
    main = _vk_of(parts[-1])
    u = _user32()
    for vk in mods:
        u.keybd_event(vk, 0, 0, 0)
    try:
        _key_tap(main)
    finally:
        for vk in reversed(mods):
            u.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
    return mods + [main]


def _type_char(ch: str) -> None:
    """敲入单个 ASCII 字符（VkKeyScanW 解析，自动处理 shift）。"""
    u = _user32()
    u.VkKeyScanW.restype = ctypes.c_short  # 返回 SHORT：低字节 vk，高字节 shift/ctrl 状态
    res = u.VkKeyScanW(ord(ch))
    if res == -1:
        raise ValueError(f"无法敲出字符 {ch!r}，请改用剪贴板粘贴")
    vk = res & 0xFF
    shift = (res >> 8) & 0x01
    ctrl = (res >> 8) & 0x02
    if ctrl:
        raise ValueError(f"字符 {ch!r} 需要 Ctrl 组合，请改用剪贴板粘贴")
    if shift:
        u.keybd_event(_VK_MAP["lshift"], 0, 0, 0)
    try:
        _key_tap(vk)
    finally:
        if shift:
            u.keybd_event(_VK_MAP["lshift"], 0, _KEYEVENTF_KEYUP, 0)


_MOUSE_LEFT, _MOUSE_RIGHT, _MOUSE_MIDDLE = "left", "right", "middle"


def _mouse_event(flags: int, extra: int = 0) -> None:
    _user32().mouse_event(flags, 0, 0, extra, 0)


def _click(button: str, clicks: int) -> None:
    if button not in (_MOUSE_LEFT, _MOUSE_RIGHT, _MOUSE_MIDDLE):
        raise ValueError(f"button 只能是 left/right/middle，收到 {button!r}")
    down = {"left": 0x02, "right": 0x08, "middle": 0x20}[button]
    up = down << 1
    for _ in range(max(1, min(int(clicks), 5))):
        _mouse_event(down)
        time.sleep(0.02)
        _mouse_event(up)
        time.sleep(0.05 if clicks > 1 else 0.01)


# =====================================================================
# 剪贴板（CF_UNICODETEXT）
# =====================================================================
def _clipboard_set(text: str) -> None:
    user32, kernel32 = _user32(), _kernel32()
    # 64 位下 HGLOBAL 是指针宽度，ctypes 默认按 c_int 截断会拿到 0 → 显式声明
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]

    data = text.encode("utf-16-le") + b"\x00\x00"
    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13
    for attempt in range(5):
        if not user32.OpenClipboard(None):
            time.sleep(0.1)
            continue
        try:
            user32.EmptyClipboard()
            handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not handle:
                raise OSError("GlobalAlloc 失败")
            ptr = kernel32.GlobalLock(ctypes.c_void_p(handle))
            if not ptr:
                kernel32.GlobalFree(ctypes.c_void_p(handle))
                raise OSError("GlobalLock 失败")
            ctypes.memmove(ptr, data, len(data))
            kernel32.GlobalUnlock(ctypes.c_void_p(handle))
            if not user32.SetClipboardData(CF_UNICODETEXT, ctypes.c_void_p(handle)):
                kernel32.GlobalFree(ctypes.c_void_p(handle))
                raise OSError("SetClipboardData 失败")
            return
        finally:
            user32.CloseClipboard()
    raise OSError("剪贴板被占用（重试 5 次失败）")


def _clipboard_get() -> str:
    user32, kernel32 = _user32(), _kernel32()
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    CF_UNICODETEXT = 13
    for attempt in range(5):
        if not user32.OpenClipboard(None):
            time.sleep(0.1)
            continue
        try:
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return ""
            ptr = kernel32.GlobalLock(ctypes.c_void_p(handle))
            if not ptr:
                return ""
            try:
                return ctypes.wstring_at(ptr)
            finally:
                kernel32.GlobalUnlock(ctypes.c_void_p(handle))
        finally:
            user32.CloseClipboard()
    raise OSError("剪贴板被占用（重试 5 次失败）")


# =====================================================================
# 窗口枚举
# =====================================================================
def _list_windows(limit: int = 60) -> List[Dict[str, Any]]:
    user32 = _user32()
    windows: List[Dict[str, Any]] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def on_window(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        title = buf.value.strip()
        if not title:
            return True
        pid = ctypes.c_uint32(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = (ctypes.c_long * 4)()
        user32.GetWindowRect(hwnd, rect)
        windows.append(
            {
                "hwnd": int(hwnd),
                "title": title,
                "pid": int(pid.value),
                "rect": [rect[0], rect[1], rect[2], rect[3]],
            }
        )
        return len(windows) < 500

    callback = EnumWindowsProc(on_window)
    user32.EnumWindows(callback, 0)
    # 过滤悬浮/透明小窗（宽高 < 4 的）后取前 limit 个
    windows = [w for w in windows if w["rect"][2] - w["rect"][0] >= 4 and w["rect"][3] - w["rect"][1] >= 4]
    return windows[: max(1, int(limit))]


def _window_focus(hwnd: int) -> str:
    user32 = _user32()
    if not user32.IsWindow(hwnd):
        raise ValueError(f"无效窗口句柄：{hwnd}（请先用 sys_windows 查询）")
    if user32.IsIconic(hwnd):  # 最小化 → 先还原
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    # Windows 前台锁的常见绕法：按一下 Alt 释放，再 SetForegroundWindow
    user32.keybd_event(0x12, 0, 0, 0)
    ok = user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x12, 0, _KEYEVENTF_KEYUP, 0)
    if not ok:
        raise OSError("SetForegroundWindow 失败（窗口可能拒绝焦点）")
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


# =====================================================================
# 进程
# =====================================================================
def _process_list(filter_str: str = "", limit: int = 25) -> List[Dict[str, Any]]:
    if _IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        ).stdout
        rows: List[Dict[str, Any]] = []
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 5:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            mem_kb = parts[4].replace(",", "").replace(" K", "").replace("\u00a0", "")
            try:
                mem_kb = int(mem_kb)
            except ValueError:
                mem_kb = 0
            if filter_str and filter_str.lower() not in parts[0].lower():
                continue
            rows.append({"pid": pid, "name": parts[0], "mem_mb": round(mem_kb / 1024, 1)})
        rows.sort(key=lambda r: -r["mem_mb"])
        return rows[: max(1, int(limit))]
    else:
        out = subprocess.run(
            ["ps", "-eo", "pid,comm,rss", "--sort=-rss"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        rows = []
        for line in out.splitlines()[1:]:
            bits = line.split()
            if len(bits) < 3:
                continue
            if filter_str and filter_str.lower() not in line.lower():
                continue
            rows.append({"pid": int(bits[0]), "name": " ".join(bits[1:-1]), "mem_mb": round(int(bits[-1]) / 1024, 1)})
        return rows[: max(1, int(limit))]


def _drives() -> List[Dict[str, Any]]:
    kernel32 = _kernel32()
    mask = kernel32.GetLogicalDrives()
    types = {2: "可移动", 3: "本地磁盘", 4: "网络盘", 5: "光驱", 6: "内存盘"}
    out: List[Dict[str, Any]] = []
    for i in range(26):
        if not (mask >> i) & 1:
            continue
        letter = chr(ord("A") + i)
        dtype = types.get(kernel32.GetDriveTypeW(f"{letter}:\\"))
        out.append({"drive": f"{letter}:\\", "type": dtype or "未知"})
    return out


def _memory_status() -> Dict[str, Any]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if _kernel32().GlobalMemoryStatusEx(ctypes.byref(stat)):
        return {
            "load_pct": int(stat.dwMemoryLoad),
            "total_gb": round(stat.ullTotalPhys / 1024**3, 1),
            "avail_gb": round(stat.ullAvailPhys / 1024**3, 1),
        }
    return {"load_pct": -1, "total_gb": -1, "avail_gb": -1}


# =====================================================================
# 全盘找文件（带超时与取消）
# =====================================================================
_FAST_SKIP = {"$recycle.bin", "system volume information", "winxsx",
              "node_modules", "__pycache__", ".git", "dumpstack.log.tmp"}


def _find_files(
    pattern: str,
    start: str,
    max_results: int,
    timeout_s: int,
    skip_system: bool,
    _cancel: Any,
) -> Tuple[List[str], str]:
    deadline = time.monotonic() + max(1, min(int(timeout_s), 120))
    hits: List[str] = []
    scanned = 0
    protected = sandbox_protected_names() if skip_system else set()
    start_path = Path(start)
    lowered_pat = pattern.lower()
    for dirpath, dirnames, filenames in os.walk(start_path):
        if _cancel is not None and getattr(_cancel, "is_cancelled", False):
            return hits, f"已取消：{_cancel.reason}"
        if time.monotonic() > deadline:
            return hits, f"超过 {timeout_s}s 截断（已扫描 {scanned} 个文件）"
        # 剪枝：跳过系统目录与回收站
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _FAST_SKIP and d.lower() not in protected
        ]
        scanned += len(filenames)
        for name in filenames:
            if fnmatch.fnmatch(name.lower(), lowered_pat):
                hits.append(str(Path(dirpath) / name))
                if len(hits) >= max_results:
                    return hits, f"达到 {max_results} 条上限（已扫描 {scanned} 个文件）"
    return hits, f"扫描完成（{scanned} 个文件）"


def sandbox_protected_names() -> set:
    """系统保护目录的目录名集合（剪枝加速用）。"""
    from sandbox import _protected_write_dirs

    return {d.name.lower() for d in _protected_write_dirs()}


# =====================================================================
# 工具注册
# =====================================================================
def register(reg: ToolRegistry, sandbox: Sandbox) -> None:
    """把电脑操控工具挂到注册表。sandbox 必须是 scope="computer"。"""
    if sandbox.scope != "computer":
        raise ValueError("computer_tools 只能在电脑作用域（--full-computer）下注册")

    def _err(e: Exception) -> Dict[str, Any]:
        return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}

    # ---------------- sys_*：只读查询 ----------------
    @reg.register()
    def sys_info() -> Dict[str, Any]:
        """查看这台电脑的基本信息（系统版本 / 主机名 / CPU / 内存 / 各盘空间）。

        任务涉及「系统/磁盘/资源」时先调用它建立全局认知。
        """
        try:
            import socket

            info: Dict[str, Any] = {
                "os": f"{platform.system()} {platform.release()}（{platform.version()}）",
                "hostname": socket.gethostname(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
            }
            if _IS_WINDOWS:
                info["memory"] = _memory_status()
                disks = []
                for d in _drives():
                    if d["type"] in ("本地磁盘", "网络盘", "可移动"):
                        try:
                            usage = shutil.disk_usage(d["drive"])
                            disks.append(
                                f"{d['drive']}（{d['type']}）总量 {usage.total / 1024**3:.0f}G / "
                                f"可用 {usage.free / 1024**3:.0f}G"
                            )
                        except OSError:
                            disks.append(f"{d['drive']}（{d['type']}）不可读")
                info["disks"] = disks
            else:
                usage = shutil.disk_usage(sandbox.root)
                info["disks"] = [f"{usage.total / 1024**3:.0f}G 总量 / {usage.free / 1024**3:.0f}G 可用"]
            return {
                "ok": True,
                "output": json.dumps(info, ensure_ascii=False, indent=2),
                "error": None,
            }
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_drives() -> Dict[str, Any]:
        """列出这台电脑所有盘符与类型（本地磁盘/光驱/可移动/网络盘）。"""
        try:
            rows = _drives()
            return {
                "ok": True,
                "output": "\n".join(f"{d['drive']}  {d['type']}" for d in rows) or "（没有可用盘符）",
                "error": None,
            }
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_find_file(
        pattern: str,
        start_dir: str = "C:\\",
        max_results: int = 30,
        timeout_s: int = 20,
        skip_system: bool = True,
        _cancel: Any = None,
    ) -> Dict[str, Any]:
        """按文件名 glob 在整台电脑上找文件。

        pattern: 文件名 glob，如 "*.xlsx"、"笔记*.md"（不区分大小写）
        start_dir: 起始目录（默认 C:\\；建议先用 sys_drives 看清盘符）
        max_results: 最多返回几条（1-200）
        timeout_s: 超时秒数（1-120），超时返回已找到的部分
        skip_system: True 时跳过 Windows/Program Files 等系统目录（更快）
        """
        try:
            start = sandbox.resolve(start_dir)
            if not start.is_dir():
                return {"ok": False, "output": "", "error": f"不是目录：{start_dir}"}
            hits, note = _find_files(
                pattern, start, max(1, min(int(max_results), 200)),
                timeout_s, skip_system, _cancel,
            )
            body = "\n".join(hits) if hits else "（没找到匹配文件）"
            return {"ok": True, "output": f"{note}\n{body}", "error": None}
        except SandboxError as e:
            return {"ok": False, "output": "", "error": str(e)}
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_screenshot(save_as: str = "", region: str = "") -> Dict[str, Any]:
        """截取当前屏幕保存为 PNG（给用户核对画面，模型自己看不到图像内容）。

        save_as: 保存路径。留空自动存到工作区 .agent/screens/shot-时间戳.png
        region: "x,y,w,h" 只截区域；留空截全屏
        """
        try:
            full_w, full_h, rows = _capture_screen()
            x = y = 0
            if region and region.strip():
                try:
                    x, y, w, h = (int(v.strip()) for v in region.split(","))
                except ValueError:
                    return {"ok": False, "output": "", "error": 'region 格式："x,y,w,h"，如 "0,0,800,600"'}
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > full_w or y + h > full_h:
                    return {"ok": False, "output": "", "error": f"region 越界（屏幕 {full_w}x{full_h}）"}
                rows = [row[x * 3: (x + w) * 3] for row in rows[y: y + h]]
                width, height = w, h
            else:
                width, height = full_w, full_h
            target = sandbox.resolve(save_as or f".agent/screens/shot-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png", for_write=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_png_bytes(width, height, rows))
            return {
                "ok": True,
                "output": f"截图已保存：{target}（{width}x{height}，{target.stat().st_size / 1024:.0f} KB）。"
                          f"请在电脑上打开该文件核对画面。",
                "error": None,
            }
        except SandboxError as e:
            return {"ok": False, "output": "", "error": str(e)}
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_screen_size() -> Dict[str, Any]:
        """查看屏幕分辨率与当前鼠标坐标。"""
        try:
            user32 = _user32()
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            pt = (ctypes.c_long * 2)()
            user32.GetCursorPos(pt)
            return {
                "ok": True,
                "output": f"主屏 {width}x{height}；鼠标当前在 ({pt[0]}, {pt[1]})",
                "error": None,
            }
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_windows(limit: int = 40) -> Dict[str, Any]:
        """列出当前所有可见窗口（标题 / 句柄 / 进程 PID / 位置）。

        要操作某个窗口时先用它查句柄，再把 hwnd 传给 ui_window_focus。
        """
        try:
            rows = _list_windows(limit=max(1, min(int(limit), 200)))
            lines = [
                f"hwnd={w['hwnd']}  pid={w['pid']}  [{w['rect'][0]},{w['rect'][1]} {w['rect'][2] - w['rect'][0]}x{w['rect'][3] - w['rect'][1]}]  {w['title'][:60]}"
                for w in rows
            ]
            return {"ok": True, "output": "\n".join(lines) or "（没有可见窗口）", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_active_window() -> Dict[str, Any]:
        """查看当前前台窗口（正在使用的那个）的标题与句柄。"""
        try:
            user32 = _user32()
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            pid = ctypes.c_uint32(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return {
                "ok": True,
                "output": f"前台窗口：{buf.value}（hwnd={hwnd}，pid={pid.value}）",
                "error": None,
            }
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_processes(filter: str = "", limit: int = 25) -> Dict[str, Any]:
        """列出正在运行的进程（按内存占用排序）。

        filter: 只看名字包含该关键词的进程；留空看全部前 N 个
        limit: 最多返回几条（1-100）
        """
        try:
            rows = _process_list(filter, limit=max(1, min(int(limit), 100)))
            lines = [f"pid={r['pid']}  {r['mem_mb']}MB  {r['name']}" for r in rows]
            return {"ok": True, "output": "\n".join(lines) or "（没有匹配进程）", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register()
    def sys_clipboard_read() -> Dict[str, Any]:
        """读取剪贴板文本内容。"""
        try:
            text = _clipboard_get()
            shown = text if len(text) <= 4000 else text[:4000] + f"…（共 {len(text)} 字符）"
            return {"ok": True, "output": shown or "（剪贴板为空或不是文本）", "error": None}
        except Exception as e:
            return _err(e)

    # ---------------- ui_*：真实桌面动作（全部需要确认）----------------
    @reg.register(dangerous=True, require_confirm=True)
    def ui_mouse_move(x: int, y: int) -> Dict[str, Any]:
        """把鼠标移动到屏幕坐标 (x, y)。真实桌面上移动光标。

        x: 屏幕横坐标（0 在最左）
        y: 屏幕纵坐标（0 在最上）
        """
        try:
            user32 = _user32()
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            if not (0 <= int(x) < width and 0 <= int(y) < height):
                return {"ok": False, "output": "", "error": f"坐标越界：({x}, {y})，屏幕 {width}x{height}"}
            user32.SetCursorPos(int(x), int(y))
            return {"ok": True, "output": f"鼠标已移动到 ({x}, {y})", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_mouse_click(x: int = -1, y: int = -1, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        """在真实桌面上点击鼠标。

        x: 点击位置横坐标；-1 表示在当前位置点
        y: 点击位置纵坐标；-1 表示在当前位置点
        button: left / right / middle
        clicks: 连点次数（1-5；2=双击）
        """
        try:
            if x >= 0 or y >= 0:
                move = ui_mouse_move(int(x), int(y))
                if not move.get("ok"):
                    return move
            _click(button, int(clicks))
            pos = "当前位置" if x < 0 else f"({x}, {y})"
            return {"ok": True, "output": f"已点击 {pos}（{button}，{clicks} 次）", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_scroll(amount: int = 3, x: int = -1, y: int = -1) -> Dict[str, Any]:
        """滚动鼠标滚轮。正数向上滚，负数向下滚。

        amount: 滚动格数（-10 到 10）
        """
        try:
            amount = max(-10, min(int(amount), 10))
            if x >= 0 and y >= 0:
                move = ui_mouse_move(int(x), int(y))
                if not move.get("ok"):
                    return move
            _user32().mouse_event(0x0800, 0, 0, amount * 120, 0)  # MOUSEEVENTF_WHEEL
            return {"ok": True, "output": f"已滚动 {amount} 格", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_key(combo: str = "enter") -> Dict[str, Any]:
        """按下键盘组合键。会在真实桌面生效。

        combo: 组合键写法。示例："enter"、"esc"、"win"、"alt+tab"、"ctrl+s"、"ctrl+shift+esc"、"f5"
        """
        try:
            keys = _key_combo(combo)
            return {"ok": True, "output": f"已按 {combo}（vk={keys}）", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_type_text(text: str, interval_ms: int = 12) -> Dict[str, Any]:
        """逐字输入文本到当前聚焦的输入框（真实键盘输入）。

        text: 要输入的完整文本
        interval_ms: 每个字符间隔毫秒（太快可能丢字）
        含中文等非 ASCII 字符时自动改用「写入剪贴板 + Ctrl+V」方式。
        """
        try:
            if not text:
                return {"ok": False, "output": "", "error": "text 为空"}
            delay = max(0, min(int(interval_ms), 200)) / 1000
            if all(ord(c) < 128 for c in text):
                for ch in text:
                    _type_char(ch)
                    time.sleep(delay)
                return {"ok": True, "output": f"已逐字输入 {len(text)} 个字符", "error": None}
            # 非 ASCII：走剪贴板粘贴（保底还原用户原剪贴板内容）
            old = _clipboard_get()
            _clipboard_set(text)
            time.sleep(0.05)
            _key_combo("ctrl+v")
            time.sleep(0.35)
            try:
                _clipboard_set(old)
            except Exception:
                pass
            return {
                "ok": True,
                "output": f"已通过剪贴板粘贴输入 {len(text)} 个字符（含非 ASCII，原剪贴板已还原）",
                "error": None,
            }
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_window_focus(hwnd: int) -> Dict[str, Any]:
        """把某个窗口切到前台（先还原最小化再抢焦点）。

        hwnd: 窗口句柄，来自 sys_windows 的返回值
        """
        try:
            title = _window_focus(int(hwnd))
            return {"ok": True, "output": f"已切到前台：{title}", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_clipboard_write(text: str) -> Dict[str, Any]:
        """把文本写入系统剪贴板（会覆盖现有内容）。"""
        try:
            _clipboard_set(text)
            return {"ok": True, "output": f"已写入剪贴板（{len(text)} 字符）", "error": None}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_open_app(target: str) -> Dict[str, Any]:
        """打开程序 / 文件 / 文件夹 / 网页（用系统默认方式，像双击一样）。

        target: 程序名或路径。示例："notepad"、"C:\\Program Files\\...\\app.exe"、
                "D:\\资料\\报告.docx"、"https://example.com"
        """
        try:
            if not target or not str(target).strip():
                return {"ok": False, "output": "", "error": "target 不能为空"}
            target_str = str(target).strip()
            if os.name == "nt":
                os.startfile(target_str)  # noqa: S606 —— 打开操作本身已走确认闸门
                return {"ok": True, "output": f"已请求系统打开：{target_str}", "error": None}
            return {"ok": False, "output": "", "error": f"当前平台（{os.name}）不支持 ui_open_app，请用 run_shell"}
        except Exception as e:
            return _err(e)

    @reg.register(dangerous=True, require_confirm=True)
    def ui_kill_process(pid: int) -> Dict[str, Any]:
        """强制结束一个进程（按 PID，连同子进程一起）。不可撤销，慎用。

        pid: 进程 PID，来自 sys_processes 的返回值
        """
        try:
            pid = int(pid)
            if pid <= 4:
                return {"ok": False, "output": "", "error": f"拒绝结束系统关键进程（pid={pid}）"}
            if os.getpid() == pid:
                return {"ok": False, "output": "", "error": "拒绝结束 agent 自己"}
            if _IS_WINDOWS:
                proc = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                )
                ok = proc.returncode == 0
                return {
                    "ok": ok,
                    "output": f"已结束 pid={pid}" if ok else "",
                    "error": None if ok else (proc.stdout + proc.stderr).strip() or f"退出码 {proc.returncode}",
                }
            proc = subprocess.run(
                ["kill", "-9", str(pid)], capture_output=True, text=True, timeout=15
            )
            ok = proc.returncode == 0
            return {"ok": ok, "output": f"已结束 pid={pid}" if ok else "", "error": None if ok else proc.stderr.strip()}
        except Exception as e:
            return _err(e)


def computer_tool_names() -> List[str]:
    """给 /api/scope 展示用：电脑作用域会启用哪些工具。"""
    names = [
        "sys_info", "sys_drives", "sys_find_file", "sys_screenshot",
        "sys_screen_size", "sys_windows", "sys_active_window",
        "sys_processes", "sys_clipboard_read",
        "ui_mouse_move", "ui_mouse_click", "ui_scroll", "ui_key",
        "ui_type_text", "ui_window_focus", "ui_clipboard_write",
        "ui_open_app", "ui_kill_process",
    ]
    return names
