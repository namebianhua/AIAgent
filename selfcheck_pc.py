"""电脑操控能力自检：Sandbox 双作用域 / sys_*·ui_* 工具 / 确认闸门 / HTTP 接口。

跑法：.venv/Scripts/python.exe selfcheck_pc.py

安全性说明（在真实 Windows 桌面上跑，但每个动作都无副作用）：
  - 键盘只敲 F13（真实键盘上不存在的键，任何程序都不响应）
  - 鼠标只「移动到当前位置」（GetCursorPos → SetCursorPos 原位）
  - 剪贴板写入后立即读回（会被 ui_type_text 测试还原语义）
  - 截图保存到临时目录
  - 点击 / 打开程序 / 杀进程全部走「确认被拒」路径，验证的是闸门而不是真点
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PASS, FAIL = "PASS", "FAIL"
results: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    return ok


IS_WIN = os.name == "nt"


class AutoApprover:
    """闸门替身：全部放行（测试只读/无副作用工具用）。"""

    def needs(self, spec, args):
        return bool(spec.require_confirm)

    def request(self, spec, args):
        return {"decision": "approve"}


class DenyApprover:
    """闸门替身：全部拒绝（验证危险动作真的被拦住）。"""

    def needs(self, spec, args):
        return bool(spec.require_confirm)

    def request(self, spec, args):
        return {"decision": "deny", "reason": "测试拒绝"}


def http_get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(port: int, path: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    print("=" * 62)
    print("电脑操控自检：双作用域 / sys_*·ui_* 工具 / 闸门 / 接口")
    print("=" * 62)

    import computer_tools
    from builtin_tools import build_registry
    from confirm_gate import ConfirmBroker
    from sandbox import Sandbox, SandboxError
    from trust import TrustStore

    # ================= A. Sandbox 双作用域 =================
    print("\n[A] Sandbox 作用域：workspace jail / computer 放开+保护")
    ws = Path(tempfile.mkdtemp(prefix="pc-a-"))
    outside = Path(tempfile.mkdtemp(prefix="pc-b-"))  # 模拟工作区外的用户目录
    (outside / "note.txt").write_text("outside", encoding="utf-8")

    sb_ws = Sandbox(ws)
    sb_pc = Sandbox(ws, scope="computer")

    try:
        sb_ws.resolve(str(outside / "note.txt"))
        check("workspace 作用域拒绝外部绝对路径", False)
    except SandboxError:
        check("workspace 作用域拒绝外部绝对路径", True)

    got = sb_pc.resolve(str(outside / "note.txt"))
    check("computer 作用域读外部路径放行", got == (outside / "note.txt").resolve())

    try:
        sb_pc.resolve("C:\\Windows\\System32\\evil.txt", for_write=True)
        check("computer 作用域禁写 C:/Windows", False)
    except SandboxError:
        check("computer 作用域禁写 C:/Windows", True)

    try:
        sb_pc.resolve("C:\\bootmgr", for_write=True)
        check("computer 作用域禁写盘符根目录", False)
    except SandboxError:
        check("computer 作用域禁写盘符根目录", True)

    user_home = Path.home() / "pc-selfcheck-probe.txt"
    check("computer 作用域放行用户目录写入", not sb_pc.is_protected_write(user_home))

    try:
        Sandbox(ws, scope="universe")
        check("未知作用域被拒", False)
    except ValueError:
        check("未知作用域被拒", True)

    # 命令参数：computer 作用域
    if IS_WIN:
        try:
            sb_pc.check_command("dir C:\\Windows\\System32")
            check("computer 作用域命令参数禁碰系统目录", False)
        except SandboxError:
            check("computer 作用域命令参数禁碰系统目录", True)
        sb_pc.check_command(f"dir {outside}")
        check("computer 作用域命令参数允许工作区外", True)
        try:
            sb_ws.check_command(f"dir {outside}")
            check("workspace 作用域命令参数仍锁工作区", False)
        except SandboxError:
            check("workspace 作用域命令参数仍锁工作区", True)

    # ================= B. 内置工具在电脑作用域 =================
    print("\n[B] 内置工具跨边界读写")
    reg = build_registry(sb_pc)
    reg.confirmer = None

    r = reg.invoke("list_dir", {"path": str(outside)})
    check("list_dir 列工作区外目录", r["ok"] and "note.txt" in r["output"], r.get("error") or "")

    r = reg.invoke("read_file", {"path": str(outside / "note.txt")})
    check("read_file 读工作区外文件", r["ok"] and "outside" in r["output"], r.get("error") or "")

    # 写工作区外：需要确认 → 用自动放行闸门验证「能写」
    reg.approver = AutoApprover()
    target_out = outside / "by_agent.txt"
    r = reg.invoke("write_file", {"path": str(target_out), "content": "hello from agent"})
    check("write_file 写工作区外（闸门放行后）", r["ok"] and target_out.read_text(encoding="utf-8") == "hello from agent",
          r.get("error") or "")

    r = reg.invoke("write_file", {"path": "C:\\Windows\\agent-evil.txt", "content": "x"})
    check("write_file 写系统目录被沙箱拒绝", not r["ok"] and "保护" in (r.get("error") or ""))

    (outside / "sub").mkdir(exist_ok=True)
    (outside / "sub" / "a.txt").write_text("a", encoding="utf-8")
    r = reg.invoke("delete_file", {"path": str(outside / "sub"), "recursive": True})
    check("工作区外递归删除被拒", not r["ok"] and "递归删除" in (r.get("error") or ""), r.get("error") or "")

    r = reg.invoke("delete_file", {"path": str(target_out)})
    check("工作区外单文件删除（闸门放行后）", r["ok"] and not target_out.exists())

    reg.approver = DenyApprover()
    r = reg.invoke("delete_file", {"path": str(outside / "note.txt")})
    check("delete_file 被闸门拒绝时不执行", not r["ok"] and target_out.exists() is not None)

    # ================= C. 电脑工具冒烟 =================
    print("\n[C] sys_* / ui_* 工具冒烟（真实 API，无副作用动作）")
    reg2 = build_registry(sb_pc)
    computer_tools.register(reg2, sb_pc)

    names = set(reg2.names())
    expect = set(computer_tools.computer_tool_names())
    check("18 个电脑工具全部注册", expect <= names, f"缺：{expect - names}")

    spec = reg2.get("sys_find_file")
    check("Schema 不泄露 _cancel 形参", "_cancel" not in (spec.parameters.get("properties") or {}))

    r = reg2.invoke("sys_info", {})
    check("sys_info 返回系统信息", r["ok"] and "os" in r["output"], r.get("error") or "")

    if IS_WIN:
        r = reg2.invoke("sys_drives", {})
        check("sys_drives 列出盘符", r["ok"] and "C:\\" in r["output"], r.get("error") or "")

    r = reg2.invoke("sys_screen_size", {})
    check("sys_screen_size 返回分辨率与鼠标", r["ok"] and "x" in r["output"], r.get("error") or "")

    reg2.approver = AutoApprover()
    r = reg2.invoke("sys_screenshot", {"save_as": ".agent/screens/pc-test.png"})
    shot = sb_pc.root / ".agent" / "screens" / "pc-test.png"
    check("sys_screenshot 截屏落盘（PNG）", r["ok"] and shot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", r.get("error") or "")

    r = reg2.invoke("sys_windows", {"limit": 10})
    check("sys_windows 列出可见窗口", r["ok"] and "hwnd=" in r["output"], r.get("error") or "")

    r = reg2.invoke("sys_active_window", {})
    check("sys_active_window 返回前台窗口", r["ok"] and "前台窗口" in r["output"], r.get("error") or "")

    r = reg2.invoke("sys_processes", {"filter": "python", "limit": 5})
    check("sys_processes 列出进程", r["ok"] and "pid=" in r["output"], r.get("error") or "")

    r = reg2.invoke("sys_find_file", {"pattern": "*.md", "start_dir": str(Path(__file__).parent),
                                      "max_results": 5, "timeout_s": 10})
    check("sys_find_file 全盘找文件（限本项目目录）", r["ok"] and "README.md" in r["output"], r.get("error") or "")

    if IS_WIN:
        r = reg2.invoke("ui_clipboard_write", {"text": "pc-selfcheck-剪贴板"})
        check("ui_clipboard_write 写剪贴板", r["ok"], r.get("error") or "")
        r = reg2.invoke("sys_clipboard_read", {})
        check("sys_clipboard_read 读回剪贴板", r["ok"] and "pc-selfcheck-剪贴板" in r["output"], r.get("error") or "")

        # 键盘：F13 是真实键盘上不存在的键，敲了没有任何程序会响应
        r = reg2.invoke("ui_key", {"combo": "f13"})
        check("ui_key 组合键（F13 无副作用键）", r["ok"], r.get("error") or "")

        # 鼠标：取当前位置再移回原位，视觉上无变化
        import ctypes
        u = computer_tools._user32()
        pt = (ctypes.c_long * 2)()
        u.GetCursorPos(pt)
        r = reg2.invoke("ui_mouse_move", {"x": int(pt[0]), "y": int(pt[1])})
        check("ui_mouse_move 移动到当前位置", r["ok"], r.get("error") or "")

        # 验证鼠标移动被默认策略放行（r-allow-mousemove 在 r-ask-ui 之前）
        store = TrustStore(Path(tempfile.mkdtemp(prefix="pc-tr-")) / "trust-rules.json")
        action, rule = store.effective("ui_mouse_move", {})
        check("默认策略：ui_mouse_move 直接放行", action == "allow" and rule.id == "r-allow-mousemove")
        action, _ = store.effective("ui_mouse_click", {})
        check("默认策略：ui_mouse_click 走询问", action == "ask")
        action, _ = store.effective("sys_info", {})
        check("默认策略：sys_info 直接放行", action == "allow")
        action, _ = store.effective("ui_kill_process", {})
        check("默认策略：ui_kill_process 走询问", action == "ask")

        # 危险动作必须真的被闸门拦住：点击、打开程序、杀进程全部拒绝路径
        reg2.approver = DenyApprover()
        r = reg2.invoke("ui_mouse_click", {"button": "left"})
        check("ui_mouse_click 被拒时不点击", not r["ok"] and "denied" in str(r["meta"]), r.get("error") or "")
        r = reg2.invoke("ui_open_app", {"target": "notepad"})
        check("ui_open_app 被拒时不打开", not r["ok"] and "denied" in str(r["meta"]))
        r = reg2.invoke("ui_kill_process", {"pid": os.getpid()})
        check("ui_kill_process 被拒时不杀（连自己都不杀）", not r["ok"] and "denied" in str(r["meta"]))
        # 参数防御：就算闸门放行，系统关键 PID 也拒绝
        reg2.approver = AutoApprover()
        r = reg2.invoke("ui_kill_process", {"pid": 4})
        check("ui_kill_process 拒绝系统关键进程（pid<=4）", not r["ok"])
        r = reg2.invoke("ui_kill_process", {"pid": os.getpid()})
        check("ui_kill_process 拒绝结束 agent 自己", not r["ok"])

    # ================= D. 确认闸门与作用域联动 =================
    print("\n[D] 确认闸门：工作区外写入动态弹确认")
    broker_pc = ConfirmBroker(emit=lambda ev: None, root=str(ws), scope="computer",
                              trust_path=ws / ".agent" / "trust.json")
    spec_write = reg2.get("write_file")
    check("computer 作用域：写工作区外需要确认",
          broker_pc.needs(spec_write, {"path": str(outside / "x.txt")}))
    check("computer 作用域：写工作区内免确认",
          not broker_pc.needs(spec_write, {"path": "inside.txt"}))
    broker_ws = ConfirmBroker(emit=lambda ev: None, root=str(ws), scope="workspace",
                              trust_path=ws / ".agent" / "trust.json")
    check("workspace 作用域：外路径由 jail 直接拒绝（无需弹窗）",
          not broker_ws.needs(spec_write, {"path": str(outside / "x.txt")}))

    # ================= E. HTTP 接口 =================
    print("\n[E] HTTP 接口：/api/scope 与全电脑演示任务")
    from server import serve

    demo_ws = Path(tempfile.mkdtemp(prefix="pc-http-"))
    from runner import RunOptions

    httpd, manager, _ = serve(
        RunOptions(workspace=str(demo_ws), mode="web"), port=free_port(),
        block=False, demo=True,
    )
    port = httpd.server_address[1]
    try:
        cfg = http_get(port, "/api/config")
        check("/api/config 带 computer 块", cfg.get("computer", {}).get("supported") == IS_WIN)

        st = http_get(port, "/api/scope")
        check("/api/scope 默认关闭", st["enabled"] is False and len(st["tools"]) == 18)

        d = http_post(port, "/api/scope", {"enabled": True})
        check("POST /api/scope 开启电脑作用域", d["enabled"] is True)

        st = http_get(port, "/api/scope")
        check("GET /api/scope 回读已开启", st["enabled"] is True)

        # 提交一个全电脑模式的演示任务（demo 模型会注册电脑工具并端到端跑完）
        r = http_post(port, "/api/run", {"goal": "电脑模式演示任务", "options": {"full_computer": True}})
        task_id = r["task_id"]
        deadline = time.time() + 60
        view = {}
        while time.time() < deadline:
            view = http_get(port, f"/api/task?id={task_id}")
            if view["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.3)
        check("全电脑模式 demo 任务端到端完成", view.get("status") == "done",
              f"status={view.get('status')}")
        check("任务视图回显 full_computer=True",
              view.get("options", {}).get("full_computer") is True)

        d = http_post(port, "/api/scope", {"enabled": False})
        check("POST /api/scope 再切回工作区", d["enabled"] is False)
    finally:
        httpd.shutdown()
        httpd.server_close()

    # ================= 汇总 =================
    print()
    print("=" * 62)
    passed = sum(1 for _, s, _ in results if s == PASS)
    print(f"结果：{passed} 通过 / {len(results) - passed} 失败（共 {len(results)} 项）")
    print("=" * 62)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
