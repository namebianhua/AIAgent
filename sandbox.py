"""安全沙箱（T1）：路径越狱防护 + 命令白名单 + 只读模式 + 审计日志 + 快照回滚。"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许执行的命令（按可执行文件名的小写 stem 匹配）
DEFAULT_ALLOWED_CMDS = {
    "python", "python3", "py", "pip", "pip3",
    "git", "node", "npm", "npx", "go", "cargo", "java", "javac",
    "ls", "dir", "cat", "type", "head", "tail", "grep", "find", "wc",
    "echo", "curl", "mkdir", "copy", "move", "del", "rm", "rmdir", "rd",
}

# 出现即拒绝的 shell 元字符（防命令拼接/重定向逃逸）
_META_CHARS = (";", "&", "|", "`", "$", ">", "<", "(", ")", "\n", "\r")
if os.name == "nt":
    _META_CHARS = _META_CHARS + ("%", "!")

# 破坏性片段（即使命令在白名单里也拦）
_DANGEROUS_SNIPPETS = (
    "rm -rf /", "rm -rf ~", "del /s /q c:\\", "format ", "mkfs",
    "shutdown", "reboot", ":(){:|:&};:", "diskpart", "reg delete",
    "rd /s", "rmdir /s", "del /s", "erase /s", "attrib -h -r -s",
)

# 电脑作用域下禁止写入/删除的目录（系统目录：写坏了机器就废了）
def _protected_write_dirs() -> List[Path]:
    dirs: List[Path] = []
    if os.name == "nt":
        for var in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData"):
            val = os.environ.get(var)
            if val:
                dirs.append(Path(val).resolve())
        dirs.append(Path("C:\\Windows").resolve())
    else:
        for name in ("/bin", "/sbin", "/usr/bin", "/usr/sbin", "/etc", "/boot", "/lib", "/lib64", "/usr/lib"):
            if Path(name).exists():
                dirs.append(Path(name).resolve())
    return [d for d in dirs if d.exists()]


class SandboxError(PermissionError):
    """沙箱拒绝执行。"""


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except (ValueError, OSError):
        return False


class Sandbox:
    """把 agent 的所有文件/命令操作锁在一个根目录内。

    root       工作区根目录，所有相对路径都以它为准
    readonly   True 时禁止任何写操作
    scope      "workspace"（默认）：路径 jail 到工作区内，越界即拒绝；
               "computer"：放开全电脑——读任意路径，写/删仍受保护目录约束，
               工作区外写入需人工确认（由 ConfirmBroker 按参数动态判定），
               工作区外禁止递归删除目录（防止灾难性 rm -rf 类操作）。
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        readonly: bool = False,
        audit_file: Optional[str | os.PathLike[str]] = None,
        allowed_cmds: Optional[set[str]] = None,
        scope: str = "workspace",
    ) -> None:
        if scope not in ("workspace", "computer"):
            raise ValueError(f"未知作用域：{scope}（只能是 workspace / computer）")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.scope = scope
        self.allowed_cmds = {c.lower() for c in (allowed_cmds or DEFAULT_ALLOWED_CMDS)}

        audit_path = Path(audit_file) if audit_file else self.root / ".agent" / "audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path = audit_path

    # ---------- 路径 ----------
    def is_inside(self, path: Path) -> bool:
        try:
            return _is_under(Path(path).resolve(), self.root)
        except (ValueError, OSError):
            return False

    def is_protected_write(self, path: Path) -> bool:
        """电脑作用域：系统目录（Windows/Program Files/ProgramData 等）与盘符根下的
        直接子项一律禁写。后者拦住「在 C:\\ 根下乱建文件 / 覆盖 boot 文件」。"""
        try:
            target = Path(path).resolve()
        except (ValueError, OSError):
            return True  # 解析不了的路径按危险处理
        for d in _protected_write_dirs():
            if _is_under(target, d):
                return True
        # 盘符根目录下的直接子项（C:\pagefile.sys、C:\bootmgr、在根下新建文件）
        drive = os.path.splitdrive(str(target))
        if drive[0] and Path(target).parent == Path(drive[0] + os.sep):
            return True
        return False

    def resolve(self, p: str, for_write: bool = False) -> Path:
        """解析路径。workspace 作用域：越界拒绝（原行为不变）。
        computer 作用域：绝对路径放行；写操作撞保护目录时拒绝。"""
        if p is None:
            raise SandboxError("路径不能为空")
        raw = str(p).strip()
        if not raw:
            raise SandboxError("路径不能为空")

        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / raw
        try:
            target = candidate.resolve()
        except (ValueError, OSError) as e:
            raise SandboxError(f"路径无法解析：{raw}（{e}）")

        if self.scope == "workspace":
            if not self.is_inside(target):
                raise SandboxError(f"路径越狱，已被拒绝：{raw}（工作区：{self.root}）")
            return target

        # computer 作用域
        if for_write and self.is_protected_write(target):
            raise SandboxError(
                f"系统保护路径，禁止写入/删除：{raw}"
                "（Windows / Program Files / ProgramData / 盘符根目录）"
            )
        return target

    def check_recursive_delete(self, target: Path) -> None:
        """删除目录树前的最后一道闸：工作区外不允许递归删除。"""
        if self.scope == "workspace":
            return
        if not self.is_inside(target):
            raise SandboxError(
                "电脑作用域下，工作区之外的目录禁止递归删除（防灾难性误删）。"
                "请逐个文件删除，或请用户手动处理。"
            )

    def check_writable(self) -> None:
        if self.readonly:
            raise SandboxError("当前是只读模式，禁止写入/修改")

    # ---------- 审计 ----------
    def audit(self, event: Dict[str, Any]) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
        try:
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---------- 命令 ----------
    def _extract_path_args(self, cmd: str) -> List[str]:
        try:
            tokens = shlex.split(cmd, posix=(os.name != "nt"))
        except ValueError:
            tokens = cmd.split()
        out = []
        for tok in tokens[1:]:
            raw = tok.split("=", 1)[1] if tok.startswith("-") and "=" in tok else tok
            if "://" in raw:
                continue
            if any(c in raw for c in ("/", "\\", ":")):
                out.append(raw)
        return out

    def check_command(self, cmd: str) -> List[str]:
        """校验命令是否允许执行，返回拆分后的 token。"""
        if not cmd or not cmd.strip():
            raise SandboxError("空命令")

        for ch in _META_CHARS:
            if ch in cmd:
                raise SandboxError(
                    f"命令包含被禁用的 shell 元字符 {ch!r}（不支持链式/管道/重定向）"
                )

        lowered = cmd.lower()
        for bad in _DANGEROUS_SNIPPETS:
            if bad in lowered:
                raise SandboxError(f"命令包含高危片段，已拦截：{bad}")

        try:
            tokens = shlex.split(cmd, posix=(os.name != "nt"))
        except ValueError as e:
            raise SandboxError(f"命令解析失败：{e}")
        if not tokens:
            raise SandboxError("空命令")

        base = Path(tokens[0].replace("\\", "/")).stem.lower()
        if base not in self.allowed_cmds:
            raise SandboxError(
                f"命令 {tokens[0]} 不在白名单内。允许：{sorted(self.allowed_cmds)}"
            )

        for raw in self._extract_path_args(cmd):
            target = Path(raw)
            probe = (target if target.is_absolute() else self.root / raw)
            try:
                probe = probe.resolve()
            except (ValueError, OSError):
                continue
            if self.scope == "workspace":
                if not self.is_inside(probe):
                    raise SandboxError(f"命令参数指向工作区之外，已拒绝：{raw}")
            else:
                # 电脑作用域：参数可以出工作区，但系统保护目录仍然不碰
                if self.is_protected_write(probe):
                    raise SandboxError(f"命令参数指向系统保护目录，已拒绝：{raw}")

        return tokens

    def safe_run(
        self, cmd: str, timeout: int = 30, _cancel: Optional[Any] = None
    ) -> Dict[str, Any]:
        """在工作区内执行单条白名单命令；支持取消令牌与超时杀进程树。

        不能用 subprocess.run + communicate()：那会一直阻塞到命令结束，
        「停止」按钮就只是摆设。这里改成 Popen + 轮询，每 0.2s 检查一次
        取消/超时，杀的时候用 taskkill /F /T（Windows）整棵树带走。
        """
        try:
            self.check_command(cmd)
        except SandboxError as e:
            return {"ok": False, "output": "", "error": str(e)}

        if _cancel is not None and getattr(_cancel, "is_cancelled", False):
            return {"ok": False, "output": "", "error": f"已取消：{_cancel.reason}",
                    "_cancelled": True}

        import platform

        timeout = max(1, min(int(timeout), 600))
        env = os.environ.copy()
        # 让 "python"/"pip" 解析到 agent 自己所在的解释器，避免和系统 PATH 不一致
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        popen_kwargs: Dict[str, Any] = {}
        if platform.system() == "Windows":
            # 独立进程组：terminate/kill 信号才能送达整棵树
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc: Optional[subprocess.Popen] = None
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                env=env,
                **popen_kwargs,
            )
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}

        interval = 0.2
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if _cancel is not None and getattr(_cancel, "is_cancelled", False):
                    self._kill_tree(proc)
                    return {"ok": False, "output": "", "error": f"已取消：{_cancel.reason}",
                            "_cancelled": True}
                if time.monotonic() >= deadline:
                    self._kill_tree(proc)
                    return {"ok": False, "output": "",
                            "error": f"命令超时（{timeout}s），已终止", "_killed": True}
                if _cancel is not None:
                    # wait() 被唤醒说明令牌已触发，下一轮循环会走取消分支
                    _cancel.wait(interval)
                else:
                    time.sleep(interval)

            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                out, err = "", ""
            stdout = (out or "").strip()
            stderr = (err or "").strip()
            parts = [p for p in (stdout, f"[stderr]\n{stderr}" if stderr else "") if p]
            return {
                "ok": proc.returncode == 0,
                "output": "\n".join(parts)[-20000:],
                "error": None if proc.returncode == 0 else f"退出码 {proc.returncode}",
                "returncode": proc.returncode,
            }
        except Exception as e:  # 取消路径上的窗口期异常
            if proc is not None:
                self._kill_tree(proc)
            return {"ok": False, "output": "", "error": f"已取消（{type(e).__name__}）",
                    "_cancelled": True}

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """杀掉整棵进程树；Windows 用 taskkill /T，Unix 优先 psutil、退回 killpg。"""
        import platform

        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                try:
                    import psutil  # type: ignore

                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except ImportError:
                    import signal

                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    # ---------- 快照 ----------
    def snapshot(
        self, max_files: int = 2000, max_bytes: int = 50 * 1024 * 1024
    ) -> Optional[Path]:
        """把工作区打成 zip 快照，失败/回滚时用。"""
        self.check_writable()
        snap_dir = self.root / ".agent" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"snap-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.zip"

        total, count = 0, 0
        try:
            with zipfile.ZipFile(snap_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in self.root.rglob("*"):
                    if not path.is_file():
                        continue
                    if ".agent" in path.relative_to(self.root).parts:
                        continue
                    size = path.stat().st_size
                    if size > 5 * 1024 * 1024:
                        continue
                    total += size
                    if total > max_bytes or count >= max_files:
                        break
                    zf.write(path, path.relative_to(self.root))
                    count += 1
        except Exception:
            return None
        return snap_path

    def rollback(self, snap_path: str | os.PathLike[str]) -> Dict[str, Any]:
        """从快照恢复（覆盖同名文件，不删除快照之后新建的文件）。"""
        self.check_writable()
        path = Path(snap_path)
        if not path.is_file():
            return {"ok": False, "output": "", "error": f"快照不存在：{path}"}
        try:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(self.root)
        except Exception as e:
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "output": f"已从 {path.name} 恢复", "error": None}
