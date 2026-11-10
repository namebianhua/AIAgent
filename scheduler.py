"""定时任务（T3）：按间隔或每日定点自动跑目标。

配置放在 `.agent/schedules.json`（可手改），运行状态单独放
`.agent/schedule_state.json`，避免把用户配置写花。

用法：
  python scheduler.py list
  python scheduler.py add --id daily-report --goal "统计工作区代码行数写入 stats.md" --at 09:30
  python scheduler.py add --id hourly --goal "检查 logs 目录新增错误" --every 60
  python scheduler.py run-due          # 跑一次到期的（可挂系统计划任务）
  python scheduler.py serve            # 常驻循环
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from runner import RunOptions, execute


def _now() -> datetime:
    return datetime.now()


class Scheduler:
    def __init__(
        self,
        options: Optional[RunOptions] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.options = options or RunOptions()
        root = Path(self.options.workspace).expanduser().resolve() / ".agent"
        root.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path) if config_path else root / "schedules.json"
        self.state_path = root / "schedule_state.json"
        self.log_path = root / "schedule_runs.jsonl"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 配置 ----------
    def load(self) -> List[Dict[str, Any]]:
        if not self.config_path.is_file():
            return []
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        jobs = data.get("jobs") if isinstance(data, dict) else data
        return jobs if isinstance(jobs, list) else []

    def save(self, jobs: List[Dict[str, Any]]) -> None:
        self.config_path.write_text(
            json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_state(self) -> Dict[str, str]:
        if not self.state_path.is_file():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_state(self, state: Dict[str, str]) -> None:
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- 增删 ----------
    def add(
        self,
        job_id: str,
        goal: str,
        every_minutes: Optional[int] = None,
        at: Optional[str] = None,
        autonomous: bool = True,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        if not job_id or not goal:
            raise ValueError("id 和 goal 都不能为空")
        if not every_minutes and not at:
            raise ValueError("必须指定 --every（分钟）或 --at（HH:MM）")
        if every_minutes and int(every_minutes) < 1:
            raise ValueError("--every 至少 1 分钟")
        if at:
            datetime.strptime(at, "%H:%M")  # 提前校验格式

        jobs = [j for j in self.load() if j.get("id") != job_id]
        job = {
            "id": job_id,
            "goal": goal,
            "autonomous": bool(autonomous),
            "enabled": bool(enabled),
            "created": _now().isoformat(timespec="seconds"),
        }
        if every_minutes:
            job["every_minutes"] = int(every_minutes)
        if at:
            job["at"] = at
        jobs.append(job)
        self.save(jobs)
        return job

    def remove(self, job_id: str) -> bool:
        jobs = self.load()
        kept = [j for j in jobs if j.get("id") != job_id]
        if len(kept) == len(jobs):
            return False
        self.save(kept)
        state = self.load_state()
        state.pop(job_id, None)
        self.save_state(state)
        return True

    # ---------- 调度计算 ----------
    def next_run(self, job: Dict[str, Any], last_run: Optional[datetime] = None) -> Optional[datetime]:
        now = _now()
        base = last_run or now
        if job.get("every_minutes"):
            minutes = max(1, int(job["every_minutes"]))
            return base + timedelta(minutes=minutes)
        if job.get("at"):
            hh, mm = (int(x) for x in str(job["at"]).split(":"))
            for delta in range(0, 3):
                day = (last_run or now).date() + timedelta(days=delta)
                candidate = datetime.combine(day, datetime.min.time()).replace(hour=hh, minute=mm)
                if candidate > (last_run or now):
                    return candidate
            return None
        return None

    def due(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = now or _now()
        state = self.load_state()
        out = []
        for job in self.load():
            if not job.get("enabled", True):
                continue
            last_raw = state.get(job.get("id", ""))
            last = None
            if last_raw:
                try:
                    last = datetime.fromisoformat(last_raw)
                except Exception:
                    last = None
            nxt = self.next_run(job, last)
            if nxt is None or nxt <= now:
                out.append(job)
        return out

    def describe(self) -> List[Dict[str, Any]]:
        state = self.load_state()
        now = _now()
        out = []
        for job in self.load():
            jid = job.get("id", "")
            last = None
            if state.get(jid):
                try:
                    last = datetime.fromisoformat(state[jid])
                except Exception:
                    last = None
            nxt = self.next_run(job, last)
            out.append(
                {
                    "id": jid,
                    "goal": job.get("goal", ""),
                    "every_minutes": job.get("every_minutes"),
                    "at": job.get("at"),
                    "autonomous": job.get("autonomous", True),
                    "enabled": job.get("enabled", True),
                    "last_run": state.get(jid),
                    "next_run": nxt.isoformat(timespec="seconds") if nxt else None,
                    "due": bool(nxt and nxt <= now),
                }
            )
        return out

    # ---------- 执行 ----------
    def run_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        jid = job.get("id", "?")
        opts = RunOptions(**self.options.to_dict())
        opts.autonomous = bool(job.get("autonomous", True))
        if job.get("workspace"):
            opts.workspace = job["workspace"]
        print(f"[scheduler] 运行 {jid}：{job.get('goal', '')[:60]}")

        started = _now()
        try:
            result = execute(job.get("goal", ""), opts)
            status = "done" if result.get("ok") else "failed"
        except Exception as e:
            result = {"ok": False, "answer": f"{type(e).__name__}: {e}"}
            status = "error"
            traceback.print_exc()

        state = self.load_state()
        state[jid] = started.isoformat(timespec="seconds")
        self.save_state(state)

        record = {
            "ts": _now().isoformat(timespec="seconds"),
            "id": jid,
            "goal": job.get("goal", ""),
            "status": status,
            "ok": result.get("ok"),
            "stop_reason": result.get("stop_reason"),
            "answer": (result.get("answer") or "")[:800],
            "steps": result.get("steps"),
            "duration_ms": result.get("duration_ms"),
            "task_id": result.get("task_id"),
        }
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        print(f"[scheduler] {jid} -> {status}：{(result.get('answer') or '')[:80]}")
        return record

    def tick(self) -> List[Dict[str, Any]]:
        results = []
        for job in self.due():
            results.append(self.run_job(job))
        return results


def schedule_loop(sched: Scheduler, interval: int = 30, stop_event=None) -> None:
    print(f"[scheduler] 已启动，每 {interval}s 检查一次到期任务")
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            sched.tick()
        except Exception:
            traceback.print_exc()
        time.sleep(max(5, interval))


# ---------------- CLI ----------------
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="定时任务管理（T3）")
    parser.add_argument("action", choices=["list", "add", "remove", "run", "run-due", "serve"])
    parser.add_argument("--id")
    parser.add_argument("--goal")
    parser.add_argument("--every", type=int, dest="every_minutes", help="每 N 分钟")
    parser.add_argument("--at", help="每日 HH:MM")
    parser.add_argument("--no-autonomous", action="store_true")
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--interval", type=int, default=30, help="serve 检查间隔秒数")
    parser.add_argument("-w", "--workspace", default=".")
    parser.add_argument("-m", "--mode", default="web", choices=["api", "web", "hybrid"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    opts = RunOptions(
        workspace=args.workspace,
        mode=args.mode,
        headless=args.headless,
        auto_yes=args.yes,
    )
    sched = Scheduler(opts)

    if args.action == "list":
        rows = sched.describe()
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        elif not rows:
            print("（还没有定时任务）")
        else:
            for r in rows:
                when = f"每 {r['every_minutes']} 分钟" if r["every_minutes"] else f"每天 {r['at']}"
                flag = "启用" if r["enabled"] else "停用"
                print(f"- {r['id']}  [{flag}] {when}  下次 {r['next_run']}")
                print(f"    {r['goal'][:100]}")
        return 0

    if args.action == "add":
        if not args.id or not args.goal:
            print("需要 --id 和 --goal")
            return 2
        job = sched.add(
            args.id,
            args.goal,
            every_minutes=args.every_minutes,
            at=args.at,
            autonomous=not args.no_autonomous,
            enabled=not args.disabled,
        )
        print(f"已添加：{job['id']}")
        print(f"配置文件：{sched.config_path}")
        return 0

    if args.action == "remove":
        if not args.id:
            print("需要 --id")
            return 2
        print("已删除" if sched.remove(args.id) else "没有这个任务")
        return 0

    if args.action == "run":
        jobs = [j for j in sched.load() if j.get("id") == args.id]
        if not jobs:
            print(f"没有 id={args.id} 的任务")
            return 2
        rec = sched.run_job(jobs[0])
        print(json.dumps(rec, ensure_ascii=False, indent=2) if args.json else "")
        return 0 if rec.get("status") == "done" else 1

    if args.action == "run-due":
        results = sched.tick()
        print(f"本次执行 {len(results)} 个到期任务")
        return 0

    if args.action == "serve":
        try:
            schedule_loop(sched, interval=args.interval)
        except KeyboardInterrupt:
            print("\n[scheduler] 退出")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
