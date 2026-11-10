"""T3 自检脚本（临时，不参与交付）。"""
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import runner
from builtin_tools import build_registry
from planner import Planner
from sandbox import Sandbox
from skills import crystallize, load_skills, register_skills, save_skill
from task_memory import MemoryStore
from tracing import list_traces, read_trace, render_events


class FakeRouter:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.prompts = []

    @property
    def clients(self):
        return [self]

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        if self.i >= len(self.script):
            return {"content": "（脚本结束）", "model": "mock"}
        c = self.script[self.i]
        self.i += 1
        return {"content": c, "model": "mock"}


def section(t):
    print(f"\n===== {t} =====")


# ---------------- 1. 长期记忆 ----------------
section("1. 长期记忆：检索排序 + 脱敏")
ws = Path(tempfile.mkdtemp())
store = MemoryStore(ws / ".agent" / "memory.db")
store.remember(
    "统计工作区 Python 代码行数并写入 stats.md",
    answer="已写入 stats.md",
    plan=["列出 py 文件", "统计行数", "写入 stats.md"],
    trace=[
        {"step": 1, "tool": "list_dir", "args": {"path": "."}, "ok": True},
        {"step": 2, "tool": "write_file",
         "args": {"path": "stats.md", "content": "SECRET_CONTENT_SHOULD_NOT_BE_STORED"}, "ok": True},
    ],
    artifacts=["stats.md"], ok=True,
)
store.remember("在小红书搜索咖啡并总结", answer="7 家店", plan=["搜索", "读详情"],
               trace=[{"step": 1, "tool": "xhs_search", "args": {"keyword": "咖啡"}, "ok": True}], ok=True)
store.remember("给所有 .py 文件补 docstring", answer="补完了",
               trace=[{"step": 1, "tool": "read_file", "args": {"path": "a.py"}, "ok": False,
                       "error": "文件不存在"}], ok=False, stop_reason="too_many_failures")

hits = store.recall("统计 python 代码行数", k=3)
print(" 召回：", [(h.goal[:20], round(h.score, 3)) for h in hits])
assert hits and "Python 代码行数" in hits[0].goal, "相似任务没有排在第一"
print(" 经验卡示例：\n   " + hits[0].to_card(1).replace("\n", "\n   "))

raw_db = (ws / ".agent" / "memory.db").read_bytes()
assert b"SECRET_CONTENT_SHOULD_NOT_BE_STORED" not in raw_db, "文件正文被写进了记忆库！"
print(" 脱敏检查：正文未落库 OK")
print(" 统计：", {k: v for k, v in store.stats().items() if k in ("total", "ok", "failed")})

# ---------------- 2. 规划解析 ----------------
section("2. Planner 解析：JSON / 全角引号 / 编号列表兜底")
p = Planner(gateway=None, max_steps=5)
steps, crit = p._parse_plan('{"steps":["a","b"],"success_criteria":"文件存在"}')
print(" 标准 JSON：", steps, crit)
assert steps == ["a", "b"]
steps, _ = p._parse_plan('好的。\n{“steps”:[“读取文件”,“写入结果”],“success_criteria”:“ok”}')
print(" 全角引号：", steps)
assert steps == ["读取文件", "写入结果"], steps
steps, _ = p._parse_plan("1. 先列目录\n2. 再统计行数\n3) 写入报告")
print(" 编号兜底：", steps)
assert steps == ["先列目录", "再统计行数", "写入报告"], steps

# ---------------- 3. 技能 ----------------
section("3. 技能：从轨迹结晶 → 注册 → 调用")
skill_ws = Path(tempfile.mkdtemp())
reg = build_registry(Sandbox(skill_ws))
trace = [
    {"step": 1, "tool": "write_file", "args": {"path": "notes.md", "content": "hello"}, "ok": True},
    {"step": 2, "tool": "read_file", "args": {"path": "notes.md"}, "ok": True},
    {"step": 3, "tool": "list_dir", "args": {"path": "."}, "ok": False, "error": "x"},
]
sk = crystallize("make_notes", "帮我生成「notes.md」这个文件并读回来", trace)
print(" 技能名：", sk.tool_name, "| 参数：", sk.params)
assert sk.params.get("input", {}).get("required"), "没有识别出 input 参数"
assert sk.steps[0]["args"]["path"] == "{input}"
save_skill(skill_ws / ".agent", sk)
loaded = load_skills(skill_ws / ".agent")
print(" 载入：", [s.name for s in loaded])
report = register_skills(reg, loaded, root=skill_ws / ".agent")
print(" 注册：", report)
r = reg.invoke("skill_make_notes", {"input": "out.md"})
print(" 调用结果：", r["ok"], (r["output"] or "").splitlines()[0])
assert r["ok"] and (skill_ws / "out.md").is_file()
r = reg.invoke("skill_make_notes", {})
print(" 缺参处理：", r["ok"], "|", (r["error"] or "")[:50])
assert not r["ok"] and "input" in (r["error"] or "")
print(" 调用次数：", load_skills(skill_ws / ".agent")[0].uses)

# ---------------- 4. 自主闭环（含重规划） ----------------
section("4. 自主闭环：执行失败 → 校验不通过 → 重规划 → 成功")
auto_ws = Path(tempfile.mkdtemp())
script = [
    '{"steps":["读取 missing.txt 并总结内容"],"success_criteria":"拿到文件内容"}',
    'TOOL_CALL: {"tool":"read_file","args":{"path":"missing.txt"}}',
    "读取失败，文件不存在。",
    '{"ok":false,"reason":"没有拿到任何文件内容","advice":"先创建文件再读取"}',
    '{"steps":["在工作区创建 a.txt，内容写 hello"],"success_criteria":"a.txt 存在且内容正确"}',
    'TOOL_CALL: {"tool":"write_file","args":{"path":"a.txt","content":"hello"}}',
    "已创建 a.txt，内容为 hello。",
    '{"ok":true,"reason":"工具返回写入成功，文件已存在"}',
    "已创建 a.txt（内容 hello），任务完成。",
]
runner.build_api_router = lambda: FakeRouter(script)

opts = runner.RunOptions(workspace=str(auto_ws), mode="api", auto_yes=True,
                         autonomous=True, max_plan_steps=4, max_replans=1,
                         use_memory=True, use_critic=True)
result = runner.execute("创建 a.txt 并确认内容", opts)
print(" ok:", result["ok"], "| stop:", result["stop_reason"], "| replans:", result["replans"])
print(" plan:", result["plan"])
print(" steps:", [(s["index"], s["ok"], s.get("superseded", False), s["step"][:18])
                  for s in result["step_results"]])
print(" artifacts:", result["artifacts"])
print(" answer:", result["answer"])
assert result["ok"] and result["replans"] == 1, "重规划后仍判定失败"
assert (auto_ws / "a.txt").read_text(encoding="utf-8") == "hello"
assert "a.txt" in result["artifacts"]
print(" 轨迹：", list_traces(auto_ws / ".agent")[0]["task_id"])
print(render_events(read_trace(auto_ws / ".agent", result["task_id"]))[:800])
print(" 记忆已写入：", MemoryStore(auto_ws / ".agent" / "memory.db").stats()["total"])

# ---------------- 5. 校验两次都不过 → 诚实失败 ----------------
section("5. 重规划次数用尽 → 必须报未完成")
fail_ws = Path(tempfile.mkdtemp())
fail_script = [
    '{"steps":["读取 missing.txt"],"success_criteria":"有内容"}',
    'TOOL_CALL: {"tool":"read_file","args":{"path":"missing.txt"}}',
    "读不到。",
    '{"ok":false,"reason":"没有任何观测证据","advice":"换个文件"}',
    '{"steps":["再读一次 missing.txt"],"success_criteria":"有内容"}',
    'TOOL_CALL: {"tool":"read_file","args":{"path":"missing.txt"}}',
    "还是读不到。",
    '{"ok":false,"reason":"仍然没有证据","advice":"无"}',
]
runner.build_api_router = lambda: FakeRouter(fail_script)
opts2 = runner.RunOptions(workspace=str(fail_ws), mode="api", auto_yes=True,
                          autonomous=True, max_plan_steps=4, max_replans=1)
r2 = runner.execute("读取 missing.txt", opts2)
print(" ok:", r2["ok"], "| stop:", r2["stop_reason"])
print(" answer:", r2["answer"].replace("\n", " / "))
assert not r2["ok"] and r2["stop_reason"] == "critic_rejected"

# ---------------- 6. 定时任务 ----------------
section("6. 定时任务：到期计算")
from scheduler import Scheduler

sch_ws = Path(tempfile.mkdtemp())
sch = Scheduler(runner.RunOptions(workspace=str(sch_ws)))
sch.add("hourly", "每小时检查一次", every_minutes=60)
sch.add("daily", "每天九点半汇总", at="09:30")
print(" 配置：", sch.config_path.read_text(encoding="utf-8").replace("\n", " ")[:160])
assert not sch.due(), "刚添加的任务不应立即到期"
state = sch.load_state()
state["hourly"] = (datetime.now() - timedelta(minutes=61)).isoformat(timespec="seconds")
sch.save_state(state)
due = [j["id"] for j in sch.due()]
print(" 到期：", due)
assert due == ["hourly"], due
print(" 描述：", [(d["id"], d["next_run"], d["due"]) for d in sch.describe()])

# ---------------- 7. HTTP 服务 ----------------
section("7. HTTP 服务：/run /status /events /memory /skills")
import server as server_mod


def fake_execute(goal, opts, emit=None, should_stop=None, control=None, gate=None, **kw):
    for ev in (
        {"event": "task_start", "goal": goal},
        {"event": "plan", "steps": ["一步"], "criteria": "ok"},
        {"event": "step_start", "index": 1, "total": 1, "step": "一步"},
        {"event": "done", "ok": True, "answer": "假执行完成", "stop_reason": "finished"},
    ):
        if emit:
            emit(ev)
    return {"ok": True, "answer": "假执行完成", "stop_reason": "finished", "trace": [], "steps": 0}


server_mod.execute = fake_execute
httpd, manager, _ = server_mod.serve(
    runner.RunOptions(workspace=str(sch_ws), autonomous=True), port=8791, block=False
)
time.sleep(0.5)


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:8791{path}", timeout=5) as r:
        return r.status, r.read().decode("utf-8")


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:8791{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


print(" /health:", get("/health"))
status, body = post("/api/run", {"goal": "测试任务", "autonomous": True})
print(" /api/run:", status, body)
tid = body["task_id"]
time.sleep(1.0)
print(" /api/task:", get(f"/api/task?id={tid}")[1][:180])
print(" /api/tasks:", get("/api/tasks")[1][:160])
print(" /api/memory:", get("/api/memory")[1][:120])
print(" /api/skills:", get("/api/skills")[1][:160])
try:
    post("/api/run", {"goal": ""})
    raise AssertionError("空 goal 应该 400")
except urllib.error.HTTPError as e:
    print(" 空 goal 校验：", e.code)

with urllib.request.urlopen(f"http://127.0.0.1:8791/api/stream?id={tid}", timeout=5) as r:
    events = []
    for raw in r:
        line = raw.decode("utf-8").strip()
        if line.startswith("data:"):
            events.append(json.loads(line[5:])["event"])
        if "task_end" in events:
            break
print(" SSE 事件：", events)

httpd.shutdown()
print("\n全部 T3 断言通过")
