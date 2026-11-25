"""Tests for the code-skill training pieces (Phase 4 session 5), with a stand-in model instead of a real one.
Needs the checkers installed (python -m gameai.checks install); skipped without them."""
import json, os, random, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gameai.checks import tools

R = []
def t(name, ok, info=None):
    ok = bool(ok); R.append(ok); print(("PASS " if ok else "FAIL ") + name)
    if not ok and info is not None: print("   ", str(info)[:1500])

def main():
    if not (tools.node() and tools.node_modules()):
        print("SKIP: checkers not installed"); return finish()
    here = os.getcwd(); work = tempfile.mkdtemp(); os.chdir(work)
    try:
        from gameai.coding import tasks as T, verify as V, selfteach as S, compare as C, teacher as TE, evaluate as E, prompts as P
        val = T.load("val"); by = {x["id"]: x for x in val}
        t("16 hand-written validation tasks in 3 languages", len(val) == 16 and {x["lang"] for x in val} == {"js", "py", "lua"})
        bad = {x["id"]: T.check_task(x) for x in val}
        t("every hand-written task is sound (reference passes, tests test something)", not any(bad.values()), {k: v for k, v in bad.items() if v})
        # prompts
        p = P.first(by["js-kit-spawn-wave"])
        t("GameKit tasks get GameKit's function list in the prompt", "spawn(kind" in p and "Tests it must pass" in p)
        t("hidden tests never appear in prompts", all(h not in P.first(x) for x in val for h in x["hidden"].values()))
        # a stand-in model: right on even tries, wrong on odd ones; given a report, it fixes the code
        def make_model():
            calls = {"n": 0}
            def chat(messages, **kw):
                calls["n"] += 1
                user = messages[-1]["content"]
                task = next(x for x in train if x["prompt"] in user)
                if "What went wrong" in user: return "```\n" + task["reference"] + "```"
                if calls["n"] % 2: return "```\n" + task["reference"].replace("+", "-", 1).replace("min(", "max(", 1).replace("#self.items + 1", "1", 1) + "```"
                return "```\n" + task["reference"] + "\n// try " + str(calls["n"]) + "\n```" if task["lang"] == "js" else "```\n" + task["reference"] + "```"
            return chat
        # practice tasks: copies of three seed tasks with new ids (prompts changed so they aren't validation tasks)
        train = [
            {"id": "train-js", "lang": "js", "split": "train", "file": "dist.js", "starter": {},
             "prompt": "Create `export function manhattan(a, b)` in dist.js returning the grid distance |dx| + |dy| between points {x, y}.",
             "reference": "export function manhattan(a, b) {\n  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);\n}\n",
             "tests": {"dist.test.js": "import { manhattan } from './dist.js';\ntest('d', () => expect(manhattan({x: 0, y: 0}, {x: 2, y: 3})).toBe(5));\n"},
             "hidden": {"dist.hidden.test.js": "import { manhattan } from './dist.js';\ntest('neg', () => expect(manhattan({x: -1, y: 4}, {x: 2, y: 0})).toBe(7));\ntest('same', () => expect(manhattan({x: 3, y: 3}, {x: 3, y: 3})).toBe(0));\n"}},
            {"id": "train-py", "lang": "py", "split": "train", "file": "ammo.py", "starter": {},
             "prompt": "Make `reload(clip, reserve, size)` in ammo.py: fill the clip up to size from the reserve; return (clip, reserve).",
             "reference": "def reload(clip, reserve, size):\n    take = min(size - clip, reserve)\n    return clip + take, reserve - take\n",
             "tests": {"test_ammo.py": "from ammo import reload\n\ndef test_fill():\n    assert reload(2, 30, 10) == (10, 22)\n"},
             "hidden": {"test_ammo_hidden.py": "from ammo import reload\n\ndef test_short():\n    assert reload(0, 3, 6) == (3, 0)\n\ndef test_full():\n    assert reload(6, 9, 6) == (6, 9)\n"}},
            {"id": "train-lua", "lang": "lua", "split": "train", "file": "stack.lua", "starter": {},
             "prompt": "Build stack.lua returning M with M.new() giving an object with :push(v), :pop() (nil when empty) and :size().",
             "reference": "local M = {}\nM.__index = M\nfunction M.new() return setmetatable({items = {}}, M) end\nfunction M:push(v) self.items[#self.items + 1] = v end\nfunction M:pop() return table.remove(self.items) end\nfunction M:size() return #self.items end\nreturn M\n",
             "tests": {"stack_test.lua": "local S = require('stack')\ntest('push', function() local s = S.new(); s:push(1); expect(s:size()).toBe(1) end)\n"},
             "hidden": {"stack_hidden_test.lua": "local S = require('stack')\ntest('pop order', function() local s = S.new(); s:push(1); s:push(2); expect(s:pop()).toBe(2); expect(s:size()).toBe(1) end)\ntest('empty', function() expect(S.new():pop()).toBe(nil) end)\n"}},
        ]
        summary = S.run(make_model(), train, n=6, log=lambda *a: None)
        first, rep, pairs = S.rows("first"), S.rows("repair"), S.rows("pairs")
        t("self-teaching keeps checked solutions, at most 4 per task", first and all(sum(r["task"] == x["id"] for r in first) <= S.CAP for x in train), summary)
        t("failed tries become repair examples with what went wrong", rep and all("What went wrong" in r["prompt"] for r in rep), len(rep))
        t("preference pairs: same prompt, a checked and a failed solution", pairs and all(r["chosen"] != r["rejected"] for r in pairs))
        t("every training answer is a checked solution", all(V.verify(next(x for x in train if x["id"] == r["task"]), r["answer"])["accepted"] for r in first + rep))
        n_before = len(first)
        S.run(make_model(), train, n=6, log=lambda *a: None)
        t("running again doesn't duplicate rows", len(S.rows("first")) <= n_before + 3 * S.CAP and len({r["hash"] for r in S.rows("first")}) == len(S.rows("first")))
        S.run(make_model(), [dict(train[0], split="val")], n=2, log=lambda *a: None)
        t("validation tasks are never practised on", not any(r["task"] == "val" for r in S.rows("first")))
        leak = dict(by["py-save-format"], id="leak", split="train")
        before = len(S.rows("first")); S.run(lambda m, **k: "```\n" + leak["reference"] + "```", [leak], n=2, log=lambda *a: None)
        t("a training task that copies a validation task is skipped", len(S.rows("first")) == before)
        # lora_train picks the rows up
        from gameai.lora_train import chat_rows
        t("the adapter trainer reads the code and repair rows", len(chat_rows("code")) == len(S.rows("first")) and len(chat_rows("code_repair")) == len(S.rows("repair")))
        # repair vs resample, budget-matched
        always_wrong_first = lambda m, **k: "```\n" + (train_by[m[-1]["content"]]["reference"] if "What went wrong" in m[-1]["content"] else "broken(") + "```"
        train_by = {}
        class Map(dict):
            def __getitem__(self, user): return next(x for x in val if x["prompt"] in user)
        train_by = Map()
        res = C.run(always_wrong_first, log=lambda *a: None, task_list=[by["js-clamp-health"], by["lua-count" if False else "lua-fix-count"]])
        t("repair wins when first tries always fail but fixes work", res["use"] == {"js": "repair", "lua": "repair"}, res)
        always_right = lambda m, **k: "```\n" + train_by[m[-1]["content"]]["reference"] + "```"
        res = C.run(always_right, log=lambda *a: None, task_list=[by["py-fix-wrap"]])
        t("a tie goes to plain retries", res["use"] == {"py": "resample"}, res)
        t("strategy is read back", C.strategy("py") == "resample" and C.strategy("lua") == "resample")
        # evaluation for the trainer
        r = E.pass_rate(lambda m: "```\n" + train_by[m[-1]["content"]]["reference"] + "```")
        t("validation pass rate: references solve every task", r["pass_rate"] == 1.0 and r["tasks"] == 16, r)
        r = E.pass_rate(lambda m: "```\nnothing\n```", repair=True)
        t("repair validation uses the 3 bug-fix tasks", r["tasks"] == 3 and r["pass_rate"] == 0, r)
        # the teacher writing tasks: a good task is kept, a broken or duplicate one isn't
        good = json.loads(json.dumps(by["js-score-combo"]))
        good.update(prompt="Write `export function streak(days)` in streak.js: days is a sorted list of day numbers; return the longest run of consecutive days.",
                    file="streak.js", reference="export function streak(days) {\n  let best = 0, run = 0, last = null;\n  for (const d of days) { run = last !== null && d === last + 1 ? run + 1 : 1; best = Math.max(best, run); last = d; }\n  return best;\n}\n",
                    tests={"streak.test.js": "import { streak } from './streak.js';\ntest('run', () => expect(streak([1, 2, 3, 7])).toBe(3));\n"},
                    hidden={"streak.hidden.test.js": "import { streak } from './streak.js';\ntest('empty', () => expect(streak([])).toBe(0));\ntest('gap', () => expect(streak([1, 3, 4, 9, 10, 11, 12])).toBe(4));\n"})
        broken = dict(good, reference="export function streak(d) { return 1; }\n")
        answers = iter([json.dumps({k: broken[k] for k in ("prompt", "file", "starter", "tests", "hidden", "reference")}),
                        "not json at all",
                        json.dumps({k: good[k] for k in ("prompt", "file", "starter", "tests", "hidden", "reference")}),
                        json.dumps({k: good[k] for k in ("prompt", "file", "starter", "tests", "hidden", "reference")})] * 3)
        class AlwaysJs(random.Random):
            def choice(self, seq): return 'js' if 'js' in seq else seq[0]
        kept = TE.make_tasks(lambda m, **k: next(answers), 2, log=lambda *a: None, rng=AlwaysJs(1))
        all_tasks = T.load("train") + [x for x in T.load("val") if x.get("source") == "teacher"]
        t("the teacher's tasks: only sound, new ones are kept", kept == 1 and len(all_tasks) == 1 and all_tasks[0]["file"] == "streak.js", (kept, len(all_tasks)))
        t("a wrong reference is rejected by the task check", T.check_task(broken))
    finally:
        os.chdir(here); shutil.rmtree(work, ignore_errors=True)
    finish()

def finish():
    print(f"{sum(R)}/{len(R)} passed")
    if not all(R): sys.exit(1)

main()
