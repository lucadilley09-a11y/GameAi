"""The coding task bank (plan: Phase 4). A task: {id, lang ('js'|'py'|'lua'), prompt, file, starter: {path: text},
tests: {path: text} (visible), hidden: {path: text} (never shown to the AI), reference (a correct solution, used only
to check the task), source ('seed'|'teacher'), split ('val'|'train')}.

    data/coding/tasks_val.jsonl     the hand-written seed tasks (never trained on) plus a share of the teacher's
    data/coding/tasks_train.jsonl   the teacher's tasks, checked like the seed ones, for self-teaching and training

Every task is checked before it's used: the reference must pass all visible and hidden tests, and the starter alone
must fail at least one hidden test (so the tests really test something).
"""
import json, os

from .seed_tasks import SEED

DIR = os.path.join("data", "coding")
VAL_SHARE = 0.15   # of the teacher's tasks also kept for validation (the seed tasks are always validation)


def path(split): return os.path.join(DIR, f"tasks_{split}.jsonl")


def load(split):
    out = [dict(t, source="seed", split="val") for t in SEED] if split == "val" else []
    if os.path.exists(path(split)):
        out += [json.loads(l) for l in open(path(split), encoding="utf-8") if l.strip()]
    return out


def add(task):
    os.makedirs(DIR, exist_ok=True)
    with open(path(task["split"]), "a", encoding="utf-8") as f: f.write(json.dumps(task, ensure_ascii=False) + "\n")


def text_of(task):
    """the text the leak guard compares (the prompt and the tests)"""
    return task["prompt"] + "\n" + "\n".join(task["tests"].values())


def check_task(task):
    """-> list of problems (empty = the task is good)"""
    from .verify import run_tests
    from ..repair import debugger
    probs = []
    for k in ("id", "lang", "prompt", "file", "tests", "hidden", "reference"):
        if not task.get(k): probs.append(f"missing {k}")
    if probs: return probs
    if task["lang"] not in ("js", "py", "lua"): return [f"unknown language {task['lang']}"]
    r = run_tests(task, task["reference"])
    if not debugger.ok(r) or not r["tests"]:
        probs.append("the reference solution doesn't pass its own tests:\n" + debugger.report(r))
    if len([t for t in r["tests"] if t.get("file") in task["hidden"]]) < 2: probs.append("needs at least 2 hidden tests")
    starter = task.get("starter", {}).get(task["file"], "")
    stub = run_tests(task, starter or {"js": "export {};\n", "py": "\n", "lua": "return {}\n"}[task["lang"]])
    if all(t.get("passed") for t in stub["tests"] if t.get("file") in task["hidden"]) and not stub["load_errors"]:
        probs.append("the hidden tests pass even without a solution, so they test nothing")
    return probs
