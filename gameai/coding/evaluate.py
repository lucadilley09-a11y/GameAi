"""The code skill's validation score: validation tasks solved on the first try, checked in the sandbox like everything
else. For the repair skill: the tasks that start from buggy code, given that code and what its tests said."""
from . import prompts, tasks as T
from .verify import verify, run_tests, _visible_only
from ..repair import debugger


def repair_case(task):
    """(buggy code, visible-test report) for a task that starts from buggy code, else None"""
    code = task.get("starter", {}).get(task["file"])
    if not code: return None
    r = _visible_only(task, run_tests(task, code))
    return (code, debugger.report(r)) if not debugger.ok(r) else None


def pass_rate(generate, repair=False, task_list=None):
    """generate(messages) -> text. -> {pass_rate, tasks, by_language}"""
    task_list = task_list or T.load("val")
    ok, n, by = 0, 0, {}
    for t in task_list:
        case = repair_case(t) if repair else None
        if repair and not case: continue
        msgs = prompts.messages(t, *case) if case else prompts.messages(t)
        good = verify(t, generate(msgs))["accepted"]
        ok += good; n += 1
        b = by.setdefault(t["lang"], [0, 0]); b[0] += good; b[1] += 1
    return {"pass_rate": round(ok / max(1, n), 4), "tasks": n, "by_language": {k: f"{a}/{c}" for k, (a, c) in by.items()}}
