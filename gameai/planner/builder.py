"""Building a project one tested feature at a time (plan: Phase 4, project planner).

    plan = builder.make_plan(model, "a platformer where you collect gems and dodge bats", "my_game", lang="js")
    builder.build_next(model, "my_game")     # one feature: written, checked, tested, applied (or retried, or blocked)
    builder.build_all(model, "my_game")      # until done, blocked or stopped

model(messages) -> reply text (the AI with its code skill; any function works, which is how the tests run).

For each feature the AI sees: the project summary (goal, design, decisions, what's done), the project map, matching
checked components from the library, the feature and when it counts as done, and (when retrying) what went wrong.
It answers with structured edits (repair/edits.py) that add the feature AND its tests. A change is applied only if:
    - it doesn't touch any finished feature's tests (those are regression tests now),
    - it adds or changes at least one test file for this feature,
    - the all-or-nothing change passes (repair/atomic.py: nothing that worked before breaks),
    - and this feature's own tests pass.
After MAX_TRIES failed tries the feature is marked blocked with the last report, and the user is asked.
"""
import json, os, re

from . import project as P, components
from ..repair import atomic, debugger, repomap

MAX_TRIES = 4
LANG_NAME = {"js": "JavaScript", "py": "Python", "lua": "Lua"}
TEST_NAMING = {"js": "name.test.js, using test(name, fn) and expect(x).toBe(y)",
               "py": "test_name.py, with test_* functions and plain assert",
               "lua": "name_test.lua, using test('name', function() expect(x).toBe(y) end)"}


# ---------- planning ----------
def plan_messages(request, lang="js"):
    ex = {"name": "Gem Cave", "goal": "a short platformer: collect every gem while dodging bats",
          "design": {"mechanics": "run, jump, collect gems; bats fly in waves", "controls": "arrows/WASD, space to jump",
                     "rules": "3 lives; all gems = win; touching a bat = lose a life", "look": "pixel art cave, blue and purple", "sound": "chiptune jumps and pickups"},
          "features": [{"id": "f1", "title": "Level and player", "description": "a cave level and a player who runs and jumps",
                        "acceptance": ["the player stands on the floor", "jumping goes up then comes down"], "depends": []},
                       {"id": "f2", "title": "Gems", "description": "gems to collect; all gems wins", "acceptance": ["touching a gem removes it and adds 1 to the score",
                        "collecting the last gem wins"], "depends": ["f1"]}]}
    return [{"role": "system", "content": "You plan small games as a design and an ordered list of features that can each be built and tested on its own. Reply with JSON only."},
            {"role": "user", "content": f"Plan this {LANG_NAME[lang]} game: {request}\n\nKeep features small (each one testable by itself), in build order, "
             "with dependencies. Every feature needs acceptance criteria that a test can check. Reply in exactly this shape:\n" + json.dumps(ex, indent=1)}]


def parse_json(text):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text or "", re.S)
    raw = m.group(1) if m else (re.search(r"[\[{].*[\]}]", text or "", re.S) or [None])[0]
    if raw is None: return None
    try: return json.loads(raw)
    except ValueError: return None


def make_plan(model, request, project, lang="js", tries=3):
    """the AI plans the game; a plan with problems is asked for again with the problems listed"""
    msgs, probs = plan_messages(request, lang), []
    for _ in range(tries):
        plan = parse_json(model(msgs))
        if not isinstance(plan, dict): probs = ["the reply wasn't the JSON plan"]
        else:
            plan["lang"], plan["request"] = lang, request
            probs = P.check(plan)
            if not probs: return P.create(project, plan)
        msgs = msgs + [{"role": "assistant", "content": "(plan)"}, {"role": "user", "content": "Problems with that plan:\n- " + "\n- ".join(probs) + "\nReply with the corrected JSON."}]
    raise P.PlanError("couldn't make a sound plan: " + "; ".join(probs))


# ---------- building ----------
EDIT_HELP = """Reply with a JSON list of edits in one ```json code block. Each edit is one of:
  {"file": "path", "create": "full text of a new file"}
  {"file": "path", "search": "exact text that appears once", "replace": "new text"}
  {"file": "path", "symbol": "functionName or Class.method", "replace": "the whole new definition"}
  {"file": "path", "insert_after": "exact text", "text": "text to add"}
Use small edits for existing files; create files that don't exist yet."""


def feature_messages(project, plan, feat, report=None):
    lang = plan.get("lang", "js")
    files = repomap.read_dir(project)
    m = repomap.build(files)
    focus = [p for f in plan["features"] if f["id"] in feat["depends"] for p in f.get("files", []) + f.get("tests", [])]
    found = components.search(feat["title"] + " " + feat["description"], lang=lang)
    parts = [P.summary(plan),
             "Project files (signatures):\n" + (repomap.render(m, focus=focus) or "(empty: this is the first feature)"),
             components.prompt_block(found),
             f"Build this feature now: {feat['title']}: {feat['description']}\nIt is done when:\n" + "\n".join(f"- {a}" for a in feat["acceptance"]),
             f"Also add tests for it ({TEST_NAMING[lang]}) that check every point above. Don't change these finished features' tests: "
             + (", ".join(sorted(P.protected_tests(plan))) or "(none yet)") + ".",
             EDIT_HELP]
    help_ = [n[6:] for n in feat.get("notes", []) if n.startswith("user: ")]
    if help_: parts.append("The user's guidance for this feature: " + " / ".join(help_))
    if report: parts.append("Your last try was not applied. What went wrong:\n" + report.strip()[:3000])
    return [{"role": "system", "content": f"You build {LANG_NAME[lang]} games one feature at a time, with tests, using small checked edits."},
            {"role": "user", "content": "\n\n".join(p for p in parts if p)}], [c["name"] for c in found]


def _strategy(lang):
    try:
        from ..coding.compare import strategy
        return strategy(lang)
    except Exception:
        return "resample"


def build_next(model, project, log=print):
    """build the next feature -> ('done' | 'blocked' | 'finished', feature or None, last report)"""
    plan = P.load(project)
    if not plan: raise P.PlanError("this project has no plan yet")
    feat = P.next_feature(plan)
    if not feat: return "finished", None, ""
    lang = plan.get("lang", "js")
    feat["status"] = "doing"; P.save(project, plan)
    protected, report, kind = P.protected_tests(plan), None, None
    for attempt in range(1, MAX_TRIES + 1):
        feat["attempts"] += 1
        # instructions (wrong format, no tests, touched regression tests) always go back; test and checker failures go back
        # only where repairing measurably beats a fresh try (coding/compare.py)
        use_report = report if kind == "instruction" or _strategy(lang) == "repair" else None
        msgs, used = feature_messages(project, plan, feat, use_report)
        edits = parse_json(model(msgs))
        if not isinstance(edits, list) or not all(isinstance(e, dict) and e.get("file") for e in edits):
            report, kind = "The reply wasn't a JSON list of edits.", "instruction"; log(f"  try {attempt}: {report}"); continue
        touched = {e["file"] for e in edits}
        hit = sorted(touched & protected)
        if hit:
            report, kind = f"Those edits change finished features' tests ({', '.join(hit)}); they are regression tests and must stay as they are.", "instruction"
            log(f"  try {attempt}: refused, it changed regression tests"); continue
        tests = sorted(f for f in touched if debugger.is_test(f) and not any(e.get("delete") for e in edits if e["file"] == f))
        if not tests:
            report, kind = f"No tests for this feature: add a test file ({TEST_NAMING[lang]}).", "instruction"; log(f"  try {attempt}: no tests"); continue
        res = atomic.change(project, edits)
        if not res["applied"]:
            report, kind = res["report"], "failure"; log(f"  try {attempt}: not applied ({len(res['problems'])} problem(s))"); continue
        r = debugger.run(project)  # the feature's own tests must pass too (the change check only stops things getting worse)
        mine = [t for t in r["tests"] if t.get("file") in tests]
        if not mine or not all(t.get("passed") for t in mine) or any(e["file"] in tests for e in r["load_errors"]):
            atomic.rollback(project)
            vis = {**r, "tests": mine, "load_errors": [e for e in r["load_errors"] if e["file"] in tests],
                   "passed": sum(1 for t in mine if t.get("passed")), "failed": sum(1 for t in mine if not t.get("passed"))}
            report, kind = "The feature's own tests fail:\n" + debugger.report(vis), "failure"
            log(f"  try {attempt}: its own tests fail; undone"); continue
        feat.update(status="done", tests=tests, files=sorted(f for f in touched if not debugger.is_test(f)), finished=res["history"])
        feat["notes"].append(f"done on try {attempt}")
        P.save(project, plan)
        if used: components.used(used)
        added = components.harvest(project, plan, feat)
        log(f"done: {feat['title']} (try {attempt})" + (f"; new components: {', '.join(added)}" if added else ""))
        return "done", feat, ""
    feat["status"] = "blocked"; feat["notes"].append((report or "")[:1500])
    P.save(project, plan)
    log(f"blocked: {feat['title']} after {MAX_TRIES} tries")
    return "blocked", feat, report


def build_all(model, project, log=print, stop=lambda: False):
    """build features until the project is finished, one is blocked, or stop() says so -> the last status"""
    while not stop():
        status, feat, _ = build_next(model, project, log)
        if status != "done": return status
    return "stopped"


def unblock(project, fid, note=""):
    """after the user helps: try a blocked feature again (its notes keep the history)"""
    plan = P.load(project)
    f = P.feature(plan, fid)
    if f and f["status"] == "blocked":
        f["status"] = "todo"
        if note: f["notes"].append("user: " + note)
        P.save(project, plan)
    return f
