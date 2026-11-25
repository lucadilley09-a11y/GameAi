"""Tests for the repair tools (Phase 4 session 3). The debugger and atomic changes need the checkers installed
(python -m gameai.checks install); those parts are skipped without them."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gameai.repair import repomap, edits as E, debugger, atomic
from gameai.checks import tools

R = []
def t(name, ok, info=None):
    ok = bool(ok); R.append(ok); print(("PASS " if ok else "FAIL ") + name)
    if not ok and info is not None: print("   ", str(info)[:1500])

GAME = {
 "src/util.js": "export function clamp(v, lo, hi) {\n  return Math.max(lo, Math.min(hi, v));\n}\n",
 "src/player.js": "import { clamp } from './util.js';\n\nexport class Player {\n  constructor() {\n    this.x = 0;\n    this.speed = 3;\n  }\n  move(dx) {\n    this.x = clamp(this.x + dx * this.speed, 0, 100);\n    return this.x;\n  }\n}\n",
 "src/player.test.mjs": "import { Player } from './player.js';\ntest('moves right', () => { const p = new Player(); expect(p.move(2)).toBe(6); });\ntest('stops at the wall', () => { const p = new Player(); p.x = 99; expect(p.move(5)).toBe(100); });\n",
 "index.html": "<script type=module src=\"src/player.js\"></script>\n",
}

def write(d, files):
    for p, s in files.items():
        dest = os.path.join(d, *p.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True); open(dest, "w", encoding="utf-8").write(s)

def main():
    # ---- map
    m = repomap.build(GAME)
    txt = repomap.render(m, focus=["src/player.js"])
    t("map lists classes, methods, signatures and imports", "export class Player" in txt and "move(dx)" in txt and "imports: ./util.js" in txt, txt)
    t("map: focus file first, connected file next", txt.splitlines()[0].startswith("src/player.js") and "src/util.js" in txt.splitlines()[txt.splitlines().index(next(l for l in txt.splitlines() if l.startswith("src/util.js")))], txt)
    t("find gives the definition", repomap.find(m, "clamp")[0]["signature"] == "function clamp(v, lo, hi)")
    kit = repomap.build({"gamekit.d.ts": open(os.path.join(os.path.dirname(debugger.KIT), "gamekit.d.ts"), encoding="utf-8").read()})
    t("map reads GameKit's types (Game.spawn)", repomap.find(kit, "spawn") and "spawn(kind" in repomap.find(kit, "spawn")[0]["signature"])
    t("Python, Lua and C# are mapped", {s["name"] for f in repomap.build({"a.py": "class A:\n    def go(self, n):\n        pass\n", "m.lua": "function M.spawn(x) end\n",
        "G.cs": "public class E { public void Hit(int d) {} }\n"}).values() for s in f["symbols"]} >= {"A", "A.go", "M.spawn", "E", "E.Hit"})
    # ---- edits
    new, done = E.apply(GAME, [{"file": "src/player.js", "search": "this.speed = 3;", "replace": "this.speed = 4;"}])
    t("search edit applies", "this.speed = 4;" in new["src/player.js"] and GAME["src/player.js"] != new["src/player.js"])
    try: E.apply(GAME, [{"file": "src/player.js", "search": "this.x", "replace": "this.pos"}]); t("an ambiguous anchor is refused", False)
    except E.EditError as e: t("an ambiguous anchor is refused (with line numbers)", "appears 4 times" in str(e) and "lines" in str(e), e)
    new, _ = E.apply(GAME, [{"file": "src/player.js", "search": "this.x", "replace": "this.pos", "count": 4}])
    t("an expected count allows it", new["src/player.js"].count("this.pos") == 4)
    try: E.apply(GAME, [{"file": "src/player.js", "search": "this.speed = 5;", "replace": "x"}]); t("a missing anchor is refused", False)
    except E.EditError as e: t("a missing anchor is refused, showing the closest text", "closest text is at line 6" in str(e), e)
    try: E.apply(GAME, [{"file": "src/player.js", "search": "this.speed = 3;", "replace": "x", "version": "000000000000"}]); t("a stale version is refused", False)
    except E.EditError as e: t("a stale file version is refused", "changed since it was read" in str(e))
    new, _ = E.apply(GAME, [{"file": "src/player.js", "symbol": "Player.move", "replace": "  move(dx) {\n    return this.x += dx;\n  }", "version": E.version(GAME["src/player.js"])}])
    t("symbol edit replaces a whole method", "return this.x += dx;" in new["src/player.js"] and "clamp(this.x" not in new["src/player.js"], new["src/player.js"])
    try: E.apply(GAME, [{"file": "src/player.js", "search": "return this.x;", "replace": "return this.x;\n  }}"}]); t("a syntax-breaking edit is refused", False)
    except E.EditError as e: t("an edit that breaks the syntax is refused", "no longer parses" in str(e), e)
    try: E.apply(GAME, [{"file": "src/player.js", "search": "this.speed = 3;", "replace": "this.speed = 4;"}, {"file": "nope.js", "search": "a", "replace": "b"}]); t("all-or-nothing", False)
    except E.EditError as e: t("all edits or none: the second edit's problem is reported", "edit 2 (nope.js)" in str(e))
    if not (tools.node() and tools.node_modules()):
        print("SKIP: debugger and atomic changes (checkers not installed)"); return finish()
    # ---- debugger
    bad = dict(GAME); bad["src/player.js"] = GAME["src/player.js"].replace("dx * this.speed", "dx * this.sped")
    r = debugger.run(bad)
    rep = debugger.report(r)
    t("debugger: failing test, expected vs actual, variable values", r["failed"] == 2 and "expected: 6" in rep and "actual:   NaN" in rep and "values:" in rep, rep)
    t("debugger: the project's source lines in the stack", "player.test.mjs:2" in rep, rep)
    r = debugger.run({"a.js": "function f(){ while(true){} }\n", "a.test.js": "test('loops', () => f());\n"}, timeout=5)
    t("debugger: an endless loop times out with a clear message", r.get("timed_out") and "endless loop" in debugger.report(r), r)
    r = debugger.run({"a.js": "console.error('texture missing', 3);\nfunction g(){ throw new Error('boom') }\n", "a.test.js": "test('g', () => g());\n"})
    t("debugger: console errors are structured", r["console"] and r["console"][0]["file"] == "a.js" and r["console"][0]["line"] == 1, r.get("console"))
    # ---- atomic changes
    d = tempfile.mkdtemp(); write(d, GAME)
    before = {p: open(os.path.join(d, *p.split("/")), encoding="utf-8").read() for p in GAME}
    res = atomic.change(d, [{"file": "src/player.js", "search": "dx * this.speed", "replace": "dx * this.sped"}])
    now = {p: open(os.path.join(d, *p.split("/")), encoding="utf-8").read() for p in GAME}
    t("a change that breaks a test is not applied, project untouched", not res["applied"] and now == before and "test now fails" in res["report"], res.get("report"))
    t("...and its report has the debugger's details", "expected: 6" in res["report"], res.get("report"))
    res = atomic.change(d, [{"file": "src/util.js", "search": "export function clamp", "replace": "export function clampValue"}])
    t("renaming a function its importer uses fails the checks", not res["applied"] and "new error" in res["report"], res.get("report"))
    edits = [{"file": "src/player.js", "search": "this.speed = 3;", "replace": "this.speed = 3;\n    this.jumps = 2;"},
             {"file": "src/player.test.mjs", "insert_after": "expect(p.move(5)).toBe(100); });\n", "text": "test('has jumps', () => { expect(new Player().jumps).toBe(2); });\n"}]
    res = atomic.change(d, edits)
    t("a good two-file change is applied", res["applied"] and "jumps = 2" in open(os.path.join(d, "src", "player.js")).read(), res.get("report"))
    t("history keeps it", len(atomic.history(d)) == 1)
    man = atomic.rollback(d)
    after = {p: open(os.path.join(d, *p.split("/")), encoding="utf-8").read() for p in GAME}
    t("rollback restores both files", man and after == before)
    res = atomic.change(d, [{"file": "src/extra.js", "create": "export const LEVELS = 3;\n"}])
    t("creating a file works", res["applied"] and os.path.exists(os.path.join(d, "src", "extra.js")), res.get("report"))
    atomic.rollback(d)
    t("rollback removes a created file", not os.path.exists(os.path.join(d, "src", "extra.js")))
    res = atomic.change(d, [{"file": "index.html", "search": "src/player.js", "replace": "src/plyer.js"}])
    t("a broken page reference fails the build check", not res["applied"] and "doesn't exist" in res["report"], res.get("report"))
    # an old error in the project doesn't block an unrelated fix
    write(d, {"src/old.js": "export const a = undefinedThing;\n"})
    res = atomic.change(d, [{"file": "src/player.js", "search": "this.speed = 3;", "replace": "this.speed = 3; // walking speed"}])
    t("problems the project already had don't block a change", res["applied"], res.get("report"))
    # the PC stopping mid-change is undone on the next run
    hist = atomic.history(d)[-1]["time"]
    open(os.path.join(d, "src", "player.js"), "w").write("half written")
    json.dump({"history": hist}, open(os.path.join(d, ".gameai", "journal.json"), "w"))
    note = atomic.recover(d)
    t("an interrupted change is undone", note and "this.speed = 3;\n" in open(os.path.join(d, "src", "player.js")).read())
    shutil.rmtree(d)
    finish()

def finish():
    print(f"{sum(R)}/{len(R)} passed")
    if not all(R): sys.exit(1)

main()
