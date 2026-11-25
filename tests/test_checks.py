"""Tests for the code checkers (Phase 4 session 2). Needs  python -m gameai.checks install  first (skips if not)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gameai import checks
from gameai.checks import suppressions, tools

R = []
def t(name, ok): ok = bool(ok); R.append(ok); print(("PASS " if ok else "FAIL ") + name)

def codes(r): return {d["code"] for d in r["errors"]}

def main():
    if not (tools.node() and tools.node_modules()):
        print("SKIP: checkers not installed"); return
    kit = os.path.join(os.path.dirname(checks.KIT_DTS), "examples")
    ex = {f: open(os.path.join(kit, f), encoding="utf-8").read() for f in os.listdir(kit) if f.endswith(".js")}
    r = checks.validate(ex)
    t("the GameKit examples pass", r["passed"] or print(checks.report(r)))
    bad = "function build(canvas){\n  const g = GameKit.game({canvas});\n  g.levle('rooms');\n  g.goal('colect','coin');\n  return g;\n}\n"
    r = checks.validate({"game.js": bad})
    t("wrong GameKit names are caught (levle, colect)", not r["passed"] and {"TS2551", "TS2345"} <= codes(r) or print(checks.report(r)))
    t("formatting is shown as a diff, not silent", "game.js" in r["diffs"] and "+" in r["diffs"]["game.js"])
    r = checks.validate({"a.js": "let x = 1\nx = undefinedThing + 1\n"})
    t("unknown names are errors", "TS2304" in codes(r))
    r = checks.validate({"a.js": "try { go() } catch (e) {}\nfunction go(){}\n"})
    t("an empty catch (hiding errors) fails", "no-empty" in codes(r))
    r = checks.validate({"a.ts": "const n: number = 'five';\nexport {};\n"})
    t("TypeScript type errors are caught", "TS2322" in codes(r))
    r = checks.validate({"a.js": "function f( {\n"})
    t("syntax errors fail", not r["passed"])
    # suppressions
    s = suppressions.suppressions({"a.ts": "// @ts-ignore\nlet a: any = 1;\n// eslint-disable-next-line no-console -- the debug overlay prints on purpose\nconsole.log(a);\nconst s = 'type : any here';\n"})
    kinds = [(x["kind"], x["justified"]) for x in s]
    t("unjustified @ts-ignore and any are found, a justified disable passes, strings ignored",
      kinds == [("ts-ignore", False), ("any", False), ("eslint-disable", True)] or print(kinds))
    s = suppressions.suppressions({"a.ts": "// justification: the save file is checked right below\n// @ts-expect-error\nlet b = 1;\n"})
    t("a reason on the line above counts", s and s[0]["justified"])
    s = suppressions.suppressions({"a.py": "import os  # noqa\nx: Any = 1  # because the plugin API has no types\n", "a.lua": "---@diagnostic disable-next-line: undefined-global\n"})
    t("Python and Lua suppressions are found", sorted((x["file"], x["justified"]) for x in s) == [("a.lua", False), ("a.py", False), ("a.py", True)] or print(s))
    r = checks.validate({"a.ts": "// @ts-ignore\nconst n: number = 'x';\nexport {};\n"})
    t("an unjustified suppression fails validation", not r["passed"] and r["unjustified"])
    r = checks.validate({"a.js": "// eslint-disable-next-line no-console -- nothing to hide on this line\nconst x = 1;\nexport { x };\n"})
    t("a disable comment that hides nothing is an error", "unused-directive" in str(r["errors"]) or "eslint" in {d["source"] for d in r["errors"]} or print(checks.report(r)))
    # python
    if tools.ruff():
        r = checks.validate({"a.py": "import os\ntry:\n  x=1\nexcept:\n  pass\nprint(undefined_name)\n"})
        t("Python: formatted, bare except and unknown names are errors", "a.py" in r["diffs"] and {"E722", "F821"} <= codes(r) or print(checks.report(r)))
    # lua formatting
    r = checks.validate({"a.lua": "local   x=1\nif x==1 then print( 'hi' ) end\n"})
    t("Lua is formatted", "a.lua" in r["diffs"] and r["passed"])
    r = checks.validate({"a.lua": "local x = \nif then end\n"})
    t("Lua syntax errors are caught", not r["passed"] and any(d["source"] in ("lua", "formatter") for d in r["errors"]) or print(checks.report(r)))
    # language server
    f = {"game.js": "function build(canvas) {\n  const g = GameKit.game({ canvas });\n  g.spawn('coin', 3);\n  return g;\n}\n"}
    h = checks.hover(f, "game.js", 3, 6)
    t("hover shows GameKit's types", h["text"] and "spawn" in h["text"] or print(h))
    d = checks.definition(f, "game.js", 3, 6)
    t("go to definition finds gamekit.d.ts", d and d[0]["file"] == "gamekit.d.ts" or print(d))
    c = checks.completions({"game.js": "const g = GameKit.game({});\ng.\n"}, "game.js", 2, 3)
    t("completions list GameKit methods", {"spawn", "goal", "level"} <= set(c) or print(c[:20]))
    refs = checks.references(f, "game.js", 2, 9)
    t("references find every use", len(refs) >= 3 or print(refs))
    import time; t0 = time.time(); checks.validate(ex); dt = time.time() - t0
    t(f"a repeat check is fast ({dt:.1f}s)", dt < 5)
    print(f"{sum(R)}/{len(R)} passed")
    if not all(R): sys.exit(1)

main()
