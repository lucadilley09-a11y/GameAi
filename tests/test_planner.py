"""Tests for the project planner and the component library (Phase 4 session 6), with a scripted stand-in model.
Needs the checkers installed (python -m gameai.checks install); skipped without them."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gameai.checks import tools

R = []
def t(name, ok, info=None):
    ok = bool(ok); R.append(ok); print(("PASS " if ok else "FAIL ") + name)
    if not ok and info is not None: print("   ", str(info)[:1500])

HEALTH = "export function heal(hp, amount, maxHp) {\n  return Math.max(0, Math.min(maxHp, hp + amount));\n}\n"
HEALTH_TEST = "import { heal } from './health.js';\ntest('heals', () => expect(heal(5, 3, 10)).toBe(8));\ntest('caps', () => expect(heal(9, 5, 10)).toBe(10));\n"
DAMAGE = "import { heal } from './health.js';\nexport function hit(hp, dmg) {\n  return heal(hp, -dmg, 99);\n}\n"
DAMAGE_TEST = "import { hit } from './damage.js';\ntest('hit', () => expect(hit(5, 2)).toBe(3));\ntest('floor', () => expect(hit(1, 9)).toBe(0));\n"
SCORE = "export function addScore(s, n) {\n  return s + n * 10;\n}\n"
SCORE_TEST = "import { addScore } from './score.js';\ntest('score', () => expect(addScore(0, 2)).toBe(20));\n"
PLAN = {"name": "Tiny", "goal": "a tiny game", "design": {"rules": "heal, take damage, score"},
        "features": [{"id": "f1", "title": "Health", "description": "healing with a max", "acceptance": ["heal adds and caps"], "depends": []},
                     {"id": "f2", "title": "Damage", "description": "taking hits", "acceptance": ["hit lowers hp, not below 0"], "depends": ["f1"]},
                     {"id": "f3", "title": "Score", "description": "points", "acceptance": ["10 per point"], "depends": []}]}


def main():
    if not (tools.node() and tools.node_modules()):
        print("SKIP: checkers not installed"); return finish()
    here = os.getcwd(); work = tempfile.mkdtemp(); os.chdir(work)
    try:
        from gameai.planner import project as P, builder as B, components as C
        bad = json.loads(json.dumps(PLAN)); bad["features"][1]["depends"] = ["f9"]; bad["features"][0]["acceptance"] = []
        probs = P.check(bad)
        t("plan check: unknown dependency and missing acceptance are caught", any("unknown feature f9" in p for p in probs) and any("acceptance" in p for p in probs), probs)
        circ = json.loads(json.dumps(PLAN)); circ["features"][0]["depends"] = ["f2"]
        t("plan check: circular dependencies are caught", any("circle" in p for p in P.check(circ)), P.check(circ))
        # the scripted model
        script = {"plan": [json.dumps(bad), "```json\n" + json.dumps(PLAN) + "\n```"],
                  "Health": ["sure, here you go", [{"file": "health.js", "create": HEALTH}, {"file": "health.test.js", "create": HEALTH_TEST}]],
                  "Damage": [[{"file": "health.test.js", "search": "toBe(8)", "replace": "toBe(9)"}],
                             [{"file": "health.js", "search": "Math.min(maxHp, hp + amount)", "replace": "hp + amount"}, {"file": "damage.js", "create": DAMAGE}, {"file": "damage.test.js", "create": DAMAGE_TEST}],
                             [{"file": "damage.js", "create": DAMAGE}, {"file": "damage.test.js", "create": DAMAGE_TEST}]],
                  "Score": [[{"file": "score.js", "create": SCORE.replace("10", "100")}, {"file": "score.test.js", "create": SCORE_TEST}]] * 3 +
                           [[{"file": "score.js", "create": SCORE}]],
                  "Score-after-help": [[{"file": "score.js", "create": SCORE}, {"file": "score.test.js", "create": SCORE_TEST}]]}
        seen = []
        def model(msgs):
            user = msgs[-1]["content"]
            key = "plan" if "Plan this" in msgs[1]["content"] else next(k for k in ("Health", "Damage", "Score") if f"Build this feature now: {k}" in user)
            if key == "Score" and "user's guidance" in user: key = "Score-after-help"
            seen.append((key, user))
            out = script[key].pop(0) if len(script[key]) > 1 else script[key][0]
            return out if isinstance(out, str) else "```json\n" + json.dumps(out) + "\n```"
        logs = []
        plan = B.make_plan(model, "a tiny game", "game", lang="js")
        t("a plan with problems is sent back and corrected", plan["name"] == "Tiny" and len([s for s in seen if s[0] == "plan"]) == 2)
        t("DESIGN.md and FEATURES.md are written", os.path.exists("game/DESIGN.md") and "[ ] **Health**" in open("game/FEATURES.md").read())
        st, f, _ = B.build_next(model, "game", log=logs.append)
        t("feature 1: a non-edit reply is retried, then built with its tests", st == "done" and f["id"] == "f1" and f["tests"] == ["health.test.js"] and f["attempts"] == 2, (st, f, logs))
        st, f, _ = B.build_next(model, "game", log=logs.append)
        t("feature 2: changing a regression test is refused", any("changed regression tests" in l for l in logs), logs)
        t("feature 2: breaking feature 1 is refused (all-or-nothing)", any("not applied" in l for l in logs) and "Math.min(maxHp" in open("game/health.js").read(), logs)
        t("feature 2: then built on its dependency", st == "done" and f["id"] == "f2" and f["attempts"] == 3, (st, f))
        dmg_prompt = [u for k, u in seen if k == "Damage"][0]
        t("the prompt has the project summary, the map and the regression tests", "Done (1/3): Health" in dmg_prompt and "function heal(hp, amount, maxHp)" in dmg_prompt and "health.test.js" in dmg_prompt, dmg_prompt[:800])
        st, f, rep = B.build_next(model, "game", log=logs.append)
        t("feature 3: its own failing tests are undone, then it's blocked after 4 tries", st == "blocked" and f["id"] == "f3" and not os.path.exists("game/score.js"), (st, logs[-6:]))
        t("blocked features are reported with the reason", "[!] **Score**" in open("game/FEATURES.md").read() and "Blocked" in P.summary(P.load("game")))
        B.unblock("game", "f3", "points are worth 10 each, not 100")
        st, f, _ = B.build_next(model, "game", log=logs.append)
        t("after the user's guidance it's built", st == "done" and f["id"] == "f3", (st, logs[-3:]))
        t("the guidance reached the prompt", any("points are worth 10 each" in u for k, u in seen if k == "Score-after-help"))
        st, f, _ = B.build_next(model, "game", log=logs.append)
        t("then the project is finished", st == "finished" and "Done (3/3)" in P.summary(P.load("game")))
        # components
        names = [c["name"] for c in C.load()]
        t("finished features' files became checked components", "Tiny/health" in names and "Tiny/score" in names, names)
        t("search finds a component by meaning words", C.search("heal hp up to max")[0]["name"] == "Tiny/health", [c["name"] for c in C.search("heal hp up to max")])
        ok, why = C.add("dup", "js", "health.js", HEALTH.replace("hp", "life"), {"health.test.js": HEALTH_TEST.replace("hp", "life")}, "again")
        t("a near-copy isn't added twice", not ok and "already" in why, why)
        ok, why = C.add("wrong", "js", "w.js", "export const w = () => 1;\n", {"w.test.js": "import { w } from './w.js';\ntest('w', () => expect(w()).toBe(2));\n"}, "broken")
        t("a component whose tests fail isn't added", not ok and "don't pass" in why)
        P.decide("game", P.load("game"), "bats move in waves of 3")
        t("decisions are kept for later sessions", "bats move in waves of 3" in P.summary(P.load("game")) and "Decisions" in open("game/DESIGN.md").read())
    finally:
        os.chdir(here); shutil.rmtree(work, ignore_errors=True)
    finish()

def finish():
    print(f"{sum(R)}/{len(R)} passed")
    if not all(R): sys.exit(1)

main()
