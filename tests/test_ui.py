"""Tests for the headless renderer and layout checker (Phase 5 session 1). Needs Edge or Chrome; skipped without."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gameai.ui import render as R

T = []
def t(name, ok, info=None):
    ok = bool(ok); T.append(ok); print(("PASS " if ok else "FAIL ") + name)
    if not ok and info is not None: print("   ", str(info)[:1200])

BAD = """<!doctype html><html><body style="margin:0;font-family:sans-serif;background:#222">
<div style="width:1500px;height:20px;background:#333"></div>
<h1 style="color:#444">Low contrast title</h1>
<div style="position:relative"><button style="position:absolute;left:10px;top:0;width:120px;height:40px">Play</button>
<button style="position:absolute;left:60px;top:10px;width:120px;height:40px">Settings</button></div>
<div style="margin-top:80px;color:#fff"><p style="margin:0 0 4px 0">Item one</p><p style="margin:0 0 30px 2px">Item two</p><p style="margin:0">Item three</p></div>
<a href="#" style="display:inline-block;width:10px;height:10px;background:#fff"></a><p style="color:#ddd;font-size:9px">tiny text</p>
<div style="width:80px;overflow:hidden;white-space:nowrap;color:#fff">this text is far too long for its box</div>
</body></html>"""
GOOD = """<!doctype html><html><body style="margin:24px;font-family:sans-serif;background:#111;color:#eee">
<h1>Game Over</h1><p>Your score: <b>120</b></p><div style="display:flex;gap:12px"><button style="padding:10px 18px">Play again</button>
<button style="padding:10px 18px">Menu</button></div></body></html>"""

def main():
    if not R.find_browser():
        print("SKIP: no Edge or Chrome"); return finish()
    r = R.check(BAD)
    kinds = set(r["sizes"]["desktop"]["report"]["counts"])
    want = {"overlap", "contrast", "small text", "off screen", "alignment", "spacing", "small target", "cut off"}
    t("every planted problem is found", want <= kinds, (kinds, R.report(r)))
    t("both screen sizes are checked, with screenshots", set(r["sizes"]) == {"desktop", "phone"} and all(s["screenshot"] and os.path.getsize(s["screenshot"]) > 1000 for s in r["sizes"].values()))
    t("a bad page scores low", r["score"] < 0.5, r["score"])
    g = R.check(GOOD, screenshots=False)
    t("a clean page has no problems", g["score"] == 1.0 and not any(s["report"]["problems"] for s in g["sizes"].values()), R.report(g))
    rep = R.report(r)
    t("the text report names the element and the problem", 'button "Play": overlaps button "Settings"' in rep and "contrast 1.63:1" in rep, rep)
    from gameai.export_web import build_page
    a = R.check(build_page(), screenshots=False)
    t("the app's own page passes its layout check", a["score"] >= 0.95, R.report(a))
    ui_skill()
    finish()


def ui_skill():
    """Phase 5 session 2: the GameUI design system, the screen generator, the checker and practice"""
    import random, tempfile
    from gameai.ui import showcase, tasks, verify as V, practice as P
    for th in showcase.THEMES:
        r = R.check(showcase.page(showcase.SHOWCASE, th), screenshots=False)
        t(f"every GameUI part passes the layout check in the {th} style", r["score"] >= 0.95, R.report(r))
    vs = tasks.val_set()
    t("validation covers every kind of screen in every style", {(x["kind"], x["theme"]) for x in vs} == {(k, th) for k in tasks.KINDS for th in tasks.THEME_WORDS})
    bad = [(x["id"], V.verify(x, x["reference"])["reasons"]) for x in vs]
    bad = [b for b in bad if b[1]]
    t("every validation screen's reference answer passes the checker", not bad, bad[:3])
    train = P.practice_tasks(200, seed=1)
    t("practice and validation requests never share a prompt", not ({x["prompt"] for x in train} & {x["prompt"] for x in vs}))
    x = vs[0]; ref = x["reference"]
    cases = {
        "a script": ref.replace("</div>", "<script>alert(1)</script></div>", 1),
        "an outside image": ref.replace("</div>", '<img src="https://example.com/a.png"></div>', 1),
        "the wrong style": ref.replace(f'data-theme="{x["theme"]}"', 'data-theme="' + next(th for th in showcase.THEMES if th != x["theme"]) + '"'),
        "no style wrapper": ref[ref.index(">") + 1:-len("</div>")],
        "a missing text": ref.replace(x["texts"][0], "Something else"),
        "a hidden required text": ref.replace("</div>", f'<span style="display:none">{x["texts"][0]}</span></div>', 1).replace(">" + x["texts"][0] + "<", ">Something else<", 1),
        "no GameUI parts": "<div class=\"gui\" data-theme=\"%s\"><div>%s</div></div>" % (x["theme"], " ".join(x["texts"])),
    }
    for name, html in cases.items():
        r = V.verify(x, html)
        t(f"a screen with {name} is rejected", not r["accepted"], r)
    over = ref.replace('class="gui-screen"', 'class="gui-screen" style="position:relative"', 1)
    over = over.replace("</div>", '<button class="gui-btn" style="position:absolute;left:0;top:0;width:3000px">Wide</button></div>', 1)
    r = V.verify(x, over)
    t("a screen whose layout breaks is rejected, with the problem named", not r["accepted"] and ("off screen" in r["report"] or "cut off" in r["report"] or "layout" in r["report"]), r["report"])
    t("an answer in a ```html block is read", V.verify(x, "Here you go:\n```html\n" + ref + "\n```")["accepted"])
    # practice: a fake model that is right half the time -> only checked screens are kept, repairs get the report
    old = os.getcwd(); d = tempfile.mkdtemp(); os.chdir(d)
    try:
        P._g = type("G", (), {"after_step": lambda self: None})()
        calls = []
        task = train[0]
        def chat(msgs, **k):
            calls.append(msgs[-1]["content"])
            if "rejected" in msgs[-1]["content"]: return "```html\n" + task["reference"] + "\n```"
            return "```html\n" + (task["reference"] if len(calls) % 2 else task["reference"].replace(task["texts"][0], "oops")) + "\n```"
        s = P.teach_task(chat, task, n=4)
        rows = P.rows("first"); reps = P.rows("repair")
        t("practice keeps only screens that pass, at most once per identical screen", s["passed"] == 2 and len(rows) == 1 and all(V.verify(task, r["answer"])["accepted"] for r in rows), (s, len(rows)))
        t("a rejected screen gets a repair try with the checker's report, and a passing fix is kept", s["repairs_ok"] >= 1 and reps and "missing from the screen" in reps[0]["prompt"], (s, reps[:1]))
        t("the rejected screen is only in the request, never in the answer", all("oops" not in r["answer"] for r in rows + reps))
        pr = P.pass_rate(lambda m: "```html\n" + next(v["reference"] for v in vs if v["prompt"] == m[1]["content"]) + "\n```", vs[:4])
        t("the UI validation score counts accepted screens", pr["pass_rate"] == 1.0 and pr["tasks"] == 4, pr)
    finally:
        os.chdir(old)
    from gameai import lora_train
    t("the trainer knows the ui skill", "ui" in lora_train.SKILLS)
    from gameai import autopilot
    ids = [s[0] for s in autopilot.STEPS]
    t("autopilot runs UI practice, then UI training", ids.index("ui_practice") < ids.index("ui_train"))
    # games get GameUI screens, in a style matched to their world
    from gameai.headless import random_recipe, Engine
    from gameai.app import game_html
    for th, style in (("space", "neon"), ("forest", "cozy"), ("dungeon", "pixel")):
        r = random_recipe(random.Random(5), theme=th)
        html = game_html(r).replace("</body>", '<script>dispatchEvent(new KeyboardEvent("keydown",{code:"KeyP"}))</script></body>')
        res = R.check(html, screenshots=False)
        dom = res["sizes"]["desktop"]["dom"]
        t(f"a {th} game's pause screen is a GameUI screen in the {style} style, with clean layout", ">Paused<" in dom and f'data-theme="{style}"' in dom and ">Resume<" in dom and res["score"] >= 0.95, R.report(res))
    e = Engine()
    t("the playtesters still run the engine without a page", "proven" in e.judge(random_recipe(random.Random(3))))
    e.close()

def finish():
    print(f"{sum(T)}/{len(T)} passed")
    if not all(T): sys.exit(1)

main()
