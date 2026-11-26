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
    finish()

def finish():
    print(f"{sum(T)}/{len(T)} passed")
    if not all(T): sys.exit(1)

main()
