"""Checking one answer to a UI task (plan: Phase 5). An answer is the screen's HTML built from GameUI parts.

Accepted only if:
  1. it's self-contained and offline: no scripts, no outside files or web addresses (GameUI's stylesheet is added);
  2. it's in the style asked for (data-theme) and built from the design system (GameUI parts, not ad-hoc styling);
  3. everything the request asks for is really on the screen (each required text, and enough buttons, inputs...);
  4. the layout checker finds no overlap, unreadable text, cut-off or off-screen parts at desktop and phone size, and
     scores it at least MIN_SCORE.
    r = verify(task, answer)  ->  {"accepted", "reasons", "report", "score", "html"}
"""
import re
from html.parser import HTMLParser

from . import render, showcase

MIN_SCORE = 0.95
SERIOUS = {"overlap", "contrast", "off screen", "cut off"}
FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)


class _Scan(HTMLParser):
    """what's on the page: visible text, classes, tags, attributes, themes"""
    def __init__(self):
        super().__init__(); self.text, self.classes, self.tags, self.themes, self.attrs, self.skip, self.bad = [], set(), [], [], [], 0, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append(tag); self.attrs.append((tag, a))
        self.classes |= set((a.get("class") or "").split())
        if a.get("data-theme"): self.themes.append(a["data-theme"])
        style = (a.get("style") or "").replace(" ", "").lower()
        if tag in ("script", "iframe", "object", "embed"): self.bad.append(f"<{tag}> isn't allowed (screens are plain HTML with GameUI)")
        if tag == "link": self.bad.append("outside stylesheets aren't allowed (GameUI is added for you)")
        for k, v in a.items():
            if v and re.search(r"(https?:)?//|url\(\s*['\"]?(https?:|//)", v): self.bad.append(f"{k}=\"{v[:40]}\" points outside (it must work offline)")
            if k.startswith("on"): self.bad.append(f"{k}=... scripts aren't allowed")
        if tag in ("style", "script") or "display:none" in style or "visibility:hidden" in style: self.skip += 1; self._skip_tag = tag
        else: self._skip_tag = None
        self._stack = getattr(self, "_stack", []) + [(tag, self._skip_tag is not None)]

    def handle_endtag(self, tag):
        st = getattr(self, "_stack", [])
        while st:
            t, s = st.pop()
            if s: self.skip -= 1
            if t == tag: break

    def handle_data(self, data):
        if not self.skip and data.strip(): self.text.append(data)


def extract(answer):
    blocks = FENCE.findall(answer or "")
    return (blocks[-1] if blocks else answer or "").strip()


def build(task, html):
    """the full page to render: the answer's body inside GameUI, in the answer's theme (or the task's)"""
    body = re.sub(r"(?is)<!doctype.*?>|</?(html|head|body)[^>]*>|<meta[^>]*>|<title>.*?</title>", "", html)
    m = re.search(r'<div[^>]*class="[^"]*\bgui\b[^"]*"[^>]*data-theme="([a-z]+)"[^>]*>', body) or re.search(r'<div[^>]*data-theme="([a-z]+)"[^>]*class="[^"]*\bgui\b', body)
    if m:  # the answer brought its own GameUI wrapper: keep it, just add the stylesheet
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f"<style>html,body{{margin:0}}{showcase.css()}</style></head><body>{body}</body></html>")
    return showcase.page(body, task["theme"])


def _count(scan, selector):
    """how many elements match a simple selector list: tag, .class or [attr=value], comma-separated"""
    n = 0
    for tag, a in scan.attrs:
        for sel in selector.split(","):
            sel = sel.strip()
            if sel.startswith("."): ok = sel[1:] in (a.get("class") or "").split()
            elif sel.startswith("["): k, _, v = sel[1:-1].partition("="); ok = a.get(k) == v.strip("'\"")
            else: ok = tag == sel
            if ok: n += 1; break
    return n


def verify(task, answer, sizes=None):
    html = extract(answer)
    out = {"accepted": False, "reasons": [], "report": "", "score": 0.0, "html": html}
    scan = _Scan()
    try: scan.feed(html)
    except Exception as e: out["reasons"].append(f"the HTML couldn't be read ({e})")
    reasons = list(dict.fromkeys(scan.bad))
    theme = scan.themes[0] if scan.themes else None
    if not theme: reasons.append('wrap the screen in <div class="gui" data-theme="..."> with the style asked for')
    elif theme != task["theme"]: reasons.append(f'the style is "{theme}", but "{task["theme"]}" was asked for')
    parts = {c for c in scan.classes if c.startswith("gui-")}
    if len(parts) < 2: reasons.append("build it from GameUI parts (gui-screen, gui-menu, gui-btn, gui-panel, ...), not your own styling")
    text = re.sub(r"\s+", " ", " ".join(scan.text)).lower()
    missing = [t for t in task["texts"] if t.lower() not in text]
    if missing: reasons.append("missing from the screen: " + ", ".join(f'"{m}"' for m in missing))
    for sel, n in task["need"].items():
        have = _count(scan, sel)
        if have < n: reasons.append(f"needs at least {n} × {sel.split(',')[0]}, has {have}")
    if not reasons:  # only render what could pass anyway: rendering takes a few seconds
        r = render.check(build(task, html), sizes=sizes, screenshots=False)
        out["score"] = r["score"]
        serious = [(name, p) for name, s in r["sizes"].items() for p in s["report"].get("problems", []) if p["kind"] in SERIOUS]
        if any(s["report"].get("error") for s in r["sizes"].values()): reasons.append("the page didn't render")
        if serious: reasons += [f"{name}: [{p['kind']}] {p['element']}: {p['message']}" for name, p in serious[:8]]
        elif r["score"] < MIN_SCORE: reasons.append(f"layout score {r['score']:.2f} (needs {MIN_SCORE}):\n" + render.report(r, max_items=8))
    out["reasons"] = reasons
    out["accepted"] = not reasons
    out["report"] = "\n".join(reasons)
    return out
