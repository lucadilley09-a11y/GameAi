"""The headless renderer (plan: Phase 5, session 1): opens a page in your own Edge or Chrome without a window, at one or
more screen sizes, runs the layout checker inside it and takes a screenshot. Works offline (the page can't reach the
network), uses no extra downloads, and nothing appears on your screen.

    from gameai.ui import render
    r = render.check(html)                              # or a path to an .html file
    r["sizes"]["desktop"]["report"]["problems"]        # overlaps, contrast, cut off, alignment, spacing, ...
    print(render.report(r))                             # as text, for a repair prompt and for you

Screen sizes: desktop 1280x800 and phone 390x844 (a page has to work on both); any other with sizes={...}.
"""
import json, os, re, shutil, subprocess, tempfile

from ..phase0.fullstack import find_browser

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "layout_check.js")
SIZES = {"desktop": (1280, 800), "phone": (390, 844)}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _flags(w, h, profile):
    f = ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio", f"--window-size={w},{h}", f"--user-data-dir={profile}",
         "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost", "--no-first-run", "--no-default-browser-check",
         "--virtual-time-budget=4000", "--run-all-compositor-stages-before-draw"]
    if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0: f.append("--no-sandbox")  # only needed when running as root (tests)
    return f


def _page(html):
    """the page with the checker added, run once everything has loaded"""
    checker = open(CHECKER, encoding="utf-8").read()
    script = "<script>window.addEventListener('load',()=>setTimeout(()=>{try{" + checker + "}catch(e){const p=document.createElement('pre');p.id='__layout_report';p.textContent=JSON.stringify({error:String(e)});document.body.appendChild(p)}},300))</script>"
    return html.replace("</body>", script + "</body>") if "</body>" in html else html + script


def check(page, sizes=None, screenshots=True, out_dir=None, timeout=60):
    """page: HTML text or a path to an .html file -> {"sizes": {name: {"report", "screenshot"}}, "score"}"""
    browser = find_browser()
    if not browser: raise RuntimeError("the renderer needs Microsoft Edge or Google Chrome (Edge comes with Windows)")
    html = open(page, encoding="utf-8").read() if os.path.exists(str(page)) and str(page).endswith((".html", ".htm")) else page
    tmp = tempfile.mkdtemp(prefix="gameai-render-")
    out_dir = out_dir or tmp
    os.makedirs(out_dir, exist_ok=True)
    try:
        f = os.path.join(tmp, "page.html")
        open(f, "w", encoding="utf-8").write(_page(html))
        url = "file:///" + f.replace("\\", "/").lstrip("/")
        res = {"sizes": {}}
        for name, (w, h) in (sizes or SIZES).items():
            prof = os.path.join(tmp, "profile-" + name)
            p = subprocess.run([browser, *_flags(w, h, prof), "--dump-dom", url], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout, creationflags=NO_WINDOW)
            found = re.findall(r'<pre id="__layout_report"[^>]*>(.*?)</pre>', p.stdout, re.S)  # the last one: the checker's own comment mentions the tag
            import html as H
            try: rep = json.loads(H.unescape(found[-1]))
            except (IndexError, ValueError): rep = {"error": "the page didn't finish loading (or a script on it failed)", "problems": []}
            shot = None
            if screenshots:
                shot = os.path.join(out_dir, f"screenshot-{name}.png")
                subprocess.run([browser, *_flags(w, h, prof), f"--screenshot={shot}", url], capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
                if not os.path.exists(shot): shot = None
            res["sizes"][name] = {"size": [w, h], "report": rep, "screenshot": shot}
        scores = [s["report"].get("score") for s in res["sizes"].values() if s["report"].get("score") is not None]
        res["score"] = min(scores) if scores else 0.0
        return res
    finally:
        if out_dir != tmp: shutil.rmtree(tmp, ignore_errors=True)


def report(res, max_items=20):
    """the result as text: each size's score, then its problems"""
    out = [f"layout score {res['score']:.2f} (1 = no problems found)"]
    for name, s in res["sizes"].items():
        rep = s["report"]
        if rep.get("error"): out.append(f"{name}: {rep['error']}"); continue
        out.append(f"{name} {s['size'][0]}x{s['size'][1]}: " + (", ".join(f"{n} {k}" for k, n in rep["counts"].items()) or "no problems"))
        for p in rep["problems"][:max_items]:
            out.append(f"  [{p['kind']}] {p['element']}: {p['message']}")
    return "\n".join(out)
