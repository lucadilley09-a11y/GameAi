"""Full-stack memory and speed: the whole app's parts running at once, at several concurrency levels.

Measured for each level (1, 2, 4, ... playtesters at once):
  playtests per second, code-sandbox checks per second, peak RAM, peak VRAM, pagefile growth,
  and how quickly the AI still answers while all of that runs (median and slowest 5% of answers).
Plus headless browser rendering with the GPU vs software rendering (if Edge or Chrome is installed).
"""
import glob, json, os, random, statistics, subprocess, tempfile, time
from concurrent.futures import ProcessPoolExecutor

from .common import Sampler, median_spread, progress, save, say, stage

SECONDS = 20  # per concurrency level


def _playtest_job(seed):
    from ..headless import _worker_judge, random_recipe
    return _worker_judge(random_recipe(random.Random(seed)))["proven"]


def _sandbox_job(seed):
    from ..code import sandbox, specs
    rng = random.Random(seed)
    s = specs.random_spec(rng)
    return sandbox.verify(specs.spec_text(s), specs.code(s, rng), seed=seed)[0]


def _warm(_):
    from ..headless import _worker_judge, random_recipe
    _worker_judge(random_recipe(random.Random(0), enemy="none"))
    return True


def levels():
    n = os.cpu_count() or 2
    out = [1, 2, 4, 8, 12, 16]
    return [k for k in out if k <= n] + ([n] if n not in out else [])


def ai_answer_times(stop_at):
    """The game AI answering one request at a time while everything else runs; -> list of ms per answer"""
    try:
        import torch
        from ..generate import load, decode
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok, _ = load("models/current.pt", dev, upgrade=True)
    except Exception as e:
        return None, f"no game AI to test ({e})"
    prompts = ["a snowy level with dragons", "ninja in space collecting gems", "hard dungeon with ghosts and a boss"]
    decode(model, tok, prompts, dev)  # warm-up
    times = []
    while time.time() < stop_at:
        t = time.perf_counter(); decode(model, tok, [random.choice(prompts)], dev); times.append((time.perf_counter() - t) * 1000)
    return times, None


def pct(xs, q):
    if not xs: return None
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def concurrency_test():
    rows = []
    idle, why = ai_answer_times(time.time() + 5)
    base = {"answer_ms_median": pct(idle, .5), "answer_ms_p95": pct(idle, .95)} if idle else {"note": why}
    say("AI answering with nothing else running:", base)
    lv = levels()
    for i, n in enumerate(lv):
        progress(i, len(lv))
        say(f"{n} playtester(s) and {n} sandbox(es) at once, for {SECONDS} s...")
        with ProcessPoolExecutor(n) as pt, ProcessPoolExecutor(n) as sb:
            list(pt.map(_warm, range(n)))  # every worker loads its engine first
            with Sampler() as smp:
                t0, end = time.time(), time.time() + SECONDS
                futs_p = [pt.submit(_playtest_job, 1000 * n + k) for k in range(n * 2)]
                futs_s = [sb.submit(_sandbox_job, 1000 * n + k) for k in range(n * 2)]
                import threading
                box = {}
                th = threading.Thread(target=lambda: box.update(a=ai_answer_times(end)[0]), daemon=True); th.start()
                done_p = done_s = 0; k = n * 2
                while time.time() < end:
                    for f in [f for f in futs_p if f.done()]: futs_p.remove(f); done_p += 1; futs_p.append(pt.submit(_playtest_job, 1000 * n + k)); k += 1
                    for f in [f for f in futs_s if f.done()]: futs_s.remove(f); done_s += 1; futs_s.append(sb.submit(_sandbox_job, 1000 * n + k)); k += 1
                    time.sleep(0.02)
                secs = time.time() - t0
                th.join(30); answers = box.get("a")
                for f in futs_p + futs_s: f.cancel()
            row = {"at_once": n, "playtests_per_s": round(done_p / secs, 2), "sandbox_checks_per_s": round(done_s / secs, 2),
                   "answer_ms_median": pct(answers, .5), "answer_ms_p95": pct(answers, .95), **smp.result()}
        rows.append(row); say(" ", row)
    progress(len(lv), len(lv))
    return {"idle": base, "levels": rows}


TEST_PAGE = """<!doctype html><body><canvas id=c width=960 height=640></canvas><script>
const c=document.getElementById('c').getContext('2d'),t0=performance.now();let f=0;
for(;f<300;f++){c.fillStyle='#123';c.fillRect(0,0,960,640);
 for(let i=0;i<400;i++){c.fillStyle='hsl('+(i*7+f)%360+',70%,50%)';c.fillRect((i*37+f*3)%960,(i*53+f*2)%640,24,24)}
 for(let i=0;i<60;i++){c.beginPath();c.arc((i*91+f*5)%960,(i*61+f*4)%640,10,0,6.28);c.fill()}
 c.getImageData(0,0,1,1)}
document.body.setAttribute('data-ms',((performance.now()-t0)/f).toFixed(3));
</script></body>"""


def find_browser():
    cands = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    cands += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"))
    for name in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        from shutil import which
        if which(name): cands.append(which(name))
    return next((c for c in cands if c and os.path.exists(c)), None)


def render_test():
    """The same drawing work with the GPU and with software rendering; ms per frame (median of 3)."""
    b = find_browser()
    if not b: return {"note": "no Edge or Chrome found, so rendering wasn't measured"}
    d = tempfile.mkdtemp(); page = os.path.join(d, "render.html"); open(page, "w").write(TEST_PAGE)
    res = {"browser": os.path.basename(b)}
    for mode, flags in (("gpu", ["--enable-gpu", "--use-angle=default"]), ("software", ["--disable-gpu"])):
        ms, mem = [], 0
        for _ in range(3):
            with Sampler(0.2) as smp:
                out = subprocess.run([b, "--headless=new", "--no-first-run", f"--user-data-dir={d}\\{mode}" if os.name == "nt" else f"--user-data-dir={d}/{mode}",
                                      *flags, "--dump-dom", "file:///" + page.replace("\\", "/")], capture_output=True, text=True, timeout=120).stdout
            import re
            m = re.search(r'data-ms="([\d.]+)"', out)
            if m: ms.append(float(m.group(1)))
            mem = max(mem, smp.result()["procs_gb"])
        res[mode] = {"ms_per_frame": median_spread(ms), "peak_memory_gb": round(mem, 2)}
    return res


def run():
    stage("Full-stack memory and concurrency")
    res = {"concurrency": concurrency_test()}
    say("Rendering with the GPU vs software...")
    try: res["rendering"] = render_test()
    except Exception as e: res["rendering"] = {"error": str(e)}
    say(" ", res["rendering"])
    path = save("fullstack", res); say(f"Saved: {path}")
    return res
