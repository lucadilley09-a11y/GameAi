"""The Game AI app: the terminal in its own window, with your AI running on the GPU.

    python -m gameai.app        (or double-click start.bat)

Runs a small local server (only reachable from this PC) and opens the terminal in an
app window (Microsoft Edge/Chrome app mode, or your browser if neither is found)."""
import json, re, os, shutil, subprocess, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

PORT = 8765
SHARE_PORT = 8766
STATE = {"busy": 0, "mtime": 0, "model": None, "tok": None, "device": "cpu", "lock": threading.Lock(),
         "job": {"running": False, "stage": "", "step": 0, "total": 0, "loss": None, "evals": [], "log": [], "ok": None, "started": 0}}


def load_model():
    if not os.path.exists("models/current.pt"):
        STATE["model"] = None; return
    import torch
    from .generate import load
    STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        STATE["model"], STATE["tok"], _ = load("models/current.pt", STATE["device"])
        STATE["note"] = None
    except ValueError as e:  # trained on an older version of the game language
        STATE["model"], STATE["note"] = None, str(e)
    STATE["mtime"] = os.path.getmtime("models/current.pt")


def hot_swap():
    """Switch to a newer model the moment always-on learning promotes one."""
    if os.path.exists("models/current.pt") and os.path.getmtime("models/current.pt") != STATE["mtime"]:
        time.sleep(0.5)  # let the copy finish
        load_model()


def status():
    m, s = STATE["model"], None
    if os.path.exists("models/current.json"):
        s = json.load(open("models/current.json"))["fresh"]
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    from .update import local_version
    v = STATE.setdefault("started_version", local_version())  # the version this running app is, not the one just downloaded
    from .update import mode as update_mode
    return {"updating": STATE.get("updating"), "update_waiting": bool(STATE.get("update_waiting")), "update_available": STATE.get("update_available"), "leftovers": STATE.get("leftovers", 0), "autopilot": autopilot_status(),
            "update_mode": update_mode(), "app": True, "version": (v.get("sha") or "dev")[:7], "update_msg": v.get("message"), "sharing": bool(STATE.get("sharing")), "note": STATE.get("note"), "loaded": m is not None, "params": m.num_params() if m else 0, "gpu": gpu, "fresh": s,
            "training": STATE["job"]["running"], "learning": STATE["job"].get("kind") == "learn" and STATE["job"]["running"]}


def decode(prompt):
    from .generate import decode as dec
    with STATE["lock"]:
        hot_swap()
        out = dec(STATE["model"], STATE["tok"], [prompt], STATE["device"], gate=0.0)[0]
    return {f: {"value": v, "conf": c} for f, (v, c) in out.items()}


def run_job(args, kind="job"):
    j = STATE["job"]
    j.update(stopped=False, kind=kind, running=True, stage="Starting", step=0, total=0, loss=None, evals=[], log=[], ok=None, started=time.time())
    env = dict(os.environ, GAMEAI_PROGRESS="1", PYTHONUNBUFFERED="1")
    if os.name != "nt": env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # GPU memory wastes less space when the teacher shares it
    p = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
    STATE["proc"] = p
    os.makedirs("logs", exist_ok=True)
    jl = open("logs/job.log", "a", encoding="utf-8")  # the whole output of every job, for finding problems later
    jl.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(args)} ---\n"); jl.flush()
    try:
        for line in p.stdout:
            line = line.rstrip()
            if not line.startswith("@@P") or line.startswith('@@P{"step": 1,') or re.match(r'@@P\{"step": \d*01,', line):
                jl.write(time.strftime("%H:%M:%S ") + line + "\n"); jl.flush()  # every line, but progress only every 100 steps
            try:
                if line.startswith("@@P"):
                    d = json.loads(line[3:]); j.update(step=d["step"], total=d["total"], loss=d["loss"])
                    continue
                if line.startswith("@@L"): continue  # round results: the page reads learn/history.jsonl
                if line.startswith("@@E"):
                    j["evals"] = (j["evals"] + [json.loads(line[3:])])[-200:]; continue
            except (ValueError, KeyError):
                pass  # a garbled progress line: show it as text instead of crashing
            if line.startswith("==="): j["stage"] = line.strip("= ")
            j["log"] = (j["log"] + [line])[-40:]
        j["ok"] = p.wait() == 0
        jl.write(f"--- finished: {'ok' if j['ok'] else 'failed'} ---\n")
    except Exception as e:  # whatever happens, the job must not stay "running" forever
        j["log"] = (j["log"] + [f"The app lost track of the job: {e}"])[-40:]; j["ok"] = False
        if p.poll() is None: p.kill()
    finally:
        jl.close()
        try:
            with STATE["lock"]:
                load_model()
        finally:
            j["running"] = False


def page():
    from .export_web import build_page
    return build_page().encode()


def game_html(recipe):
    tpl = open(os.path.join(os.path.dirname(__file__), "play_template.html"), encoding="utf-8").read()
    return tpl.replace("/*GAMEAI_RECIPE*/", "window.GAMEAI_RECIPE=" + json.dumps(recipe) + ";")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        if isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path == "/": return self.send(200, page(), "text/html; charset=utf-8")
        if path.startswith("/kit/") and re.fullmatch(r"/kit/[a-z0-9_]+", path):  # play a GameKit example game
            ex = os.path.join(os.path.dirname(__file__), "kit", "examples", path[5:] + ".js")
            if not os.path.exists(ex): return self.send(404, "no such example")
            kit = open(os.path.join(os.path.dirname(__file__), "kit", "gamekit.js"), encoding="utf-8").read()
            html = ("<!doctype html><meta charset=utf-8><title>" + path[5:] + "</title><style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}"
                    "canvas{image-rendering:pixelated;max-width:100vw;max-height:100vh}</style><canvas id=c></canvas><script>" + kit + "\n"
                    + open(ex, encoding="utf-8").read() + "\nconst cv=document.getElementById('c'),g=build(cv,(Math.random()*1e9)|0);cv.width=g.o.width;cv.height=g.o.height;"
                    "if(location.search.includes('play'))g.start();else(function loop(){g.draw();if(g.state==='title')requestAnimationFrame(loop)})();</script>")
            return self.send(200, html, "text/html; charset=utf-8")
        if path == "/api/storage":
            from . import storage
            return self.send(200, {"text": storage.text_report()})
        if path == "/api/kit":
            d = os.path.join(os.path.dirname(__file__), "kit", "examples")
            return self.send(200, {"examples": sorted(f[:-3] for f in os.listdir(d) if f.endswith(".js"))})
        if path == "/api/status":
            STATE["seen"] = time.time()  # the PC window or your Chromebook is using the app
            return self.send(200, status())
        if path == "/api/model" and "meta=1" in self.path:  # just the list of settings (a few KB): decoding runs on the PC
            if not STATE["model"]: return self.send(409, {"error": "no model yet"})
            from .export_web import export_model
            m, _ = export_model(weights=False)
            return self.send(200, m)
        if path == "/api/model":
            if not STATE["model"]: return self.send(409, {"error": "no model yet"})
            from .export_web import single_file
            return self.send(200, single_file(compact=self.client_address[0] != "127.0.0.1"))  # your PC: full precision; Chromebook over Wi-Fi: compact
        if path == "/api/learnhist":
            rows = [json.loads(l) for l in open("learn/history.jsonl")] if os.path.exists("learn/history.jsonl") else []
            return self.send(200, rows[-300:])
        if path == "/api/versions":
            from .versions import list_versions
            return self.send(200, list_versions())
        if path == "/api/progress": return self.send(200, {k: v for k, v in STATE["job"].items()})
        if path == "/api/runs":
            rows = [json.loads(l) for l in open("runs/index.jsonl")] if os.path.exists("runs/index.jsonl") else []
            return self.send(200, rows[-20:])
        if path.startswith("/games/") and "/" not in path[7:] and path.endswith(".html") and os.path.exists("games/" + path[7:]):
            return self.send(200, open("games/" + path[7:], "rb").read(), "text/html; charset=utf-8")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        STATE["busy"] += 1  # an update never restarts the app in the middle of a request (a game, code, a test...)
        try:
            return self.post()
        finally:
            STATE["busy"] -= 1

    def post(self):
        path, b = self.path, self.body()
        if path == "/api/decode":
            if not STATE["model"]: return self.send(409, {"error": "no model yet: type /train"})
            return self.send(200, decode(str(b.get("prompt", ""))[:400]))
        if path == "/api/train":
            if STATE["job"]["running"]: return self.send(409, {"error": busy_msg()})
            size = b.get("size", "small")
            if size == "upgrade":  # teach the current AI a bigger game language, keeping what it learned
                if not os.path.exists("models/current.pt"): return self.send(409, {"error": "There's no AI to upgrade yet. Type /train."})
                threading.Thread(target=run_job, args=(["gameai.all", "--init", "models/current.pt"],), daemon=True).start()
                return self.send(200, {"ok": True})
            if size == "coder":  # the code-writing AI (code track)
                threading.Thread(target=run_job, args=(["gameai.code.coder", "train", "--size", "small"], "coder"), daemon=True).start()
                return self.send(200, {"ok": True})
            if size not in ("tiny", "small", "base", "large"): return self.send(400, {"error": "size must be tiny, small, base, large or coder"})
            threading.Thread(target=run_job, args=(["gameai.all", "--size", size],), daemon=True).start()
            return self.send(200, {"ok": True})
        if path == "/api/active":  # you typed something in the app: you're here, so the autopilot doesn't start long work
            STATE["last_use"] = time.time(); return self.send(200, {"ok": True})
        if path == "/api/answer":
            from . import autopilot
            try: autopilot.answer(str(b.get("id", "")), str(b.get("choice", "")))
            except (ValueError, KeyError) as e: return self.send(400, {"error": str(e)})
            return self.send(200, {"ok": True, "autopilot": autopilot_status()})
        if path == "/api/qlog":  # which parts you care about when you didn't mention them (clarify.py learns from it)
            os.makedirs("memory", exist_ok=True)
            row = {"field": str(b.get("field", ""))[:20], "cared": bool(b.get("cared")), "time": time.strftime("%Y-%m-%d %H:%M")}
            open(os.path.join("memory", "questions.jsonl"), "a", encoding="utf-8").write(json.dumps(row) + "\n")
            return self.send(200, {"ok": True})
        if path == "/api/storage":
            from . import storage
            if STATE["job"]["running"]: return self.send(409, {"error": "A job is running; cleaning waits until it's done (it also runs by itself every day)."})
            freed = storage.clean(log=lambda m: None)
            return self.send(200, {"freed_gb": round(freed / 2**30, 2), "text": storage.text_report()})
        if path == "/api/auto":
            from . import autopilot
            st = autopilot.load(); st["on"] = bool(b.get("on", True))
            if st["on"] and (st["answers"].get("when") or {}).get("choice", "").startswith("Not now"): st["answers"].pop("when")
            autopilot.save(st); return self.send(200, {"ok": True, "autopilot": autopilot_status()})
        if path == "/api/phase0":  # measurements before choosing the base model (see gameai/phase0)
            if STATE["job"]["running"]: return self.send(409, {"error": busy_msg()})
            what = str(b.get("what", "")).strip() or "help"
            if what not in ("hardware", "fullstack", "profile", "licences", "bakeoff", "independence", "report", "all"):
                return self.send(400, {"error": "use /phase0 hardware, fullstack, profile, licences, bakeoff, independence, report or all"})
            extra = [x for x in str(b.get("args", "")).split() if re.fullmatch(r"[\w.\-]+", x)][:8]
            threading.Thread(target=run_job, args=(["gameai.phase0", what, *extra], "phase0"), daemon=True).start()
            return self.send(200, {"ok": True})
        if path == "/api/learn":
            if b.get("on", True):
                if STATE["job"]["running"]: return self.send(409, {"error": busy_msg()})
                if not STATE["model"]: return self.send(409, {"error": "train a first AI with /train"})
                threading.Thread(target=run_job, args=(["gameai.learn"], "learn"), daemon=True).start()
                return self.send(200, {"ok": True})
            os.makedirs("learn", exist_ok=True); open("learn/STOP", "w").close()
            if os.path.exists("learn/RESUME"): os.remove("learn/RESUME")
            return self.send(200, {"ok": True, "stopping": True})
        if path == "/api/share":
            if b.get("on", True):
                url, pin = start_share()
                return self.send(200, {"url": url, "pin": pin})
            STATE["sharing"] = False
            save_share()
            return self.send(200, {"ok": True})
        if path == "/api/teacher":
            if STATE["job"]["running"]: return self.send(409, {"error": busy_msg()})
            n = max(50, min(int(b.get("n") or 2000), 100000))
            threading.Thread(target=run_job, args=(["gameai.teacher", "--n", str(n)], "teacher"), daemon=True).start()
            return self.send(200, {"ok": True, "n": n})
        if path == "/api/playlog":
            os.makedirs("memory", exist_ok=True)
            with open("memory/playtests.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(dict(b, time=time.time())) + "\n")
            return self.send(200, {"ok": True})
        if path in ("/api/correct", "/api/rate"):
            from .recipe import OUT_FIELDS, is_valid_out
            prompt, rec = str(b.get("prompt", "")).strip()[:240], b.get("recipe") or {}
            from .recipe import FIELDS
            rec = {f: str(rec.get(f) or ("auto" if f in FIELDS else "none")) for f in OUT_FIELDS}
            if not prompt or not is_valid_out(rec): return self.send(400, {"error": "that answer doesn't fit the game language"})
            os.makedirs("memory", exist_ok=True)
            row = json.dumps({"prompt": prompt, "recipe": rec, "time": time.time()}) + "\n"
            if path == "/api/rate":
                if int(b.get("rating") or 0) < 4: return self.send(200, {"ok": True, "kept": False})
                open("memory/confirmed.jsonl", "a", encoding="utf-8").write(row)
                return self.send(200, {"ok": True, "kept": True})
            n = sum(1 for _ in open("memory/corrections.jsonl", encoding="utf-8")) if os.path.exists("memory/corrections.jsonl") else 0
            n += sum(1 for _ in open("memory/corrections_test.jsonl", encoding="utf-8")) if os.path.exists("memory/corrections_test.jsonl") else 0
            test = n % 5 == 4  # every 5th one is kept aside to test the AI on, never trained on
            open("memory/corrections_test.jsonl" if test else "memory/corrections.jsonl", "a", encoding="utf-8").write(row)
            return self.send(200, {"ok": True, "test": test, "count": n + 1})
        if path == "/api/rollback":
            from .versions import rollback
            try:
                n = rollback(int(b["n"]) if b.get("n") else None)
            except (ValueError, KeyError) as e:
                return self.send(400, {"error": str(e)})
            with STATE["lock"]: load_model()
            return self.send(200, {"ok": True, "version": n})
        if path == "/api/code":
            if not os.path.exists("models/coder.pt"): return self.send(409, {"error": "No coder yet. Type /train coder first (about 10-20 minutes on your GPU)."})
            text = str(b.get("text", "")).strip()[:160]
            if not text: return self.send(400, {"error": "say what the enemies should do"})
            try:
                with STATE["lock"]:
                    from .code import coder
                    dev = "cuda" if torch_ok() else "cpu"
                    if STATE.get("coder_mtime") != os.path.getmtime("models/coder.pt"):
                        STATE["coder"] = coder.load("models/coder.pt", dev); STATE["coder_mtime"] = os.path.getmtime("models/coder.pt")
                    m, tok, _ = STATE["coder"]
                    return self.send(200, coder.write(m, tok, text, dev))
            except Exception as e:
                return self.send(500, {"error": f"The coder hit a problem: {e}"})
        if path == "/api/stop":
            return self.send(200, {"ok": True, "stopped": kill_job()})
        if path == "/api/autostart":
            if os.name != "nt": return self.send(400, {"error": "start-with-Windows only works on Windows"})
            f = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "GameAI.bat")
            if b.get("on", True):
                root = os.path.abspath(".")
                open(f, "w", newline="").write(f'@echo off\r\ncd /d "{root}"\r\nstart "" "{root}\\.venv\\Scripts\\pythonw.exe" -m gameai.launch --background\r\n')
            elif os.path.exists(f):
                os.remove(f)
            return self.send(200, {"ok": True, "on": bool(b.get("on", True))})
        if path == "/api/quit":
            threading.Thread(target=quit_soon, daemon=True).start()
            return self.send(200, {"ok": True})
        if path == "/api/update":
            from .update import check, rollback
            if b.get("rollback"):
                if STATE["job"]["running"]: return self.send(409, {"error": busy_msg()})
                r = rollback()
                if r.get("ok"): threading.Thread(target=restart_soon, daemon=True).start()
                return self.send(200, r)
            r = check(install=not STATE["job"]["running"], quiet=True)
            if r.get("installed"): STATE["update_available"] = None; threading.Thread(target=restart_soon, daemon=True).start()
            return self.send(200, r)
        if path == "/api/updates":
            from .update import MODES, save_config
            m = str(b.get("mode", ""))
            if m not in MODES: return self.send(400, {"error": "use instant, manual or off"})
            save_config(mode=m)
            if m != "manual": STATE["update_available"] = None
            return self.send(200, {"ok": True, "mode": m})
        if path == "/api/test":
            if STATE["job"]["running"]: return self.send(409, {"error": busy_msg()})
            threading.Thread(target=run_job, args=(["gameai.bench"],), daemon=True).start()
            return self.send(200, {"ok": True})
        if path == "/api/save":
            r, name = b.get("recipe"), "".join(ch for ch in str(b.get("name") or "") if ch.isalnum() or ch in "-_")
            if not isinstance(r, dict): return self.send(400, {"error": "no game"})
            name = name or f"{r.get('player', 'game')}-{r.get('theme', '')}-{time.strftime('%H%M%S')}"
            os.makedirs("games", exist_ok=True)
            open(f"games/{name}.html", "w", encoding="utf-8").write(game_html(r))
            return self.send(200, {"path": os.path.abspath(f"games/{name}.html"), "url": f"/games/{name}.html"})
        if path == "/api/chromebook":
            if not STATE["model"]: return self.send(409, {"error": "no model yet"})
            from .export_web import main as ex
            _, pin = start_share()
            ex(pc={"urls": share_urls(), "pin": pin})
            folder = os.path.abspath("for-chromebook")
            if os.name == "nt": os.startfile(folder)
            return self.send(200, {"path": os.path.join(folder, "GameAI.html")})
        if path == "/api/folder":
            folder = os.path.abspath(str(b.get("which", "games")) if b.get("which") in ("games", "for-chromebook", "runs") else ".")
            os.makedirs(folder, exist_ok=True)
            if os.name == "nt": os.startfile(folder)
            return self.send(200, {"path": folder})
        self.send(404, {"error": "not found"})


def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1)); return s.getsockname()[0]  # no data is sent; this just picks the Wi-Fi address
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class Shared(H):
    """The same app for other devices on your Wi-Fi (your Chromebook), protected by a 4-digit code."""
    LOCAL_ONLY = ("/api/share", "/api/folder", "/api/chromebook", "/api/quit", "/api/autostart", "/api/updates")

    def allowed(self):
        from http.cookies import SimpleCookie
        from urllib.parse import urlparse, parse_qs
        pin = STATE.get("pin")
        q = parse_qs(urlparse(self.path).query).get("pin", [""])[0]
        c = SimpleCookie(self.headers.get("Cookie", "")).get("gameai_pin")
        if pin and (q == pin or (c and c.value == pin)):
            self._set_cookie = q == pin
            return True
        return False

    def send(self, code, body, ctype="application/json"):
        if getattr(self, "_set_cookie", False):
            self._extra = True
        super().send(code, body, ctype)

    def end_headers(self):
        if getattr(self, "_set_cookie", False):
            self.send_header("Set-Cookie", f"gameai_pin={STATE['pin']}; Path=/; Max-Age=31536000; SameSite=Strict")
        super().end_headers()

    def gate(self):
        if not STATE.get("sharing"):
            self.send(403, "Sharing is off. Type /share on in the Game AI app on your PC.", "text/plain"); return False
        if not self.allowed():
            self.send(403, "<!doctype html><meta name=viewport content='width=device-width'><body style='background:#0b0d16;color:#e6e8f2;font:16px monospace;padding:30px'>"
                      "<h2 style='color:#b6f24a'>Game AI</h2><form>Enter the 4-digit code shown on your PC: <input name=pin inputmode=numeric autofocus style='font:inherit'> <button>Connect</button></form>", "text/html"); return False
        return True

    def ping(self):
        """Lets GameAI.html on your Chromebook find out if your PC is on (says nothing else)."""
        body = json.dumps({"gameai": True, "sharing": bool(STATE.get("sharing"))}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", str(len(body))); BaseHTTPRequestHandler.end_headers(self); self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET"); self.send_header("Access-Control-Allow-Private-Network", "true")
        BaseHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/ping": return self.ping()
        if self.gate(): super().do_GET()

    def do_POST(self):
        if not self.gate(): return
        if self.path in self.LOCAL_ONLY: return self.send(403, {"error": "only from the PC itself"})
        super().do_POST()


SHARE_FILE = "memory/share.json"


def save_share():
    os.makedirs("memory", exist_ok=True)
    json.dump({"on": bool(STATE.get("sharing")), "pin": STATE.get("pin")}, open(SHARE_FILE, "w"))


def share_urls():
    import socket
    return [f"http://{lan_ip()}:{SHARE_PORT}/", f"http://{socket.gethostname().lower()}.local:{SHARE_PORT}/"]


def start_share():
    import secrets
    if not STATE.get("share_srv"):
        srv = ThreadingHTTPServer(("0.0.0.0", SHARE_PORT), Shared)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        STATE["share_srv"] = srv
    STATE["pin"] = STATE.get("pin") or f"{secrets.randbelow(10000):04d}"
    STATE["sharing"] = True
    save_share()  # remembered: sharing comes back on (same code) every time the app starts
    return f"http://{lan_ip()}:{SHARE_PORT}/", STATE["pin"]


def open_window(url):
    """Open as its own app window (no tabs or address bar) when Edge or Chrome is available."""
    cands = [shutil.which("msedge"), shutil.which("chrome"),
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for c in cands:
        if c and os.path.exists(c):
            subprocess.Popen([c, f"--app={url}", "--window-size=1500,920"]); return
    webbrowser.open(url)


RESTART = 42  # start.bat restarts the app when it exits with this code (after an update)


def restart_soon():
    time.sleep(1.5)  # let the page get its answer first
    t0 = time.time()
    while (STATE["busy"] or STATE["job"]["running"]) and time.time() - t0 < 3600:
        time.sleep(1)  # something started meanwhile: let it finish first (the new code is already in place)
    os._exit(RESTART)


def busy_msg():
    what = {"teacher": "The teacher is still writing examples", "learn": "Always-on learning is running"}.get(STATE["job"].get("kind"), "Training is still running")
    return f"{what}. Type /stop to cancel it, then try again."


def torch_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


JOB_MODULES = ("gameai.train", "gameai.teacher", "gameai.learn", "gameai.all", "gameai.improve", "gameai.bench", "gameai.code.coder", "gameai.phase0")


def leftover_jobs():
    """Game AI jobs still running from before (the app was closed or crashed while they ran). They keep using
    the GPU and slow everything down, and nothing shows them. -> list of psutil processes"""
    try:
        import psutil
    except ImportError:
        return []
    mine, here, out = os.getpid(), os.path.abspath("."), []
    p = STATE.get("proc")
    ours = {p.pid} if p and p.poll() is None else set()
    if ours:
        try: ours |= {c.pid for c in psutil.Process(p.pid).children(recursive=True)}
        except Exception: pass
    for q in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            cmd = " ".join(q.info["cmdline"] or [])
            if q.pid == mine or q.pid in ours or q.info.get("cwd") != here: continue
            if any(f"-m {m}" in cmd for m in JOB_MODULES): out.append(q)
        except Exception:
            continue
    return out


def kill_leftovers():
    n = 0
    for q in leftover_jobs():
        try: q.kill(); n += 1
        except Exception: pass
    return n


def kill_job():
    """Stop the running job and everything it started (like the teacher inside learning), and any leftover jobs."""
    old = kill_leftovers(); STATE["leftovers"] = 0
    p = STATE.get("proc")
    if not (p and p.poll() is None): return old > 0
    STATE["job"]["stopped"] = True
    if os.path.exists("learn/RESUME"): os.remove("learn/RESUME")
    if os.name == "nt": subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        import signal
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: p.kill()
    return True


def quit_soon():
    """Stop everything: the training job (if any) and the app."""
    time.sleep(0.5)
    kill_job()
    os._exit(0)


def watchdog():
    """With the hidden launcher there is no black window, so the app quits by itself
    about 2 minutes after you close its window (unless it is training or learning)."""
    while True:
        time.sleep(5)
        seen = STATE.get("seen")
        if seen and time.time() - seen > 150 and not STATE["job"]["running"] and not autopilot_has_work():
            os._exit(0)


def auto_update():
    """Every 15 seconds, ask GitHub (tiny request) whether there's a newer version. If so, install it and
    restart; the open window reloads by itself. During training it waits until the job is done, and it never
    restarts in the middle of a request. Offline, it waits longer and longer between tries (up to 15 minutes).
    update.json "mode": instant (install by itself), manual (only tell you), off (never go online)."""
    from .update import check, config, head_sha, local_version, mode
    wait = 15
    while True:
        time.sleep(wait)
        cfg = config()
        if not cfg or mode(cfg) == "off":
            wait = 15; STATE["update_available"] = None; continue
        try:
            sha = head_sha(cfg)
            wait = 15
        except Exception:
            wait = min(wait * 2, 900)  # offline: back off
            continue
        if not sha or sha in (local_version().get("sha"), cfg.get("skip")):
            STATE["update_waiting"] = False; STATE["update_available"] = None
            continue
        if mode(cfg) == "manual":
            STATE["update_available"] = sha[:7]  # the page asks you; /update installs it
            continue
        if STATE["busy"]:
            continue  # something is being made right now: check again in 15 seconds
        if STATE["job"]["running"]:
            STATE["update_waiting"] = True  # the page tells you it will install after the job
            if STATE["job"].get("kind") == "learn" and not os.path.exists("learn/RESUME"):
                os.makedirs("learn", exist_ok=True)
                open("learn/RESUME", "w").close()  # learning pauses after this round, updates, then carries on
                open("learn/STOP", "w").close()
            continue
        try:
            r = check(quiet=True)
            if r.get("installed"):
                STATE["updating"] = r.get("message") or "new version"
                restart_soon()
        except Exception:
            pass


def autopilot_status():
    try:
        from . import autopilot
        return autopilot.status()
    except Exception as e:
        return {"on": False, "label": f"autopilot unavailable: {e}"}


def autopilot_has_work():
    try:
        from . import autopilot
        st = autopilot.load(); sid, _, _ = autopilot.next_step(st)
        return bool(st.get("on", True) and sid)
    except Exception:
        return False


def autopilot_loop():
    """Every 20 seconds: if you're away (no typing in the app for 10 minutes) and nothing else runs, do the next step."""
    from . import autopilot
    STATE.setdefault("last_use", time.time())
    while True:
        time.sleep(20)
        try:
            if STATE["job"]["running"] or STATE.get("updating"): continue
            st = autopilot.load()
            sid, label, cmd = autopilot.next_step(st)
            if not sid or not autopilot.may_start(st, time.time() - STATE.get("last_use", time.time())): continue
            STATE["auto_step"] = sid
            run_job(cmd, "autopilot")
            j = STATE["job"]
            if j.get("stopped"): continue  # you stopped it: not a failure, it carries on later
            autopilot.finished(sid, bool(j.get("ok")), " / ".join(l for l in j.get("log", [])[-3:] if l))
            tidy(force=True)  # downloads, checkpoints and leftovers from that step
        except Exception as e:
            STATE["auto_error"] = str(e)
        finally:
            STATE["auto_step"] = None
            if not STATE["job"]["running"]: tidy()


def tidy(force=False):
    """storage cleaning (storage.py): after every autopilot step, and once a day; never while a job runs"""
    if STATE["job"]["running"]: return
    if not force and time.time() - STATE.get("tidied", 0) < 86400: return
    STATE["tidied"] = time.time()
    try:
        from . import storage
        freed = storage.clean(log=lambda m: None)
        if freed > 200 * 2**20: STATE["note"] = f"Freed {freed / 2**30:.1f} GB of storage (old downloads, checkpoints and leftovers)."
    except Exception as e:
        STATE["tidy_error"] = str(e)[:200]


def main():
    from .update import local_version
    STATE["started_version"] = local_version()
    threading.Thread(target=autopilot_loop, daemon=True).start()
    STATE["leftovers"] = len(leftover_jobs())  # the page tells you, and /stop ends them
    load_model()
    if not os.environ.get("GAMEAI_NO_UPDATE"): threading.Thread(target=auto_update, daemon=True).start()
    if os.path.exists(SHARE_FILE):
        try:
            sh = json.load(open(SHARE_FILE))
            if sh.get("on"):
                STATE["pin"] = sh.get("pin"); start_share()
        except Exception:
            pass
    if os.path.exists("learn/RESUME"):  # learning paused for an update: carry on
        os.remove("learn/RESUME")
        if STATE["model"]: threading.Thread(target=run_job, args=(["gameai.learn"], "learn"), daemon=True).start()
    if "--auto-quit" in sys.argv and "--stay" not in sys.argv: threading.Thread(target=watchdog, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Game AI app running at {url}")
    if "--no-window" not in sys.argv:
        open_window(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
