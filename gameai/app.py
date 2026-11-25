"""The Game AI app: the terminal in its own window, with your AI running on the GPU.

    python -m gameai.app        (or double-click start.bat)

Runs a small local server (only reachable from this PC) and opens the terminal in an
app window (Microsoft Edge/Chrome app mode, or your browser if neither is found)."""
import json, os, shutil, subprocess, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

PORT = 8765
SHARE_PORT = 8766
STATE = {"mtime": 0, "model": None, "tok": None, "device": "cpu", "lock": threading.Lock(),
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
    v = local_version()
    return {"app": True, "version": (v.get("sha") or "dev")[:7], "update_msg": v.get("message"), "sharing": bool(STATE.get("sharing")), "note": STATE.get("note"), "loaded": m is not None, "params": m.num_params() if m else 0, "gpu": gpu, "fresh": s,
            "training": STATE["job"]["running"], "learning": STATE["job"].get("kind") == "learn" and STATE["job"]["running"]}


def decode(prompt):
    from .generate import decode as dec
    with STATE["lock"]:
        hot_swap()
        out = dec(STATE["model"], STATE["tok"], [prompt], STATE["device"], gate=0.0)[0]
    return {f: {"value": v, "conf": c} for f, (v, c) in out.items()}


def run_job(args, kind="job"):
    j = STATE["job"]
    j.update(kind=kind, running=True, stage="Starting", step=0, total=0, loss=None, evals=[], log=[], ok=None, started=time.time())
    env = dict(os.environ, GAMEAI_PROGRESS="1", PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    STATE["proc"] = p
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("@@P"):
            d = json.loads(line[3:]); j.update(step=d["step"], total=d["total"], loss=d["loss"])
        elif line.startswith("@@L"):
            pass  # round results: the page reads learn/history.jsonl
        elif line.startswith("@@E"):
            j["evals"].append(json.loads(line[3:]))
        else:
            if line.startswith("==="): j["stage"] = line.strip("= ")
            j["log"] = (j["log"] + [line])[-40:]
    j["ok"] = p.wait() == 0
    with STATE["lock"]:
        load_model()
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
        if path == "/api/status":
            if self.client_address[0] == "127.0.0.1": STATE["seen"] = time.time()
            return self.send(200, status())
        if path == "/api/model":
            if not STATE["model"]: return self.send(409, {"error": "no model yet"})
            from .export_web import single_file
            return self.send(200, single_file())
        if path == "/api/learnhist":
            rows = [json.loads(l) for l in open("learn/history.jsonl")] if os.path.exists("learn/history.jsonl") else []
            return self.send(200, rows[-300:])
        if path == "/api/progress": return self.send(200, {k: v for k, v in STATE["job"].items()})
        if path == "/api/runs":
            rows = [json.loads(l) for l in open("runs/index.jsonl")] if os.path.exists("runs/index.jsonl") else []
            return self.send(200, rows[-20:])
        if path.startswith("/games/") and "/" not in path[7:] and path.endswith(".html") and os.path.exists("games/" + path[7:]):
            return self.send(200, open("games/" + path[7:], "rb").read(), "text/html; charset=utf-8")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        path, b = self.path, self.body()
        if path == "/api/decode":
            if not STATE["model"]: return self.send(409, {"error": "no model yet: type /train"})
            return self.send(200, decode(str(b.get("prompt", ""))[:400]))
        if path == "/api/train":
            if STATE["job"]["running"]: return self.send(409, {"error": "already training"})
            size = b.get("size", "small")
            if size not in ("tiny", "small", "base", "large"): return self.send(400, {"error": "size must be tiny, small, base or large"})
            threading.Thread(target=run_job, args=(["gameai.all", "--size", size],), daemon=True).start()
            return self.send(200, {"ok": True})
        if path == "/api/learn":
            if b.get("on", True):
                if STATE["job"]["running"]: return self.send(409, {"error": "busy: wait for the current job to finish"})
                if not STATE["model"]: return self.send(409, {"error": "train a first AI with /train"})
                threading.Thread(target=run_job, args=(["gameai.learn"], "learn"), daemon=True).start()
                return self.send(200, {"ok": True})
            os.makedirs("learn", exist_ok=True); open("learn/STOP", "w").close()
            return self.send(200, {"ok": True, "stopping": True})
        if path == "/api/share":
            if b.get("on", True):
                url, pin = start_share()
                return self.send(200, {"url": url, "pin": pin})
            STATE["sharing"] = False
            return self.send(200, {"ok": True})
        if path == "/api/teacher":
            if STATE["job"]["running"]: return self.send(409, {"error": "busy: wait for the current job to finish"})
            n = max(50, min(int(b.get("n") or 2000), 100000))
            threading.Thread(target=run_job, args=(["gameai.teacher", "--n", str(n)], "teacher"), daemon=True).start()
            return self.send(200, {"ok": True, "n": n})
        if path == "/api/playlog":
            os.makedirs("memory", exist_ok=True)
            with open("memory/playtests.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(dict(b, time=time.time())) + "\n")
            return self.send(200, {"ok": True})
        if path == "/api/quit":
            threading.Thread(target=quit_soon, daemon=True).start()
            return self.send(200, {"ok": True})
        if path == "/api/update":
            from .update import check
            r = check(install=not STATE["job"]["running"], quiet=True)
            if r.get("installed"): threading.Thread(target=restart_soon, daemon=True).start()
            return self.send(200, r)
        if path == "/api/test":
            if STATE["job"]["running"]: return self.send(409, {"error": "busy"})
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
            ex()
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
    LOCAL_ONLY = ("/api/share", "/api/folder", "/api/chromebook", "/api/quit")

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

    def do_GET(self):
        if self.gate(): super().do_GET()

    def do_POST(self):
        if not self.gate(): return
        if self.path in self.LOCAL_ONLY: return self.send(403, {"error": "only from the PC itself"})
        super().do_POST()


def start_share():
    import secrets
    if not STATE.get("share_srv"):
        srv = ThreadingHTTPServer(("0.0.0.0", SHARE_PORT), Shared)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        STATE["share_srv"] = srv
    STATE["pin"] = STATE.get("pin") or f"{secrets.randbelow(10000):04d}"
    STATE["sharing"] = True
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
    os._exit(RESTART)


def quit_soon():
    """Stop everything: the training job (if any) and the app."""
    time.sleep(0.5)
    p = STATE.get("proc")
    if p and p.poll() is None:  # also stop anything it started (like the teacher)
        if os.name == "nt": subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else: p.kill()
    os._exit(0)


def watchdog():
    """With the hidden launcher there is no black window, so the app quits by itself
    about 2 minutes after you close its window (unless it is training or learning)."""
    while True:
        time.sleep(5)
        seen = STATE.get("seen")
        if seen and time.time() - seen > 150 and not STATE["job"]["running"]:
            os._exit(0)


def auto_update():
    """Every 10 minutes: install a newer version from GitHub, then restart (never during a training job)."""
    from .update import check
    while True:
        time.sleep(600)
        if STATE["job"]["running"]:
            continue
        try:
            if check(quiet=True).get("installed"):
                restart_soon()
        except Exception:
            pass


def main():
    load_model()
    if not os.environ.get("GAMEAI_NO_UPDATE"): threading.Thread(target=auto_update, daemon=True).start()
    if "--auto-quit" in sys.argv: threading.Thread(target=watchdog, daemon=True).start()
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
