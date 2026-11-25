"""The offline-independence test: with every connection to the internet refused, every part of the AI must still work.

Inside this program, any connection to anything other than this PC (127.0.0.1) is refused, and updates and
downloads are switched off. Then: the game AI reads a request, the playtesters prove a level, the code sandbox runs
code, the app's page is built, and (if a llama.cpp model is installed) it answers through the runtime.
For the strongest version, also run it with your Wi-Fi off: /phase0 independence.
"""
import os, socket

from .common import save, say, stage

LOCAL = ("127.0.0.1", "localhost", "::1")


def block_internet():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", GAMEAI_NO_UPDATE="1")
    real = socket.socket.connect

    def connect(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if isinstance(host, str) and host not in LOCAL and not host.startswith("127."):
            raise ConnectionRefusedError(f"independence test: no internet allowed (tried {host})")
        return real(self, addr)
    socket.socket.connect = connect
    real_gai = socket.getaddrinfo

    def gai(host, *a, **k):
        if host not in LOCAL and not str(host).startswith("127."):
            raise socket.gaierror(f"independence test: no internet allowed (tried to look up {host})")
        return real_gai(host, *a, **k)
    socket.getaddrinfo = gai


def check(name, fn, out):
    try:
        r = fn(); out[name] = {"ok": True, **({"detail": r} if r is not None else {})}
    except Exception as e:
        out[name] = {"ok": False, "why": str(e)[:300]}
    say(f"  {'PASS' if out[name]['ok'] else 'FAIL'}  {name}" + ("" if out[name]["ok"] else f": {out[name]['why']}"))


def run():
    stage("Offline-independence test")
    block_internet()
    out = {}
    try:
        import urllib.request
        urllib.request.urlopen("https://github.com", timeout=5)
        out["internet is blocked"] = {"ok": False, "why": "a connection got through"}
    except Exception:
        out["internet is blocked"] = {"ok": True}
    say(f"  {'PASS' if out['internet is blocked']['ok'] else 'FAIL'}  internet is blocked inside this test")

    def game_ai():
        import torch
        from ..generate import load, decode
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m, tok, _ = load("models/current.pt", dev, upgrade=True)
        p = decode(m, tok, ["a fast ninja in a snowy level with dragons"], dev)[0]
        return {k: p[k][0] for k in ("theme", "player", "enemy")}

    def playtest():
        import random
        from ..headless import Engine, random_recipe
        e = Engine(); r = e.judge(random_recipe(random.Random(3), enemy="none")); e.close()
        if not r["proven"]: raise RuntimeError("the level wasn't proven")
        return {"proven": True}

    def sandbox():
        from ..code import sandbox as S
        ok, why = S.verify("chase", "return dir(e, p);")
        if not ok: raise RuntimeError(why)

    def page():
        from ..export_web import build_page
        if len(build_page()) < 10000: raise RuntimeError("the page came out empty")

    def updater():
        from ..update import check as upd
        off = os.environ.pop("GAMEAI_NO_UPDATE", None)  # really try to update, with the internet blocked
        try: r = upd(install=True, quiet=True)
        finally:
            if off: os.environ["GAMEAI_NO_UPDATE"] = off
        if r.get("installed"): raise RuntimeError("it installed something while offline")
        return r.get("error")

    def llm():
        import glob
        from . import runtime as R
        ggufs = glob.glob(os.path.join("models", "**", "*.gguf"), recursive=True)
        if not R.exe("llama-server") or not ggufs: return "no llama.cpp model installed yet (comes in Phase 2)"
        with R.Server(ggufs[0]) as s:
            text, info = s.chat([{"role": "user", "content": "Say hi in one word."}], max_tokens=10)
        return {"answer": text[:40], "model": os.path.basename(ggufs[0])}

    for name, fn in (("game AI reads a request", game_ai), ("playtesters prove a level", playtest), ("code sandbox runs code", sandbox),
                     ("app page builds", page), ("updater stays quiet offline", updater), ("llama.cpp runtime answers", llm)):
        check(name, fn, out)
    out["passed"] = all(v["ok"] for v in out.values() if isinstance(v, dict) and "ok" in v)
    say(f"\n  {'PASSED' if out['passed'] else 'FAILED'}: everything {'works' if out['passed'] else 'did not work'} with no internet")
    save("independence", out)
    return out
