"""The debugger tool (plan: Phase 4). The tests run in a sandbox: the project's code gets no files, network, programs
or Node modules, at most 512 MB of memory and 10 seconds per test.

It runs a project's tests and says why one failed, most useful facts first:
the error, the stack trace in the project's own files (with each source line), the failing assertion (expected vs
actual), the variable values where it went wrong, and console errors. Full game state only when asked (inspect=True).

    from gameai.repair import debugger
    r = debugger.run("my_game")                    # or a {path: text} dict
    print(debugger.report(r))                      # plain text for a repair prompt (and for you)

Each language runs in its own sandbox (no files outside the project, no writing, no network, no programs):
    JavaScript  *.test.js or tests/*.js: test(name, fn), expect(x).toBe(y)..., assert(cond, msg). Other .js files load
                first (alphabetically, main/index last), with GameKit. Set globalThis.__game = g to see the game's state.
    Python      test_*.py, *_test.py or tests/*.py: functions named test_*, plain assert (both sides are shown)
    Lua         *_test.lua or tests/*.lua: test("name", function() expect(x).toBe(y) end), assert(cond, msg)
"""
import json, os, shutil, subprocess, sys, tempfile, time

from ..checks import tools

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "debug_runner.mjs")
PY_RUNNER = os.path.join(HERE, "py_runner.py")
LUA_RUNNER = os.path.join(HERE, "lua_runner.py")
KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kit", "gamekit.js")
JS = (".js", ".mjs", ".cjs")
MEMORY_MB = 512          # the most memory the tests may use
TEST_SECONDS = 10        # the longest one test may run


def lang_of_test(path):
    """'js' | 'py' | 'lua' if this file is a test file, else None"""
    p = path.replace("\\", "/"); base = os.path.basename(p)
    in_tests = "/tests/" in "/" + p
    if p.endswith(JS): return "js" if base.endswith((".test.js", ".test.mjs", ".spec.js")) or in_tests else None
    if p.endswith(".py"): return "py" if base.startswith("test_") or base.endswith("_test.py") or (in_tests and base != "__init__.py") else None
    if p.endswith(".lua"): return "lua" if base.endswith(("_test.lua", ".test.lua")) or in_tests else None
    return None


def is_test(path):
    return lang_of_test(path) is not None


def _order(scripts):
    last = lambda p: os.path.splitext(os.path.basename(p))[0] in ("main", "index", "game")
    return sorted(scripts, key=lambda p: (last(p), p))


def _sandbox(cmd, cfg, root, timeout, env):
    """run one language's runner with a time limit and a memory limit (watched from outside, so it works on Windows too)"""
    import psutil
    fd, cfg_path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump(cfg, open(cfg_path, "w"))
    t0, killed = time.time(), None
    p = subprocess.Popen(cmd + [cfg_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                         env=env, cwd=root, creationflags=tools.NO_WINDOW)
    try:
        import threading
        res = {}
        th = threading.Thread(target=lambda: res.update(zip(("out", "err"), p.communicate())), daemon=True); th.start()
        proc = psutil.Process(p.pid)
        while th.is_alive():
            th.join(0.2)
            if not th.is_alive(): break
            try: rss = proc.memory_info().rss
            except psutil.Error: rss = 0
            if rss > (MEMORY_MB + 200) * 2**20: killed = f"used more than {MEMORY_MB} MB of memory"; p.kill()
            elif time.time() - t0 > timeout: killed = f"timed out after {timeout}s: probably an endless loop"; p.kill()
        th.join(5)
    finally:
        os.remove(cfg_path)
    out, err = res.get("out") or "", res.get("err") or ""
    if killed:
        ran = [l[9:] for l in err.splitlines() if l.startswith("running: ")]
        return {"passed": max(0, len(ran) - 1), "failed": 1, "timed_out": "timed out" in killed, "load_errors": [], "console": [],
                "tests": [{"name": ran[-1] if ran else "(loading the project)", "passed": False, "error": killed}]}
    try: r = json.loads(out)
    except ValueError:
        if "heap limit" in err or "out of memory" in err.lower() or "MemoryError" in err:
            return {"passed": 0, "failed": 1, "load_errors": [], "console": [], "tests": [{"name": "(memory)", "passed": False,
                    "error": f"used more than {MEMORY_MB} MB of memory"}]}
        return {"passed": 0, "failed": 1, "load_errors": [], "console": [], "tests": [{"name": "(runner)", "passed": False,
                "error": (err or out or "the test runner stopped").strip()[-600:]}]}
    r["seconds"] = round(time.time() - t0, 2)
    return r


def _merge(results):
    out = {"passed": 0, "failed": 0, "tests": [], "load_errors": [], "console": [], "timed_out": False, "seconds": 0}
    for lang, r in results:
        for t in r.get("tests", []): t["lang"] = lang
        for k in ("passed", "failed", "seconds"): out[k] += r.get(k, 0)
        for k in ("tests", "load_errors", "console"): out[k] += r.get(k, [])
        out["timed_out"] = out["timed_out"] or bool(r.get("timed_out"))
    out["seconds"] = round(out["seconds"], 2)
    return out


def run(project, tests=None, inspect=False, timeout=30, kit=True):
    """project: a folder or {path: text}. Runs the JavaScript, Python and Lua tests it has, each in its sandbox.
    -> {passed, failed, tests: [...], load_errors, console, timed_out}"""
    tmp = None
    if isinstance(project, dict):
        tmp = tempfile.mkdtemp(prefix="gameai-debug-")
        for p, t in project.items():
            dest = os.path.join(tmp, *p.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "w", encoding="utf-8").write(t)
        root, names = tmp, list(project)
    else:
        root = project
        from .repomap import read_dir
        names = list(read_dir(project))
    try:
        tests = tests if tests is not None else sorted(n for n in names if is_test(n))
        by_lang = {lang: [t for t in tests if lang_of_test(t) == lang] for lang in ("js", "py", "lua")}
        base_env = {k: v for k, v in os.environ.items() if k.upper() in ("PATH", "SYSTEMROOT", "TEMP", "TMP")}  # nothing else of yours
        common = {"root": os.path.abspath(root), "test_timeout_ms": TEST_SECONDS * 1000, "memory_mb": MEMORY_MB}
        results = []
        if by_lang["js"] or (tests is not None and not any(by_lang.values()) and any(n.endswith(JS) for n in names)):
            node = tools.node()
            if not node: raise RuntimeError("the JavaScript sandbox needs Node.js: run  python -m gameai.checks install")
            js = [n for n in names if n.endswith(JS)]
            cfg = dict(common, scripts=_order([n for n in js if n not in by_lang["js"]]), tests=by_lang["js"], inspect=inspect, kit=KIT if kit else None)
            results.append(("js", _sandbox([node, "--experimental-vm-modules", f"--max-old-space-size={MEMORY_MB}", RUNNER], cfg, root, timeout,
                                           dict(base_env, NODE_NO_WARNINGS="1"))))
        if by_lang["py"]:
            results.append(("py", _sandbox([sys.executable, "-I", "-S", PY_RUNNER], dict(common, tests=by_lang["py"]), root, timeout,
                                           dict(base_env, PYTHONIOENCODING="utf-8"))))
        if by_lang["lua"]:
            results.append(("lua", _sandbox([sys.executable, "-I", LUA_RUNNER], dict(common, tests=by_lang["lua"]), root, timeout,
                                            dict(base_env, PYTHONIOENCODING="utf-8"))))
        return _merge(results)
    finally:
        if tmp: shutil.rmtree(tmp, ignore_errors=True)


def ok(r):
    return not r.get("failed") and not r.get("load_errors") and not r.get("timed_out")


def report(r, max_failures=5):
    """the result as text: a summary line, then each failure with its most useful facts first"""
    out = [f"tests: {r.get('passed', 0)} passed, {r.get('failed', 0)} failed" + (" (timed out)" if r.get("timed_out") else "")]
    def where(fr):
        line = f"  at {fr['function']} ({fr['file']}:{fr['line']})"
        if fr.get("source"): line += f"\n      {fr['source'][:160]}"
        if fr.get("variables"): line += "\n      values: " + ", ".join(f"{k} = {v}" for k, v in fr["variables"].items())[:400]
        if fr.get("this"): line += "\n      this: " + ", ".join(f"{k} = {v}" for k, v in fr["this"].items())[:300]
        return line
    for e in r.get("load_errors", []):
        out.append(f"\nloading {e['file']} failed: {e['error']}"); out += [where(f) for f in e.get("stack", [])[:4]]
    for t in [t for t in r.get("tests", []) if not t.get("passed")][:max_failures]:
        out.append(f"\nFAILED: {t['name']}" + (f" ({t['file']})" if t.get("file") else ""))
        out.append(f"  {t['error']}")
        if t.get("assertion"): out.append(f"  expected: {t['assertion']['expected']}\n  actual:   {t['assertion']['actual']}")
        out += [where(f) for f in t.get("stack", [])[:4]]
        if t.get("game"): out.append("  game: " + json.dumps(t["game"]))
    if r.get("console"):
        out.append("\nconsole:")
        out += [f"  {c['level']}: {c['message']}" + (f" ({c['file']}:{c['line']})" if c.get("file") else "") for c in r["console"][:8]]
    return "\n".join(out)
