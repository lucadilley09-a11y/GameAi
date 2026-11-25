"""The debugger tool (plan: Phase 4): runs a project's tests and says why one failed, most useful facts first:
the error, the stack trace in the project's own files (with each source line), the failing assertion (expected vs
actual), the variable values where it went wrong, and console errors. Full game state only when asked (inspect=True).

    from gameai.repair import debugger
    r = debugger.run("my_game")                    # or a {path: text} dict
    print(debugger.report(r))                      # plain text for a repair prompt (and for you)

Tests are JavaScript files named *.test.js (or in a tests/ folder) using test(name, fn), expect(x).toBe(y) and
friends, or assert(cond, msg). Other .js files are loaded first (alphabetically, main/index last), with GameKit.
A test can set globalThis.__game = g so a failure shows the game's state.
"""
import json, os, shutil, subprocess, tempfile, time

from ..checks import tools

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_runner.mjs")
KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kit", "gamekit.js")
JS = (".js", ".mjs", ".cjs")


def is_test(path):
    p = path.replace("\\", "/")
    return p.endswith((".test.js", ".test.mjs", ".spec.js")) or "/tests/" in "/" + p or p.startswith("tests/")


def _order(scripts):
    last = lambda p: os.path.splitext(os.path.basename(p))[0] in ("main", "index", "game")
    return sorted(scripts, key=lambda p: (last(p), p))


def run(project, tests=None, inspect=False, timeout=30, kit=True):
    """project: a folder or {path: text}. -> {passed, failed, tests: [...], load_errors, console, timed_out}"""
    node = tools.node()
    if not node: raise RuntimeError("the debugger needs Node.js: run  python -m gameai.checks install")
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
        js = [n for n in names if n.endswith(JS)]
        tests = tests if tests is not None else sorted(n for n in js if is_test(n))
        scripts = _order([n for n in js if n not in tests])
        cfg = {"root": os.path.abspath(root), "scripts": scripts, "tests": tests, "inspect": inspect, "kit": KIT if kit else None}
        fd, cfg_path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        json.dump(cfg, open(cfg_path, "w"))
        t0 = time.time()
        try:
            p = subprocess.run([node, RUNNER, cfg_path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, creationflags=tools.NO_WINDOW)
        except subprocess.TimeoutExpired as e:
            ran = [l[5:] for l in (e.stderr or b"").decode("utf-8", "replace").splitlines() if l.startswith("ran: ")]
            return {"passed": len(ran), "failed": 1, "timed_out": True, "load_errors": [], "console": [],
                    "tests": [{"name": "(the test after " + (repr(ran[-1]) if ran else "loading") + ")", "passed": False,
                               "error": f"timed out after {timeout}s: probably an endless loop"}]}
        finally:
            os.remove(cfg_path)
        try: r = json.loads(p.stdout)
        except ValueError:
            return {"passed": 0, "failed": 1, "load_errors": [], "console": [], "tests": [{"name": "(runner)", "passed": False,
                    "error": (p.stderr or p.stdout or "the test runner stopped").strip()[-600:]}]}
        r["seconds"] = round(time.time() - t0, 2)
        return r
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
