"""The Python sandbox's test runner (plan: Phase 4, code sandboxes per language). Started by debugger.py as
    python -I -S py_runner.py config.json
so it ignores your installed packages and settings. Before the project's code runs, an audit hook (Python's own
security hooks, which can't be removed once added) blocks files outside the project, writing any file, the network,
starting programs and loading native code. debugger.py adds the time and memory limits from outside.

Tests: files named test_*.py, *_test.py or inside tests/, with functions named test_*. Plain `assert` works; when an
assert fails, both sides of a comparison are shown (expected vs actual), plus the variable values where it failed.
Output: one JSON object on stdout, in the same shape as the JavaScript runner's.
"""
import builtins, contextlib, importlib.util, io, json, linecache, os, sys, time, traceback

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
ROOT = os.path.realpath(cfg["root"])
STDLIB = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.__file__)) if p}
SELF = os.path.realpath(__file__)
out = {"load_errors": [], "tests": [], "console": [], "passed": 0, "failed": 0}
REAL_STDOUT = sys.stdout

BLOCKED_EVENTS = ("socket.", "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
                  "os.kill", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate", "os.symlink", "os.link",
                  "os.chdir", "os.putenv", "os.unsetenv", "os.startfile", "shutil.", "ctypes.", "winreg.", "urllib.Request",
                  "webbrowser.open", "_winapi.", "msvcrt.")
BLOCKED_MODULES = {"ctypes", "_ctypes", "socket", "_socket", "ssl", "subprocess", "_posixsubprocess", "_winapi", "multiprocessing",
                   "_multiprocessing", "winreg", "msvcrt", "pty", "fcntl", "mmap", "http", "urllib", "ftplib", "smtplib", "webbrowser"}


def _inside(path, roots):
    try: p = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError): return False
    return any(p == r or p.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def hook(event, args):
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(path, int): return  # an already-open file (stdin/stdout)
        if mode and any(c in str(mode) for c in "wax+"): raise PermissionError(f"the sandbox doesn't allow writing files ({path})")
        if not _inside(path, STDLIB | {ROOT, SELF}): raise PermissionError(f"the sandbox only allows reading the project's own files ({path})")
    elif event == "import" and args[0].split(".")[0] in BLOCKED_MODULES:
        raise ImportError(f"'{args[0]}' isn't allowed in the sandbox (no network, programs or native code)")
    elif event.startswith(BLOCKED_EVENTS):
        raise PermissionError(f"the sandbox doesn't allow {event}")


def rel(path):
    p = os.path.realpath(path)
    return os.path.relpath(p, ROOT).replace(os.sep, "/") if p.startswith(ROOT + os.sep) else None


def show(v):
    try: s = repr(v)
    except Exception: s = f"<{type(v).__name__}>"
    return s if len(s) <= 120 else s[:117] + "..."


def frames_of(tb):
    frames = []
    for fr, line in traceback.walk_tb(tb):
        file = rel(fr.f_code.co_filename)
        if not file: continue
        frames.append({"function": fr.f_code.co_name if fr.f_code.co_name != "<module>" else "(top level)", "file": file, "line": line,
                       "source": linecache.getline(fr.f_code.co_filename, line).strip(), "_frame": fr})
    frames = frames[::-1][:6]  # innermost first, like the JavaScript runner
    for f in frames[:2]:
        fr = f["_frame"]
        names = [k for k, v in fr.f_locals.items() if not k.startswith("__") and (k in fr.f_code.co_varnames or not callable(v))]
        f["variables"] = {k: show(fr.f_locals[k]) for k in names[:15]}
    for f in frames: f.pop("_frame", None)
    return frames


def assertion_sides(tb):
    """for a bare `assert a == b` that failed: the values of both sides, worked out in the failing frame"""
    import ast
    last = None
    for fr, line in traceback.walk_tb(tb):
        if rel(fr.f_code.co_filename): last = (fr, line)
    if not last: return None
    fr, line = last
    src = linecache.getline(fr.f_code.co_filename, line).strip()
    try: node = ast.parse(src).body[0]
    except (SyntaxError, IndexError): return None
    if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1: return None
    try:
        left = eval(compile(ast.Expression(node.test.left), "<assert>", "eval"), fr.f_globals, fr.f_locals)
        right = eval(compile(ast.Expression(node.test.comparators[0]), "<assert>", "eval"), fr.f_globals, fr.f_locals)
    except Exception:
        return None
    return {"actual": show(left), "expected": show(right)}


def failure(e):
    r = {"error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__, "stack": frames_of(e.__traceback__)}
    if isinstance(e, AssertionError):
        sides = assertion_sides(e.__traceback__)
        if sides:
            r["assertion"] = sides
            if not str(e): r["error"] = f"AssertionError: expected {sides['expected']}, got {sides['actual']}"
    if isinstance(e, SyntaxError) and e.filename and rel(e.filename):
        r["stack"] = [{"function": "(syntax)", "file": rel(e.filename), "line": e.lineno or 1, "source": (e.text or "").strip()}]
    return r


def load(path):
    name = "t_" + rel(path).replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def limit_memory(mb):
    """a hard memory cap set by the system itself, so a huge allocation fails with MemoryError instead of filling your RAM"""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes as W
        class IO(ctypes.Structure): _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]
        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", W.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]
        class Ext(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = W.HANDLE; k.GetCurrentProcess.restype = W.HANDLE
        job = k.CreateJobObjectW(None, None)
        info = Ext(); info.Basic.LimitFlags = 0x100 | 0x8  # process memory limit, and at most 1 process (no child programs)
        info.Basic.ActiveProcessLimit = 1; info.ProcessMemoryLimit = mb * 2**20
        if not k.SetInformationJobObject(W.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)): raise OSError(ctypes.get_last_error())
        if not k.AssignProcessToJobObject(W.HANDLE(job), W.HANDLE(k.GetCurrentProcess())): raise OSError(ctypes.get_last_error())
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mb * 2**20, mb * 2**20))


try: limit_memory(cfg.get("memory_mb", 512) + 64)  # + Python itself
except Exception as e:  # noqa: BLE001 - the outside watcher still applies
    sys.stderr.write(f"memory cap not set: {e}\n")
sys.path[:] = [ROOT] + [p for p in sys.path if _inside(p, STDLIB)]

sys.dont_write_bytecode = True
sys.addaudithook(hook)
for test_file in cfg["tests"]:
    path = os.path.join(ROOT, *test_file.split("/"))
    try:
        mod = load(path)
    except BaseException as e:  # noqa: BLE001 - a test file that doesn't even load is reported, not fatal
        out["load_errors"].append({"file": test_file, **failure(e)}); continue
    for name in [n for n in vars(mod) if n.startswith("test_") and callable(getattr(mod, n))]:
        r = {"name": name, "file": test_file}
        sys.stderr.write(f"running: {name}\n"); sys.stderr.flush()
        buf, t0 = io.StringIO(), time.time()
        try:
            with contextlib.redirect_stdout(buf):
                getattr(mod, name)()
            r["passed"] = True
        except BaseException as e:  # noqa: BLE001 - every failure, even SystemExit, is a test result
            r.update(passed=False, **failure(e))
            printed = buf.getvalue().strip().splitlines()[-10:]
            if printed: out["console"] += [{"level": "print", "message": l[:300], "file": test_file, "line": None, "test": name} for l in printed]
        r["ms"] = int((time.time() - t0) * 1000)
        out["tests"].append(r); out["passed" if r["passed"] else "failed"] += 1
REAL_STDOUT.write(json.dumps(out)); REAL_STDOUT.flush()
