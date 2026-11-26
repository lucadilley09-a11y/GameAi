"""Keeping storage as low as possible without losing speed or capability (Optimise phase, user's request 2026-11-25).
Runs by itself (the app cleans once a day and after every autopilot step); /storage in the app shows the report.

What it removes, all safe:
    download caches     pip's cache, and Hugging Face downloads the app no longer uses
    old checkpoints     training run files once their result has been tested (the result itself is kept)
    old versions        your AI's versions beyond the last KEEP_VERSIONS (rollback still reaches them)
    big logs            logs over 20 MB are cut down to their last 5 MB
    temporary files     leftovers of interrupted downloads, conversions and staging copies
    the training copy   the base model's full-size training copy (several GB) when nothing has needed it for
                        OFFLOAD_DAYS days and no training is waiting; it's downloaded again, at the same pinned version,
                        the next time training needs it. Day-to-day use only needs the compressed copy.
What it never touches: the model in use, its compressed copy, adapters in use, releases, your games, your data.
"""
import glob, json, os, shutil, subprocess, sys, time

KEEP_VERSIONS = 10
OFFLOAD_DAYS = 3
BIG_LOG = 20 * 2**20
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def size(path):
    if os.path.isfile(path): return os.path.getsize(path)
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(d, f))
            except OSError: pass
    return total


def _hf_cache():
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")


def _pip_cache():
    try:
        p = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"], capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def report():
    """-> [(what, bytes, note)], biggest first"""
    rows = []
    def add(what, paths, note=""):
        b = sum(size(p) for p in paths if p and os.path.exists(p))
        if b: rows.append((what, b, note))
    from . import registry
    ent = registry.load().get("entries", {})
    core_hf = (ent.get("core_hf") or {}).get("path")
    add("base model, compressed (in use)", [(ent.get("core") or {}).get("path")])
    add("base model, training copy", [core_hf], "removed when unused for a few days; downloaded again for training")
    add("coding teacher", ["models/teachers"], "only one at a time")
    add("skill adapters", ["models/adapters"])
    add("releases", ["models/releases"])
    add("your game AI and its versions", ["models/current.pt", "models/versions"])
    add("training checkpoints", ["checkpoints"])
    add("data", ["data"])
    add("tools (llama.cpp, checkers)", ["runtime"])
    add("logs", ["logs"])
    add("Hugging Face download cache", [_hf_cache()], "cleaned of models the app doesn't use")
    add("pip download cache", [_pip_cache()], "cleaned automatically")
    return sorted(rows, key=lambda r: -r[1])


def text_report():
    rows = report()
    total = sum(b for _, b, _ in rows)
    out = [f"Game AI uses {total / 2**30:.1f} GB:"]
    out += [f"  {b / 2**30:6.2f} GB  {w}" + (f"  ({n})" if n else "") for w, b, n in rows]
    return "\n".join(out)


JOBS = ("gameai.train", "gameai.lora_train", "gameai.learn", "gameai.all", "gameai.coding.grpo", "gameai.teacher", "gameai.autopilot",
        "gameai.release", "gameai.phase0")


def _busy():
    """a training job or the autopilot is working: the training copy and checkpoints are left alone"""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cmd = " ".join(p.info.get("cmdline") or [])
            if any(f"-m {j}" in cmd for j in JOBS) and p.pid != os.getpid(): return True
    except Exception:
        return True  # can't tell: be careful
    return False


def _training_pending():
    try:
        from . import autopilot
        st = autopilot.load()
        done = set(st.get("done", []))
        return any(sid not in done for sid, *_ in autopilot.STEPS)
    except Exception:
        return True


def clean(log=print, offload=True):
    """-> bytes freed"""
    freed = 0
    def rm(path, why):
        nonlocal freed
        try:
            b = size(path)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            freed += b
            if b > 50 * 2**20: log(f"  freed {b / 2**30:.2f} GB: {why}")
        except OSError:
            pass
    # 1. pip's download cache
    try:
        before = size(_pip_cache() or "")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True, timeout=120, creationflags=NO_WINDOW)
        freed += max(0, before - size(_pip_cache() or "")) if _pip_cache() else 0
    except Exception:
        pass
    # 2. Hugging Face downloads the app doesn't use (the learning teacher's model is kept)
    try:
        from .teacher import CANDIDATES, LEARN_DIR
        # the learning teacher's full download is kept only while its compressed llama.cpp copy doesn't exist
        keep = set() if glob.glob(os.path.join(LEARN_DIR, "**", "*.gguf"), recursive=True) else {"models--" + r.replace("/", "--") for r in CANDIDATES}
    except Exception:
        keep = set()
    hub = _hf_cache()
    for d in glob.glob(os.path.join(hub, "models--*")):
        if os.path.basename(d) not in keep: rm(d, f"unused download {os.path.basename(d)[8:].replace('--', '/')}")
    for d in glob.glob(os.path.join(hub, "*", "blobs", "*.incomplete")): rm(d, "unfinished download")
    if not _busy():
        # 3. training checkpoints whose results have been tested (latest.pt is what was last tested)
        for f in glob.glob(os.path.join("checkpoints", "*.pt")):
            if os.path.basename(f) != "latest.pt" and time.time() - os.path.getmtime(f) > 3600: rm(f, "an old training checkpoint")
        # 4. temporary leftovers
        for pat in ("models/**/*.tmp", "models/**/*.incomplete", "models/**/merged-hf", "**/.gameai/staging", "models/base/*/*.calib.txt",
                    "models/base/*/*.plain.gguf", "models/base/*/*.imatrix.gguf", "models/releases/**/*.f16.gguf"):
            for p in glob.glob(pat, recursive=True): rm(p, "a leftover from an interrupted job")
    # 5. old versions of your game AI beyond the last KEEP_VERSIONS
    have = sorted(glob.glob(os.path.join("models", "versions", "v*.pt")))
    for f in have[:-KEEP_VERSIONS]:
        rm(f, "an old version of your game AI"); rm(f[:-3] + ".json", "")
    # 6. big logs: keep their end
    for f in glob.glob(os.path.join("logs", "*.log")):
        try:
            if os.path.getsize(f) > BIG_LOG:
                with open(f, "rb") as fh: fh.seek(-5 * 2**20, 2); tail = fh.read()
                freed += os.path.getsize(f) - len(tail)
                open(f, "wb").write(b"(older lines removed to save space)\n" + tail)
        except OSError:
            pass
    # 7. the base model's training copy, when nothing has needed it for a while
    if offload and not _busy() and not _training_pending():
        freed += offload_training_copy(log)
    return freed


def offload_training_copy(log=print):
    from . import registry
    e = registry.load()["entries"].get("core_hf") or {}
    p = e.get("path")
    if not p or not os.path.isdir(p) or not e.get("repo") or not e.get("revision"): return 0
    last = max((os.path.getatime(f) for f in glob.glob(os.path.join(p, "*.safetensors"))), default=time.time())
    marker = os.path.join(p, "..", "training_copy_used")
    if os.path.exists(marker): last = max(last, os.path.getmtime(marker))
    if time.time() - last < OFFLOAD_DAYS * 86400: return 0
    b = size(p)
    shutil.rmtree(p, ignore_errors=True)
    log(f"  freed {b / 2**30:.2f} GB: the base model's training copy (unused for {OFFLOAD_DAYS}+ days; it's downloaded again when training needs it)")
    return b


def ensure_training_copy(log=print):
    """the base model's training copy, downloaded again (same pinned version) if it was removed -> its folder"""
    from . import registry
    e = registry.load()["entries"].get("core_hf") or {}
    p = e.get("path")
    if not p: raise RuntimeError("no base model is installed yet")
    if not glob.glob(os.path.join(p, "*.safetensors")):
        from .phase0 import candidates as C
        from .phase0.common import need_space
        rec = C.record(C.by_key(e["key"])) if e.get("key") else None
        if not rec: raise RuntimeError("can't tell which model to download again")
        rec["revision"] = e.get("revision") or rec["revision"]  # exactly the version the AI was trained on
        need_space(rec["download_gb"] + 1, "downloading the base model's training copy again")
        log(f"Downloading the base model's training copy again ({rec['download_gb']} GB, the same version as before)...")
        C.download(rec, p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(os.path.join(p, "..", "training_copy_used"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
    return p


if __name__ == "__main__":
    print(text_report())
    if "--clean" in sys.argv:
        print(f"freed {clean() / 2**30:.2f} GB")
