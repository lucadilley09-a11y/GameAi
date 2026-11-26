"""Training speed-ups (Optimise phase, approved by the user 2026-11-25), installed and chosen automatically.

    install()    Triton for Windows (lets torch.compile work for the from-scratch game AI) and Unsloth (faster,
                 leaner adapter training). Each is installed without changing the PyTorch / Transformers versions that
                 already work (a constraints file pins them); if that isn't possible, or the library doesn't import
                 and run here, it's removed again and training carries on exactly as before.
    choose(base) a short timed run of the adapter trainer with each backend on the same rows; Unsloth is used only if
                 it is at least MIN_GAIN faster and its loss after the same steps is no worse than LOSS_TOL. The result
                 is saved in runtime/training.json and read by lora_train.py and the GRPO trial.
"""
import importlib, json, os, subprocess, sys, tempfile, time

CHOICE = os.path.join("runtime", "training.json")
MIN_GAIN = 0.10
LOSS_TOL = 0.03
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def backend():
    try: return json.load(open(CHOICE)).get("backend", "transformers")
    except (OSError, ValueError): return "transformers"


def _pip(*args):
    p = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True, creationflags=NO_WINDOW)
    return p.returncode == 0, (p.stderr or p.stdout)[-600:]


def _pins():
    """a constraints file holding today's versions of the libraries that must not change"""
    from importlib.metadata import version, PackageNotFoundError
    lines = []
    for name in ("torch", "torchvision", "torchaudio", "transformers", "peft", "bitsandbytes", "accelerate"):
        try: lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError: pass
    fd, path = tempfile.mkstemp(suffix=".txt"); os.close(fd)
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def _triton_spec():
    """triton-windows releases follow PyTorch: torch 2.6 -> triton 3.2, 2.7 -> 3.3, 2.8 -> 3.4, ... (3.(N-4))"""
    import torch
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    return f"triton-windows<3.{minor - 3}" if major == 2 and minor >= 6 else None


def _works(module, test):
    try:
        importlib.invalidate_caches()
        m = importlib.import_module(module)
        return bool(test(m))
    except Exception:
        return False


def _triton_runs(_):
    import torch
    if not torch.cuda.is_available(): return False
    f = torch.compile(lambda x: (x * 2 + 1).relu())
    x = torch.randn(64, device="cuda")
    return torch.allclose(f(x), (x * 2 + 1).relu())


def install(log=print):
    import torch
    out = {"triton": "not needed", "unsloth": "not tried"}
    if not torch.cuda.is_available():
        log("no NVIDIA GPU: the training speed-ups don't apply"); return out
    pins = _pins()
    try:
        if os.name == "nt":
            spec = _triton_spec()
            if not spec: out["triton"] = "this PyTorch is too old for it"
            elif _works("triton", _triton_runs): out["triton"] = "already working"
            else:
                ok, why = _pip("install", "--no-cache-dir", spec, "-c", pins)
                if ok and subprocess.run([sys.executable, "-c", "from gameai.phase0.speedups import _triton_runs; import sys; sys.exit(0 if _triton_runs(None) else 1)"],
                                         capture_output=True, creationflags=NO_WINDOW).returncode == 0:
                    out["triton"] = "installed and working"
                else:
                    _pip("uninstall", "-y", "triton-windows"); out["triton"] = "couldn't be made to work here (removed): " + why[-200:]
        log(f"Triton: {out['triton']}")
        if _works("unsloth", lambda m: True): out["unsloth"] = "already installed"
        else:
            ok, why = _pip("install", "--no-cache-dir", "unsloth", "-c", pins)
            ok = ok and subprocess.run([sys.executable, "-c", "import unsloth"], capture_output=True, creationflags=NO_WINDOW).returncode == 0
            if ok: out["unsloth"] = "installed"
            else:
                _pip("uninstall", "-y", "unsloth", "unsloth_zoo"); out["unsloth"] = "couldn't be installed without changing working libraries (skipped)"
        log(f"Unsloth: {out['unsloth']}")
    finally:
        os.remove(pins)
    return out


def _bench(base, backend_name, steps=30):
    """time `steps` training steps of the code skill (or the general rows) with one backend -> (s per step, final loss)"""
    cmd = [sys.executable, "-u", "-m", "gameai.lora_train", "--skill", "code", "--base", base, "--minutes", "0", "--bench-steps", str(steps),
           "--backend", backend_name]
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW, env=dict(os.environ, GAMEAI_FULL_SPEED="1"))
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("@@BENCH"): d = json.loads(line[7:]); return d["s_per_step"], d["loss"]
    raise RuntimeError((p.stderr or p.stdout)[-400:])


def choose(base, log=print):
    from .. import storage
    storage.ensure_training_copy(log)
    res = {"measured": time.strftime("%Y-%m-%d %H:%M"), "backend": "transformers"}
    try:
        res["transformers"] = dict(zip(("s_per_step", "loss"), _bench(base, "transformers")))
    except Exception as e:
        res["error"] = f"the timing run failed: {str(e)[:200]}"; log(res["error"]); _save(res); return res
    if _works("unsloth", lambda m: True):
        try:
            res["unsloth"] = dict(zip(("s_per_step", "loss"), _bench(base, "unsloth")))
            t, u = res["transformers"], res["unsloth"]
            faster = t["s_per_step"] / u["s_per_step"] - 1
            if faster >= MIN_GAIN and u["loss"] <= t["loss"] * (1 + LOSS_TOL):
                res["backend"] = "unsloth"
            log(f"adapter training: Unsloth {faster:+.0%} speed, loss {u['loss']:.3f} vs {t['loss']:.3f} -> using {res['backend']}")
        except Exception as e:
            res["unsloth_error"] = str(e)[:300]; log("Unsloth didn't run here; keeping the standard trainer")
    _save(res)
    return res


def _save(res):
    os.makedirs(os.path.dirname(CHOICE), exist_ok=True)
    json.dump(res, open(CHOICE, "w"), indent=1)
