"""Shared pieces for Phase 0: where results go, a memory sampler, system info, and the free-space guard."""
import json, os, platform, shutil, statistics, subprocess, threading, time

OUT = os.path.join("bench", "phase0")
MIN_FREE_GB = 8  # downloads and training pause below this, instead of filling the disk


def say(*a):
    print(*a, flush=True)


def stage(title):
    print(f"\n=== {title} ===", flush=True)


def progress(step, total):
    """read by the app's progress bar"""
    if os.environ.get("GAMEAI_PROGRESS"):
        print("@@P" + json.dumps({"step": step, "total": total, "loss": None}), flush=True)


def save(name, data):
    os.makedirs(OUT, exist_ok=True)
    data = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "app_version": app_version(), **data}
    path = os.path.join(OUT, f"{name}.json")
    json.dump(data, open(path, "w"), indent=1)
    with open(os.path.join(OUT, "history.jsonl"), "a") as f: f.write(json.dumps({"name": name, **data}) + "\n")
    return path


def app_version():
    try:
        from ..update import local_version
        return (local_version().get("sha") or "dev")[:7]
    except Exception:
        return "dev"


def median_spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    return {"median": round(statistics.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3), "runs": len(xs)}


def free_gb(path="."):
    return shutil.disk_usage(os.path.abspath(path)).free / 2**30


def need_space(gb, what):
    """Refuse to start something that would leave less than MIN_FREE_GB free."""
    left = free_gb() - gb
    if left < MIN_FREE_GB:
        raise RuntimeError(f"Not enough disk space for {what}: it needs about {gb:.1f} GB and would leave {left:.1f} GB "
                           f"(at least {MIN_FREE_GB} GB must stay free). Free up some space and try again.")


def nvidia_smi(query="memory.used,memory.total,utilization.gpu,driver_version,name"):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()
        return [x.strip() for x in out[0].split(",")] if out else None
    except Exception:
        return None


def sysinfo():
    import psutil
    info = {"os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
            "cores_physical": psutil.cpu_count(logical=False), "cores_logical": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "disk_free_gb": round(free_gb(), 1)}
    if os.name == "nt":
        try:
            info["cpu"] = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True,
                                         timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip() or info["cpu"]
        except Exception:
            pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 2), cuda=torch.version.cuda)
    except Exception:
        pass
    smi = nvidia_smi()
    if smi: info["gpu_driver"] = smi[3]
    return info


class Sampler:
    """Samples memory every half second while a test runs; keeps the peaks.
    ram: whole PC; procs: this program and everything it started; vram: whole GPU (nvidia-smi); swap: pagefile."""

    def __init__(self, every=0.5):
        import psutil
        self.ps, self.every, self.stop_ = psutil, every, threading.Event()
        self.peak = {"ram_used_gb": 0, "procs_gb": 0, "vram_used_gb": 0, "swap_used_gb": 0, "cpu_percent": 0}
        self.base = self._sample()

    def _sample(self):
        ps = self.ps
        me = ps.Process()
        try: procs = [me] + me.children(recursive=True)
        except Exception: procs = [me]
        rss = 0
        for p in procs:
            try: rss += p.memory_info().rss
            except Exception: pass
        smi = nvidia_smi("memory.used")
        return {"ram_used_gb": (ps.virtual_memory().total - ps.virtual_memory().available) / 2**30, "procs_gb": rss / 2**30,
                "vram_used_gb": float(smi[0]) / 1024 if smi else 0, "swap_used_gb": ps.swap_memory().used / 2**30,
                "cpu_percent": ps.cpu_percent()}

    def _loop(self):
        while not self.stop_.is_set():
            try:
                s = self._sample()
                for k, v in s.items(): self.peak[k] = max(self.peak[k], v)
            except Exception:
                pass  # a worker process ended while being measured: just measure again
            self.stop_.wait(self.every)

    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.stop_.set(); self.t.join(5)

    def result(self):
        r = {k: round(v, 2) for k, v in self.peak.items()}
        r["swap_growth_gb"] = round(self.peak["swap_used_gb"] - self.base["swap_used_gb"], 2)
        r["vram_before_gb"] = round(self.base["vram_used_gb"], 2)
        return r
