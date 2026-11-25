"""Reproducible hardware baselines: same sizes, same seeds, warm-up first, every test run 3 times (median and range).

GPU: matrix maths speed in the number formats training uses, memory speed, and CPU-to-GPU copy speed.
CPU: matrix maths and plain Python speed. Disk: read and write speed (a 512 MB temporary file, deleted after).
"""
import os, time

from .common import median_spread, save, say, stage, sysinfo, progress

REPEATS = 3


def _time(fn, sync=None, n=1):
    fn()  # warm-up
    if sync: sync()
    out = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        for _ in range(n): fn()
        if sync: sync()
        out.append((time.perf_counter() - t) / n)
    return out


def gpu_tests():
    import torch
    if not torch.cuda.is_available(): return {"gpu": "none found"}
    torch.manual_seed(0); sync = torch.cuda.synchronize; r = {}
    for name, dt in (("fp32", torch.float32), ("tf32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
        n = 4096
        a, b = torch.randn(n, n, device="cuda", dtype=dt), torch.randn(n, n, device="cuda", dtype=dt)
        secs = _time(lambda: a @ b, sync, n=10)
        r[f"matmul_{name}_tflops"] = median_spread([2 * n**3 / s / 1e12 for s in secs])
        del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.empty(256 * 2**20 // 4, device="cuda")  # 256 MB
    y = torch.empty_like(x)
    secs = _time(lambda: y.copy_(x), sync, n=10)
    r["gpu_memory_gb_per_s"] = median_spread([2 * x.numel() * 4 / s / 2**30 for s in secs])
    h = torch.empty(x.numel(), pin_memory=True)
    secs = _time(lambda: x.copy_(h, non_blocking=True), sync, n=5)
    r["cpu_to_gpu_gb_per_s"] = median_spread([h.numel() * 4 / s / 2**30 for s in secs])
    del x, y, h; torch.cuda.empty_cache()
    return r


def cpu_tests():
    import numpy as np
    rng = np.random.default_rng(0); r = {}
    a, b = rng.standard_normal((2048, 2048), dtype=np.float32), rng.standard_normal((2048, 2048), dtype=np.float32)
    secs = _time(lambda: a @ b, n=3)
    r["cpu_matmul_gflops"] = median_spread([2 * 2048**3 / s / 1e9 for s in secs])

    def py():
        s = 0
        for i in range(2_000_000): s += i * i % 7
    secs = _time(py)
    r["python_loop_ms"] = median_spread([s * 1000 for s in secs])
    return r


def disk_tests():
    path, size = os.path.join("bench", "phase0", "disk_test.tmp"), 512 * 2**20
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = os.urandom(16 * 2**20)
    w, rd = [], []
    try:
        for _ in range(REPEATS):
            t = time.perf_counter()
            with open(path, "wb") as f:
                for _ in range(size // len(data)): f.write(data)
                f.flush(); os.fsync(f.fileno())
            w.append(size / (time.perf_counter() - t) / 2**20)
            t = time.perf_counter()
            with open(path, "rb") as f:
                while f.read(16 * 2**20): pass
            rd.append(size / (time.perf_counter() - t) / 2**20)  # (may be helped by Windows' file cache)
    finally:
        if os.path.exists(path): os.remove(path)
    return {"disk_write_mb_per_s": median_spread(w), "disk_read_mb_per_s": median_spread(rd)}


def run():
    stage("Hardware baseline")
    info = sysinfo(); say("System:", info)
    res = {"system": info}
    for i, (name, fn) in enumerate((("GPU", gpu_tests), ("CPU", cpu_tests), ("disk", disk_tests))):
        progress(i, 3); say(f"Testing the {name}...")
        try: res[name.lower()] = fn()
        except Exception as e: res[name.lower()] = {"error": str(e)}
        say(" ", res[name.lower()])
    progress(3, 3)
    path = save("hardware", res)
    say(f"Saved: {path}")
    return res
