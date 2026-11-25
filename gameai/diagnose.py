"""A visible check of everything training needs, step by step with times, so a stall shows exactly where it is.
Run by training-check.bat (double-click it)."""
import os, subprocess, sys, time

t0 = time.time()
def step(msg): print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

step(f"Python {sys.version.split()[0]} at {sys.executable}")
step(f"folder: {os.getcwd()}")
try:
    import shutil
    step(f"disk free: {shutil.disk_usage('.').free / 2**30:.1f} GB")
except Exception as e:
    step(f"disk check failed: {e}")
step("programs using the GPU (nvidia-smi):")
try:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    print("    " + (r.stdout.strip().replace("\n", "\n    ") or "(none)"), flush=True)
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    print("    GPU: " + r.stdout.strip(), flush=True)
except Exception as e:
    step(f"nvidia-smi not available: {e}")
try:
    import psutil
    mine = [p for p in psutil.process_iter(["pid", "cmdline"]) if any("gameai." in (c or "") for c in (p.info["cmdline"] or []))]
    step(f"Game AI programs running: {len(mine)}")
    for p in mine: print(f"    pid {p.pid}: {' '.join(p.info['cmdline'][-4:])}", flush=True)
except Exception as e:
    step(f"process list failed: {e}")
step("importing PyTorch...")
import torch
step(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    step("starting the GPU...")
    x = torch.ones(1, device="cuda"); torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    step(f"GPU ready: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free")
    a = torch.randn(2048, 2048, device="cuda"); t = time.time()
    for _ in range(20): a @ a
    torch.cuda.synchronize(); step(f"GPU maths test: {20 * 2 * 2048**3 / (time.time() - t) / 1e12:.1f} TFLOPS")
step("importing the game AI...")
from gameai import train  # noqa: F401
step("now a short training (30 steps), the same way the upgrade does it:")
args = [sys.executable, "-u", "-m", "gameai.train", "--steps", "30", "--eval-every", "10", "--name", "check"]
if os.path.exists("models/current.pt"): args += ["--init", "models/current.pt", "--lr", "1.5e-3", "--muon-lr", "0.01"]
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    if not line.startswith("@@"): print(f"[{time.time() - t0:6.1f}s]   " + line.rstrip(), flush=True)
step(f"training check finished with code {p.wait()}" + (" (OK)" if p.returncode == 0 else " (FAILED: send this to Claude)"))
