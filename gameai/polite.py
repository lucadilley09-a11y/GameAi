"""Sharing the PC (Optimise phase, user's request 2026-11-25): learning and training slow themselves down while you're
using the PC, so your own programs stay smooth, and go full speed again when you step away. Fully automatic.

    from gameai import polite
    polite.lower_priority()          # once, at the start of a training job
    gate = polite.Gate()
    for step in ...:
        ...one training step...
        gate.after_step()            # pauses a little when you're busy, not at all when you're away

How busy you are is worked out every few seconds:
    away     no keyboard or mouse for AWAY_S seconds           -> full speed
    using    you're using the PC                              -> the GPU is shared: training runs USING_DUTY of the time
    heavy    you're using the PC and other programs are working hard (a game, a render, compiling: more than
             HEAVY_CPU % of the processor that isn't ours, or a full-screen window)  -> training runs HEAVY_DUTY of the time
CPU priority is also lowered for training processes, so Windows always serves your programs first.
Pausing between steps changes nothing about what is learned, only how long it takes.
"""
import os, sys, time

AWAY_S = 120
USING_DUTY = 0.5
HEAVY_DUTY = 0.15
HEAVY_CPU = 45
CHECK_S = 5
OFF = os.environ.get("GAMEAI_FULL_SPEED") == "1"   # the autopilot's away-time jobs and tests can turn it off


def lower_priority(pid=None):
    """Below-normal CPU priority and low disk priority for this process (or process pid), so your own programs come
    first for the processor AND the disk (big model downloads and loads otherwise keep the disk at 100%)."""
    try:
        import psutil
        p = psutil.Process(pid)
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        try: p.ionice(psutil.IOPRIO_LOW if os.name == "nt" else psutil.IOPRIO_CLASS_BE, *([] if os.name == "nt" else [7]))
        except Exception: pass
        return True
    except Exception:
        return False


def idle_seconds():
    """seconds since the last keyboard or mouse input (Windows); elsewhere, 'always away'"""
    if os.name != "nt": return float("inf")
    try:
        import ctypes
        class LASTINPUTINFO(ctypes.Structure): _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
        li = LASTINPUTINFO(); li.cbSize = ctypes.sizeof(li)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)): return float("inf")
        return ((ctypes.windll.kernel32.GetTickCount() - li.dwTime) & 0xFFFFFFFF) / 1000.0
    except Exception:
        return float("inf")


def fullscreen_app():
    """is a full-screen window in front (a game or video)?"""
    if os.name != "nt": return False
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        if not hwnd or hwnd in (u.GetDesktopWindow(), u.GetShellWindow()): return False
        r = wintypes.RECT(); u.GetWindowRect(hwnd, ctypes.byref(r))
        return r.left <= 0 and r.top <= 0 and r.right >= u.GetSystemMetrics(0) and r.bottom >= u.GetSystemMetrics(1)
    except Exception:
        return False


def others_cpu():
    """% of the processor used by programs other than this one and its family"""
    try:
        import psutil
        mine = {psutil.Process().pid}
        try: mine |= {c.pid for c in psutil.Process().children(recursive=True)} | {psutil.Process().ppid()}
        except Exception: pass
        total = psutil.cpu_percent(interval=None)
        ours = sum(p.cpu_percent(interval=None) for p in map(psutil.Process, mine) if p.is_running()) / max(1, psutil.cpu_count())
        return max(0.0, total - ours)
    except Exception:
        return 0.0


def state():
    if OFF: return "away"
    if idle_seconds() >= AWAY_S: return "away"
    return "heavy" if fullscreen_app() or others_cpu() > HEAVY_CPU else "using"


class Gate:
    """call after_step() after each unit of work; it pauses just enough to keep the chosen share of time"""
    def __init__(self, log=None):
        self.last, self.state, self.checked, self.log = time.perf_counter(), "away", 0.0, log
        try: __import__("psutil").cpu_percent(interval=None)
        except Exception: pass

    def after_step(self):
        now = time.perf_counter()
        work = now - self.last
        if now - self.checked > CHECK_S:
            new = state()
            if new != self.state and self.log:
                self.log({"away": "You're away: learning at full speed.", "using": "You're using the PC: learning slows down to keep it smooth.",
                          "heavy": "A game or heavy program is running: learning is mostly paused."}[new])
            self.state, self.checked = new, now
        duty = {"away": 1.0, "using": USING_DUTY, "heavy": HEAVY_DUTY}[self.state]
        if duty < 1.0:
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.synchronize()  # the GPU work really finished before resting
            except Exception: pass
            time.sleep(min(30.0, work * (1 / duty - 1)))
        self.last = time.perf_counter()
