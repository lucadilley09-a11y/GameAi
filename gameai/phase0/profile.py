"""Where the time goes: each stage of training and of making a game, timed separately (median of several runs).

Training: loading data, the test-question check, turning text into numbers, one learning step split into
forward pass / backward pass / optimizer, and one check. Making a game: the AI's answer and one playtest.
"""
import json, random, statistics, time

from .common import save, say, stage, progress


class Timer:
    def __init__(self): self.t = {}

    def run(self, name, fn, repeat=1, sync=None):
        out = []
        for _ in range(repeat):
            if sync: sync()
            t = time.perf_counter(); r = fn()
            if sync: sync()
            out.append((time.perf_counter() - t) * 1000)
        self.t[name] = round(statistics.median(out), 2)
        return r


def train_profile(size, steps=10):
    import torch
    from .. import train as T
    from ..tokenizer import Tokenizer
    from ..model import GameAI, make_config
    from ..muon import Muon, split_params
    from ..recipe import OUT_FIELDS
    from ..guard import filter_training
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else None
    tm, tok = Timer(), Tokenizer()
    rows = tm.run("load 20k examples (ms)", lambda: T.load_jsonl("data/train.jsonl")[:20000])
    if not rows: return {"note": "no training data yet (type /train once first)"}
    rows = tm.run("check them against test questions (ms)", lambda: filter_training(rows, log=None))
    pairs = tm.run("turn them into numbers (ms)", lambda: T.encode(rows, tok))
    data = tm.run("move them to the GPU (ms)", lambda: T.GpuData(pairs, tok, dev), sync=sync)
    model = GameAI(make_config(size, tok.vocab_size, tok.max_len)).to(dev)
    mu, ad = split_params(model)
    opts = [Muon(mu, lr=0.01, weight_decay=0.01), torch.optim.AdamW(ad, lr=1e-3)]
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    n_out = len(OUT_FIELDS) + 1
    micro = min(T.SIZE_DEFAULTS[size][1], T.MICRO.get(size, 64))
    batch = T.SIZE_DEFAULTS[size][1]
    idx = data.sample(batch)
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()

    def fwd():
        tot = 0.0
        for c in idx.split(micro):
            xb, yb, P = data.batch(c)
            with amp: tot = tot + T.recipe_loss(model, xb, yb, P, n_out) * (len(c) / len(idx))
        return tot
    fwd()  # warm-up
    f, b, o = [], [], []
    for i in range(steps):
        progress(i, steps)
        for opt in opts: opt.zero_grad(set_to_none=True)
        if sync: sync()
        t = time.perf_counter(); loss = fwd()
        if sync: sync()
        t2 = time.perf_counter(); loss.backward()
        if sync: sync()
        t3 = time.perf_counter()
        for opt in opts: opt.step()
        if sync: sync()
        t4 = time.perf_counter()
        f.append((t2 - t) * 1000); b.append((t3 - t2) * 1000); o.append((t4 - t3) * 1000)
    tm.t["learning step: forward pass (ms)"] = round(statistics.median(f), 2)
    tm.t["learning step: backward pass (ms)"] = round(statistics.median(b), 2)
    tm.t["learning step: optimizer (ms)"] = round(statistics.median(o), 2)
    tm.run("one check during training (ms)", lambda: T.quick_accuracy(model, data, tok, amp, n_out, piece=micro * 4), sync=sync)
    res = {"size": size, "params_m": round(model.num_params() / 1e6, 2), "batch": batch, "micro": micro, "device": dev, **tm.t}
    step_ms = tm.t["learning step: forward pass (ms)"] + tm.t["learning step: backward pass (ms)"] + tm.t["learning step: optimizer (ms)"]
    res["estimated minutes for a full training"] = round(step_ms * T.SIZE_DEFAULTS[size][0] / 60000, 1)
    if dev == "cuda": res["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return res


def game_profile():
    import torch
    tm = Timer()
    try:
        from ..generate import load, decode
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok, _ = load("models/current.pt", dev, upgrade=True)
        decode(model, tok, ["warm up"], dev)
        tm.run("AI reads one request (ms)", lambda: decode(model, tok, ["a fast ninja in a spooky dungeon with a boss"], dev), repeat=10)
    except Exception as e:
        tm.t["AI reads one request (ms)"] = f"not measured: {e}"
    from ..headless import Engine, random_recipe
    e, rng = Engine(), random.Random(7)
    e.judge(random_recipe(rng))
    tm.run("one playtest, all bots (ms)", lambda: e.judge(random_recipe(rng)), repeat=10)
    e.close()
    return tm.t


def run(sizes=("small", "base")):
    stage("Profiler: where the time goes")
    res = {"game": game_profile()}
    say("Making a game:", res["game"])
    for s in sizes:
        say(f"Training ({s} size)...")
        try: res[f"train_{s}"] = train_profile(s)
        except Exception as ex: res[f"train_{s}"] = {"error": str(ex)}
        say(" ", res[f"train_{s}"])
    path = save("profile", res); say(f"Saved: {path}")
    return res
