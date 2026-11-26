"""Train the designer model from scratch.

    python -m gameai.train --size small            # recommended start on an RTX 3060 Ti
    python -m gameai.train --size tiny --steps 500 # quick smoke test (works on CPU)

Doc sections used: 1 (architecture), 2 (Muon + AdamW, warmup-stable-decay schedule),
10 (BF16, SDPA, optional torch.compile), 13/16 (tracking, frozen tests, best checkpoint).
"""
import argparse, json, os, random, shutil, time
import torch
import torch.nn.functional as F
from .recipe import OUT_FIELDS, GRAMMAR_VERSION
from .tokenizer import Tokenizer
from .model import GameAI, make_config
from .muon import Muon, split_params
from .generate import decode, left_pad
from .track import Run


def load_jsonl(path):
    """Examples saved before a part of the game language existed count as not mentioning it."""
    from .recipe import fill_defaults
    if not os.path.exists(path): return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        if "recipe" in r: fill_defaults(r["recipe"])
    return rows


def encode(rows, tok):
    return [(tok.encode_prompt(r["prompt"]), tok.encode_recipe(r["recipe"])) for r in rows]


def make_batch(pairs, tok, device):
    x, P = left_pad([p for p, _ in pairs], tok.pad)
    rec = torch.tensor([r for _, r in pairs])
    seq = torch.cat([x, rec], 1).to(device)
    return seq[:, :-1], seq[:, 1:], P  # predictions for the recipe live at positions P-1 .. P+len-2


class GpuData:
    """All examples packed into tensors on the GPU once, so a batch is just an index lookup
    (no Python work per step). Same left-padding as left_pad(), so results are identical."""
    def __init__(self, pairs, tok, device):
        L = max(len(p) for p, _ in pairs)
        self.lens = torch.tensor([len(p) for p, _ in pairs], device=device)
        small = torch.int16 if tok.vocab_size < 32000 else torch.int32  # 2 bytes per symbol instead of 8: 4x less GPU memory
        self.X = torch.tensor([[tok.pad] * (L - len(p)) + p for p, _ in pairs], dtype=small, device=device)
        self.R = torch.tensor([r for _, r in pairs], dtype=small, device=device)
        self.L, self.n = L, len(pairs)
        self.order = torch.argsort(self.lens)  # examples sorted by length, for batches with little padding

    def sample(self, B):
        """a random batch of examples of similar length: much less padding (wasted work) than fully random,
        and still random: each batch comes from a random window of 16x its size"""
        w = min(self.n, B * 16); start = int(torch.randint(0, self.n - w + 1, (1,)))
        return self.order[start + torch.randperm(w, device=self.order.device)[:B]]

    def batch(self, idx):
        P = int(self.lens[idx].max())
        seq = torch.cat([self.X[idx, self.L - P:], self.R[idx]], 1).long()
        return seq[:, :-1], seq[:, 1:], P


def cap_gpu_memory(device, log=print):
    """On Windows, when a program needs more than the graphics card has, the driver doesn't fail: it quietly
    borrows normal RAM ("shared GPU memory") and everything gets 10-50x slower. Capping PyTorch at what the card
    really has free makes it fail fast instead, so training switches to smaller pieces and stays at full speed."""
    if device != "cuda": return
    try:
        free, total = torch.cuda.mem_get_info()
        mine = torch.cuda.memory_reserved()  # the model and data this program already put on the GPU count too
        frac = max(0.02, min(0.95, (free + mine - 400 * 2**20) / total))  # leave a little for Windows and the screen
        torch.cuda.set_per_process_memory_fraction(frac)
        log(f"GPU memory: using up to {frac * total / 2**30:.1f} of {total / 2**30:.1f} GB (the rest is in use by Windows and other programs)")
        if free + mine < 3 * 2**30:
            log(f"Warning: only {(free + mine) / 2**30:.1f} GB of the GPU is free. Another program is using the rest (often a leftover "
                "teacher AI or an old training run). Close it, or restart the PC, for full speed.")
    except Exception as e:
        log(f"could not cap GPU memory: {e}")


def in_pieces(fn, n, size):
    """Run fn(start, end) over 0..n in pieces of `size`, halving the pieces if the GPU runs out of memory."""
    out, i = [], 0
    while i < n:
        try:
            out.append(fn(i, min(i + size, n))); i += size
        except torch.OutOfMemoryError:
            if size <= 1: raise
            size //= 2; torch.cuda.empty_cache()
    return out


@torch.no_grad()
def quick_accuracy(model, data, tok, amp, n_out, limit=2000, piece=250):
    """Fast check during training: one pass, each setting must be the top allowed choice.
    (The full slow decode runs once at the end.)"""
    model.eval()
    ids = [torch.tensor(f, device=data.X.device) for f in tok.field_ids]
    n = min(limit, data.n)

    def part(i, j):
        xb, yb, P = data.batch(torch.arange(i, j, device=data.X.device))
        with amp:
            lg = model(xb)[:, P - 1:P - 1 + len(ids)].float()
        ok = torch.stack([ids[s][lg[:, s, ids[s]].argmax(-1)] == yb[:, P - 1 + s] for s in range(len(ids))], 1)
        return ok.sum().item(), ok.all(1).sum().item()
    res = in_pieces(part, n, piece)
    fields, exact = sum(r[0] for r in res), sum(r[1] for r in res)
    return fields / (n * len(ids)), exact / n


def recipe_loss(model, xb, yb, P, n_out):
    logits = model(xb)[:, P - 1:P - 1 + n_out]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb[:, P - 1:P - 1 + n_out].reshape(-1))


def accuracy(model, tok, rows, device):
    preds = decode(model, tok, [r["prompt"] for r in rows], device, gate=0.0)
    fields = exact = 0
    for p, r in zip(preds, rows):
        hits = [p[f][0] == r["recipe"][f] for f in OUT_FIELDS]
        fields += sum(hits); exact += all(hits)
    return fields / (len(rows) * len(OUT_FIELDS)), exact / len(rows)


def wsd(step, total, warm=0.05, decay=0.2):
    """Warmup-stable-decay: lets training be extended later without restarting (doc section 2)."""
    w, d = int(total * warm), int(total * decay)
    if step < w: return (step + 1) / w
    if step < total - d: return 1.0
    return max(0.05, (total - step) / d)


# Bigger models learn best with more steps, bigger batches and gentler learning rates.
MICRO = {"tiny": 128, "small": 128, "base": 32, "large": 16}
SIZE_DEFAULTS = {  # size: (steps, batch, AdamW lr, Muon lr)
    "tiny": (3000, 128, 3e-3, 0.02),
    "small": (4000, 128, 3e-3, 0.02),
    "base": (6000, 256, 2e-3, 0.015),
    "large": (8000, 256, 1.5e-3, 0.01),
}


def maybe_compile(model, mode="auto", device="cuda"):
    """-> (model to call, what happened). Compiling fuses the model's small steps into fewer GPU kernels (research: up
    to ~2x faster steps for small transformers). It needs Triton, which on Windows is the triton-windows package; if
    anything is missing it's skipped, and the results are the same either way."""
    import torch
    if mode == "off": return model, "off"
    if device != "cuda": return model, "skipped (no GPU)"
    try:
        import triton  # noqa: F401  (triton-windows on Windows)
    except ImportError:
        return model, "skipped (Triton isn't installed; the autopilot's speed-up step installs it)"
    try:
        return torch.compile(model), "on (the first step takes about a minute longer while it compiles)"
    except Exception as e:
        return model, f"skipped ({str(e).splitlines()[0][:120]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--steps", type=int, default=None, help="default depends on --size (see SIZE_DEFAULTS)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="AdamW learning rate")
    ap.add_argument("--muon-lr", type=float, default=None)
    ap.add_argument("--no-muon", action="store_true")
    ap.add_argument("--compile", choices=["auto", "on", "off"], default="auto",
                    help="torch.compile: auto = use it when it works here (on Windows it needs triton-windows), else carry on without")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue training from this checkpoint (warm start) instead of from scratch")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = Tokenizer()
    import shutil
    free = shutil.disk_usage(".").free / 2**30
    if free < 2:  # lean storage: stop cleanly instead of filling the disk halfway through
        raise SystemExit(f"Only {free:.1f} GB of disk space is free. Training needs at least 2 GB: free up some space and try again.")
    t_load = time.time()
    print(f"Loading the practice data on {device}...", flush=True)
    rows = load_jsonl(f"{a.data}/train.jsonl")
    print(f"  {len(rows)} practice examples loaded ({time.time() - t_load:.0f}s)", flush=True)
    if os.path.exists("data/teacher.jsonl"):  # natural examples written by the teacher AI, counted twice
        t_rows = load_jsonl("data/teacher.jsonl")
        rows += t_rows * 2
        print(f"using {len(t_rows)} teacher examples", flush=True)
    if os.path.exists("data/extra.jsonl"):  # the self-improving loop's lessons: its own mistakes and your corrections
        e_rows = load_jsonl("data/extra.jsonl")
        rows += e_rows
        print(f"using {len(e_rows)} extra lessons (its mistakes and your corrections)", flush=True)
    from .guard import filter_training
    t1 = time.time()
    rows = filter_training(rows)  # never learn a test question, or a reworded copy of one (validation, tests, sealed sets)
    print(f"  checked them against the test questions ({time.time() - t1:.0f}s)", flush=True)
    val_rows = load_jsonl(f"{a.data}/val.jsonl")
    val = encode(val_rows, tok)
    t1 = time.time()
    train_d, val_d = GpuData(encode(rows, tok), tok, device), GpuData(val, tok, device)
    print(f"  turned them into numbers for the GPU ({time.time() - t1:.0f}s)", flush=True)
    n_out = len(OUT_FIELDS) + 1

    if a.init:
        from .model import Config
        t1 = time.time()
        print(f"Loading your AI from {a.init}...", flush=True)
        ck0 = torch.load(a.init, map_location=device, weights_only=False)
        if ck0.get("tokenizer") != tok.itos:  # older game language: carry it over instead of starting from scratch
            from .generate import upgrade_model
            model = upgrade_model(ck0, tok, device)
        else:
            model = GameAI(Config(**ck0["config"])).to(device); model.load_state_dict(ck0["model"])
        a.size = ck0.get("size", a.size)
        print(f"  ready ({time.time() - t1:.0f}s, size {a.size})", flush=True)
    else:
        model = GameAI(make_config(a.size, tok.vocab_size, tok.max_len)).to(device)
    n_params = model.num_params()
    d = SIZE_DEFAULTS.get(a.size, SIZE_DEFAULTS["small"])
    a.steps = a.steps or d[0]; a.batch = a.batch or d[1]; a.lr = a.lr or d[2]; a.muon_lr = a.muon_lr or d[3]
    micro = min(a.batch, MICRO.get(a.size, 64))  # pieces per step that fit in 8 GB (it halves them by itself if not)
    cap_gpu_memory(device)
    if a.no_muon:
        opts = [torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)]
    else:
        mu, ad = split_params(model)
        opts = [Muon(mu, lr=a.muon_lr, weight_decay=0.01), torch.optim.AdamW(ad, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)]
    for o in opts:
        for g in o.param_groups: g["base_lr"] = g["lr"]
    fwd, how = maybe_compile(model, a.compile, device)
    print(f"torch.compile: {how}", flush=True)
    eager = model  # used instead if the compiled version fails on its first step
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    cfg = vars(a) | {"params": n_params, "device": device, "grammar_version": GRAMMAR_VERSION, "model": model.config_dict()}
    run = Run(a.name or a.size, cfg)
    print(f"run {run.id}: {n_params/1e6:.2f}M parameters on {device}")
    os.makedirs("checkpoints", exist_ok=True)
    ck_path = f"checkpoints/{run.id}.pt"

    def save():
        torch.save({"model": model.state_dict(), "config": model.config_dict(), "tokenizer": tok.itos,
                    "grammar_version": GRAMMAR_VERSION, "run": run.id, "size": a.size}, ck_path)

    from . import polite
    polite.lower_priority(); gate = polite.Gate(log=lambda m: print(m, flush=True))
    best, bad, t0, ema = float("inf"), 0, time.time(), None
    for step in range(a.steps):
        for o in opts:
            for g in o.param_groups: g["lr"] = g["base_lr"] * wsd(step, a.steps)
        model.train()
        idx = train_d.sample(a.batch)
        while True:
            try:
                # the batch is done in smaller pieces that add up (same learning, much less GPU memory)
                for o in opts: o.zero_grad(set_to_none=True)
                loss = 0.0
                for c in idx.split(micro):
                    xb, yb, P = train_d.batch(c)
                    try:
                        with amp:
                            part = recipe_loss(fwd, xb, yb, P, n_out) * (len(c) / len(idx))
                    except torch.OutOfMemoryError:
                        raise
                    except Exception as e:  # compiling failed on this PC: carry on without it (same results, just slower)
                        if fwd is eager: raise
                        print(f"torch.compile didn't work here ({str(e).splitlines()[0][:150]}); training without it", flush=True)
                        fwd = eager
                        with amp:
                            part = recipe_loss(fwd, xb, yb, P, n_out) * (len(c) / len(idx))
                    part.backward(); loss = loss + part.detach()
                break
            except torch.OutOfMemoryError:
                if micro <= 8:
                    print("The GPU is out of memory even in the smallest pieces: another program is probably using it. "
                          "Close it (or restart the PC) and try again.", flush=True)
                    raise
                micro //= 2
                for o in opts: o.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"The GPU ran out of memory, so training uses smaller pieces now ({micro} at a time). Same result, a bit slower.", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts: o.step()
        gate.after_step()  # slows down while you're using the PC (polite.py)
        ema = loss.detach() if ema is None else 0.98 * ema + 0.02 * loss.detach()  # stays on the GPU: no waiting every step
        if step % 10 == 0 or step + 1 == a.steps:
            ema = float(ema)
        if os.environ.get("GAMEAI_PROGRESS") and (step % 10 == 0 or step + 1 == a.steps):  # read by the Game AI app
            print("@@P" + json.dumps({"step": step + 1, "total": a.steps, "loss": round(ema, 4), "sec": round(time.time() - t0, 1),
                                      "size": a.size, "params": n_params}), flush=True)

        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ema = float(ema)
            model.eval()
            piece = micro * 4  # checking needs less memory than learning, but not unlimited
            with torch.no_grad(), amp:
                nv = min(val_d.n, 2000)
                parts = in_pieces(lambda i, j: (recipe_loss(model, *val_d.batch(torch.arange(i, j, device=device)), n_out).item() * (j - i)), nv, piece)
                vl = sum(parts) / max(1, nv)
            fa, ex = quick_accuracy(model, val_d, tok, amp, n_out, piece=piece)
            run.log(step + 1, train_loss=round(ema, 4), val_loss=round(vl, 4), field_acc=round(fa, 4), exact=round(ex, 4),
                    lr_mult=round(wsd(step, a.steps), 3), sec=round(time.time() - t0, 1))
            print(f"step {step+1:5d}  train {ema:.4f}  val {vl:.4f}  fields {fa:.2%}  exact {ex:.2%}  {time.time()-t0:.0f}s", flush=True)
            if os.environ.get("GAMEAI_PROGRESS"):
                print("@@E" + json.dumps({"step": step + 1, "val": round(vl, 4), "fields": round(fa, 4), "exact": round(ex, 4)}), flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0; save()
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop: validation stopped improving"); break

    ck = torch.load(ck_path, weights_only=False); model.load_state_dict(ck["model"])
    fa, ex = accuracy(model, tok, val_rows, device)
    shutil.copy(ck_path, "checkpoints/latest.pt")
    s = run.finish(params=n_params, val_loss=round(best, 4), field_acc=round(fa, 4), exact=round(ex, 4), checkpoint=ck_path)
    print("done:", s, "\nnext: python -m gameai.bench")


if __name__ == "__main__":
    main()
