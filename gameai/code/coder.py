"""The coder: your own AI that writes JavaScript enemy brains. Same kind of transformer as the game AI,
but it reads and writes letter by letter, so it can write any code.

    python -m gameai.code.coder train --size small        # train it (about 10-20 min on your GPU)
    python -m gameai.code.coder bench                     # how often its code works
    python -m gameai.code.coder write "enemies that circle you but run away when you get close"

It first writes what it understood (spec: ...), then the code. Writing works like the research plan says
("checking beats size"): it writes several tries, each is safety-checked and run in a sandbox against the
reference for its spec, and the first one that works is kept. If none works, it says so instead of guessing.
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from ..model import GameAI, Config, SIZES
from ..muon import Muon, split_params
from . import specs, sandbox

# 5 combinations it never practises: the test of whether it can put pieces together in new ways
HOLDOUT = {("circle", "flee"), ("zigzag", "pace_y"), ("spiral", "still"), ("pace_x", "chase"), ("still", "zigzag")}
MAX_PROMPT, MAX_OUT = 160, 340


class CharTok:
    SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"]

    def __init__(self):
        self.itos = self.SPECIALS + [chr(c) for c in range(32, 127)] + ["\n"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad, self.bos, self.sep, self.eos = 0, 1, 2, 3
        self.vocab_size = len(self.itos)
        self.max_len = 1 + MAX_PROMPT + 1 + MAX_OUT + 1

    def enc(self, s): return [self.stoi.get(c, self.stoi["?"]) for c in s]
    def prompt(self, p): return [self.bos] + self.enc(p.lower()[:MAX_PROMPT]) + [self.sep]
    def dec(self, ids): return "".join(self.itos[i] for i in ids if i > 3)


def held_out(s):
    return s["kind"] == "near" and (s["a"]["kind"], s["b"]["kind"]) in HOLDOUT


def make_rows(n, seed, holdout=False):
    """practice requests and correct answers ("spec: ...\\n<code>"); holdout=True gives only the never-practised combos"""
    rng, rows = random.Random(seed), []
    while len(rows) < n:
        s = specs.random_spec(rng, True if holdout else None)
        if held_out(s) != holdout: continue
        rows.append({"prompt": specs.say(s, rng), "spec": specs.spec_text(s), "out": "spec: " + specs.spec_text(s) + "\n" + specs.code(s, rng)})
    return rows


def load(path="models/coder.pt", device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GameAI(Config(**ck["config"])).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m, CharTok(), ck


@torch.no_grad()
def sample(model, tok, prompts, device, n=1, temp=0.0):
    """write n tries per prompt (the first try is always the most likely one) -> list of lists of strings"""
    seqs = [tok.prompt(p) for p in prompts for _ in range(n)]
    L = max(map(len, seqs))
    x = torch.tensor([[tok.pad] * (L - len(s)) + s for s in seqs], device=device)
    last, st = model.prefill(x, tok.pad)
    out = [[] for _ in seqs]; done = torch.zeros(len(seqs), dtype=torch.bool, device=device)
    greedy = torch.tensor([(i % n == 0) or temp <= 0 for i in range(len(seqs))], device=device)
    for _ in range(MAX_OUT):
        lg = last.float()
        g = lg.argmax(-1)
        s = torch.multinomial(F.softmax(lg / max(temp, 1e-3), -1), 1)[:, 0] if temp > 0 else g
        nxt = torch.where(greedy, g, s)
        nxt = torch.where(done, torch.full_like(nxt, tok.pad), nxt)
        for i, t in enumerate(nxt.tolist()):
            if not done[i] and t != tok.eos: out[i].append(t)
        done |= nxt == tok.eos
        if bool(done.all()): break
        last = model.extend(torch.where(done, torch.full_like(nxt, tok.eos), nxt), st)
    texts = [tok.dec(o) for o in out]
    return [texts[i * n:(i + 1) * n] for i in range(len(prompts))]


def split_out(text):
    """'spec: X\\ncode' -> (X, code)"""
    head, _, code = text.partition("\n")
    return head.replace("spec:", "", 1).strip(), code.strip()


def write(model, tok, prompt, device, tries=8, temp=0.8):
    """-> {"ok", "spec", "code", "tries", "reason"}: the first try that is safe and works"""
    t0, why = time.time(), "no tries"
    cands = sample(model, tok, [prompt], device, n=tries, temp=temp)[0]
    for i, c in enumerate(cands):
        sp, code = split_out(c)
        ok, why = sandbox.verify(sp, code, seed=i)
        if ok:
            return {"ok": True, "spec": sp, "code": code, "tries": i + 1, "words": specs.describe(specs.parse(sp)), "ms": int((time.time() - t0) * 1000)}
    sp, code = split_out(cands[0])
    return {"ok": False, "spec": sp, "code": code, "tries": len(cands), "reason": why, "ms": int((time.time() - t0) * 1000)}


def bench(model, tok, device, n=300):
    """pass@1 (its first try works and matches what was asked) and pass@8, on new requests and on never-practised combos"""
    res = {}
    for name, rows in (("fresh", make_rows(n, 777)), ("new_combos", make_rows(n // 2, 888, holdout=True))):
        p1 = p8 = und = 0
        outs = sample(model, tok, [r["prompt"] for r in rows], device, n=8, temp=0.8)
        for r, cands in zip(rows, outs):
            good = []
            for i, c in enumerate(cands):
                sp, code = split_out(c)
                good.append(specs.parse(sp) == specs.parse(r["spec"]) and sandbox.verify(sp, code, seed=i)[0])
            p1 += good[0]; p8 += any(good); und += specs.parse(split_out(cands[0])[0]) == specs.parse(r["spec"])
        res[name] = {"pass1": p1 / len(rows), "pass8": p8 / len(rows), "understood": und / len(rows), "n": len(rows)}
    return res


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = CharTok()
    rows = make_rows(a.data, a.seed)
    if os.path.exists("data/coder_teacher.jsonl"):
        rows += [json.loads(l) for l in open("data/coder_teacher.jsonl", encoding="utf-8")] * 2
    seqs = []
    for r in rows:
        pr, o = tok.prompt(r["prompt"]), tok.enc(r["out"][:MAX_OUT]) + [tok.eos]
        seqs.append((pr, o))
    L = max(len(p) + len(o) for p, o in seqs)
    X = torch.full((len(seqs), L), tok.pad, dtype=torch.long); Y = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, (p, o) in enumerate(seqs):  # loss only on the answer (spec + code), not on the request
        s = p + o; X[i, :len(s)] = torch.tensor(s); Y[i, len(p) - 1:len(s) - 1] = torch.tensor(o)
    X, Y, lens = X.to(device), Y.to(device), torch.tensor([len(p) + len(o) for p, o in seqs], device=device)
    l, d, h, kv = SIZES[a.size]
    model = GameAI(Config(tok.vocab_size, tok.max_len, l, d, h, kv)).to(device)
    if a.init and os.path.exists(a.init):
        model.load_state_dict(load(a.init, device)[0].state_dict())
    mu, ad = split_params(model)
    opts = [Muon(mu, lr=a.muon_lr, weight_decay=0.01), torch.optim.AdamW(ad, lr=a.lr, betas=(0.9, 0.95))]
    for o in opts:
        for g in o.param_groups: g["base_lr"] = g["lr"]
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    print(f"coder: {model.num_params() / 1e6:.2f}M parameters on {device}, {len(rows)} examples", flush=True)
    t0, ema = time.time(), None
    for step in range(a.steps):
        f = min(1, step / 100) * (1 if step < a.steps * .8 else (a.steps - step) / (a.steps * .2))  # warm up, hold, cool down
        for o in opts:
            for g in o.param_groups: g["lr"] = g["base_lr"] * max(f, 0.02)
        idx = torch.randint(len(seqs), (a.batch,), device=device)
        m = int(lens[idx].max())
        with amp:
            lg = model(X[idx, :m])
        loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), Y[idx, :m].reshape(-1), ignore_index=-100)
        for o in opts: o.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts: o.step()
        ema = loss.detach() if ema is None else .98 * ema + .02 * loss.detach()
        if step % 10 == 0 and os.environ.get("GAMEAI_PROGRESS"):
            print("@@P" + json.dumps({"step": step, "total": a.steps, "loss": round(float(ema), 4)}), flush=True)
        if (step + 1) % 250 == 0 or step + 1 == a.steps:
            print(f"step {step + 1:5d}  loss {float(ema):.4f}  {time.time() - t0:.0f}s", flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    ck = {"model": model.state_dict(), "config": model.config_dict(), "size": a.size, "kind": "coder", "time": time.time()}
    torch.save(ck, "checkpoints/coder_latest.pt")
    model.eval()
    b = bench(model, tok, device, a.bench_n)
    print(f"Tested: first try works {b['fresh']['pass1']:.1%} (within 8 tries {b['fresh']['pass8']:.1%}); "
          f"combos it never practised: {b['new_combos']['pass1']:.1%} ({b['new_combos']['pass8']:.1%} within 8 tries)", flush=True)
    old = json.load(open("models/coder.json")) if os.path.exists("models/coder.json") and os.path.exists("models/coder.pt") else None
    score = lambda r: r["fresh"]["pass8"] + r["new_combos"]["pass8"] + .5 * (r["fresh"]["pass1"] + r["new_combos"]["pass1"])
    if old is None or score(b) > score(old["bench"]):
        os.makedirs("models", exist_ok=True); torch.save(ck, "models/coder.pt")
        json.dump({"bench": b, "size": a.size, "params": model.num_params(), "time": time.time()}, open("models/coder.json", "w"), indent=1)
        print("New best coder: now in use.", flush=True)
    else:
        print("Not better than the coder in use, so it was kept.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--size", default="small"); t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=64); t.add_argument("--data", type=int, default=30000)
    t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--muon-lr", type=float, default=0.015)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--init", default=None); t.add_argument("--bench-n", type=int, default=300)
    sub.add_parser("bench")
    w = sub.add_parser("write"); w.add_argument("text")
    a = ap.parse_args()
    if a.cmd == "train": return train(a)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load("models/coder.pt", device)
    if a.cmd == "bench": print(json.dumps(bench(model, tok, device), indent=1))
    else: print(json.dumps(write(model, tok, a.text, device), indent=1))


if __name__ == "__main__":
    main()
