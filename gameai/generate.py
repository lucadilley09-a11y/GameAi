"""Turn prompts into recipes with grammar-constrained decoding.

    python -m gameai.generate "a fast ninja escaping a spooky dungeon"
"""
import sys
import torch
from .recipe import FIELD_NAMES, OUT_FIELDS, AUTO, allowed
from .tokenizer import Tokenizer
from .model import GameAI, Config

GATE = 0.45  # below this confidence a field becomes "auto" instead of a guess


def left_pad(seqs, pad):
    P = max(len(s) for s in seqs)
    return torch.tensor([[pad] * (P - len(s)) + s for s in seqs]), P


_MASKS = {}


def _masks(tok, device):
    """Grammar rules as small lookup tables, built once: for each slot, which values are allowed
    given the value chosen for the field it depends on. Lets the whole batch be checked at once."""
    key = (id(tok), str(device))
    if key not in _MASKS:
        from .recipe import CONSTRAINTS
        vals = [[tok.decode_value(s, t) for t in tok.field_ids[s]] for s in range(len(OUT_FIELDS))]
        tabs = []
        for s, f in enumerate(OUT_FIELDS):
            if f in CONSTRAINTS:
                dep = CONSTRAINTS[f][0]; ds = OUT_FIELDS.index(dep)
                t = torch.tensor([[v in allowed(f, {dep: pv}) for v in vals[s]] for pv in vals[ds]], device=device)
                tabs.append((ds, t))
            else:
                tabs.append(None)
        _MASKS[key] = (vals, tabs, [torch.tensor(ids, device=device) for ids in tok.field_ids])
    return _MASKS[key]


@torch.no_grad()
def decode(model, tok, prompts, device, batch=512, gate=GATE):
    """Constrained greedy decoding: at slot i only field i's values are allowed,
    so every output is a valid recipe. Returns [{field: (value, confidence)}]."""
    model.eval()
    vals, tabs, ids_t = _masks(tok, device)
    enc = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(enc)), key=lambda k: len(enc[k]))  # similar lengths together: far less padding
    out = [None] * len(enc)
    for i in range(0, len(enc), batch):
        idx = order[i:i + batch]
        x, _ = left_pad([enc[k] for k in idx], tok.pad)
        x = x.to(device)
        picks, confs = [], []
        last, state = model.prefill(x, tok.pad)  # read the prompt once; each setting after that is one cheap step
        for slot in range(len(OUT_FIELDS)):
            logits = last.float()[:, ids_t[slot]]
            if tabs[slot] is not None:
                ds, t = tabs[slot]
                logits = logits.masked_fill(~t[picks[ds]], float("-inf"))  # grammar: forbidden values get zero probability
            conf, j = logits.softmax(-1).max(-1)
            picks.append(j); confs.append(conf)
            if slot + 1 < len(OUT_FIELDS):
                last = model.extend(ids_t[slot][j], state)
        P, C = torch.stack(picks, 1).tolist(), torch.stack(confs, 1).tolist()
        for k, pr, cr in zip(idx, P, C):
            res = {}
            for slot, f in enumerate(OUT_FIELDS):
                v, c = vals[slot][pr[slot]], cr[slot]
                res[f] = (v if c >= gate or v == AUTO or f not in FIELD_NAMES else AUTO, c)
            out[k] = res
    return out


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    tok = Tokenizer()
    if ck["tokenizer"] != tok.itos:
        raise ValueError("This AI was trained on an older version of the game language. Train a new one with /train.")
    model = GameAI(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), tok, ck


def show(model, tok, prompt, device):
    for f, (v, c) in decode(model, tok, [prompt], device)[0].items():
        print(f"  {f:11s} {v:10s} {c:.0%}")


def main():
    import os
    path = "models/current.pt" if len(sys.argv) < 3 else sys.argv[2]
    if not os.path.exists(path):
        print("No trained AI yet. Train one first (start.bat, option 1)."); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load(path, device)
    if len(sys.argv) > 1:
        show(model, tok, sys.argv[1], device); return
    print("Describe a game and press Enter. Press Enter on an empty line to go back.\n")
    while True:
        prompt = input("> ").strip()
        if not prompt:
            break
        show(model, tok, prompt, device)
        print()


if __name__ == "__main__":
    main()
