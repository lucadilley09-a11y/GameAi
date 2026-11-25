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


@torch.no_grad()
def decode(model, tok, prompts, device, batch=256, gate=GATE):
    """Constrained greedy decoding: at slot i only field i's values are allowed,
    so every output is a valid recipe. Returns [{field: (value, confidence)}]."""
    model.eval()
    out = []
    for i in range(0, len(prompts), batch):
        x, _ = left_pad([tok.encode_prompt(p) for p in prompts[i:i + batch]], tok.pad)
        x = x.to(device)
        res = [dict() for _ in range(x.size(0))]
        chosen = [dict() for _ in range(x.size(0))]
        for slot, f in enumerate(OUT_FIELDS):
            ids = torch.tensor(tok.field_ids[slot], device=device)
            vals = [tok.decode_value(slot, t) for t in tok.field_ids[slot]]
            logits = model(x)[:, -1, :].float()[:, ids]
            mask = torch.tensor([[v in allowed(f, ch) for v in vals] for ch in chosen], device=device)
            probs = logits.masked_fill(~mask, float("-inf")).softmax(-1)  # grammar: forbidden values get zero probability
            conf, j = probs.max(-1)
            chosen_ids = ids[j]
            for r, (c, t) in enumerate(zip(conf.tolist(), chosen_ids.tolist())):
                v = tok.decode_value(slot, t)
                chosen[r][f] = v
                res[r][f] = (v if c >= gate or v == AUTO or f not in FIELD_NAMES else AUTO, c)
            x = torch.cat([x, chosen_ids[:, None]], 1)
        out += res
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
