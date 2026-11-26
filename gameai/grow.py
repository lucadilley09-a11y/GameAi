"""Growing a trained AI into a bigger size, keeping what it learned (training speed-up 1, approved by you).

Instead of a bigger AI starting from random numbers, it starts as a copy of the current AI laid out in the bigger shape,
so on its first step it already answers like the current one and training only has to add to it. Research: growing a
smaller trained model this way (bert2BERT, LiGO, HyperCloning, 2021-2024) reaches the same quality with roughly 30-70%
less training compute than starting from scratch.

How each part carries over (exactly, apart from tiny noise in the brand-new parts so they can start learning):
  - more layers: the new layers start as "pass-through" (their outputs are zero), spread evenly between the old ones;
  - wider: the old numbers go in the first columns; the norms are rescaled so the wider average gives the same result;
  - more or wider attention heads: each old head goes into a new head (a wider head holds it in every other rotary pair
    and is scaled to match), and each group of heads that shares keys and values gets a copy of its old keys/values;
  - bigger feed-forward: the extra neurons start switched off (their output weights are zero).
If the new shape has fewer head groups than the old one, some old heads can't be placed: those are dropped and it says
so (an approximate start, still far ahead of random).

    model, info = grow(old_model, "large")
"""
import math

import torch

from .model import Config, GameAI, SIZES

NOISE = 1e-4   # size of the noise in brand-new parts (keeps answers the same while letting them learn)
INIT = 0.02    # brand-new parts that are cut off by a zero (their effect starts at zero anyway) get normal starting values


def _rand(*shape, std=INIT, like=None):
    return torch.randn(*shape, dtype=like.dtype, device=like.device) * std


def _place_heads(h, kv, H, KV):
    """-> (slot of each old q head or None, source old kv group for each new kv group or None)"""
    r, R = h // kv, H // KV
    slot, src, g = [None] * h, [None] * KV, 0
    for old_g in range(kv):
        heads = list(range(old_g * r, (old_g + 1) * r))
        while heads and g < KV:
            src[g] = old_g
            for s in range(R):
                if heads: slot[heads.pop(0)] = g * R + s
            g += 1
    return slot, src


def _dim_map(hd, HD):
    """old head dim i -> new head dim, keeping each rotary pair at the same speed (pair p -> pair p * k)"""
    k = HD // hd
    return [2 * (i // 2) * k + i % 2 for i in range(hd)], k


def grow(old, size=None, config=None):
    """old: a trained GameAI. -> (bigger GameAI with the same answers, info)"""
    c0 = old.config
    if config is None:
        l, d, h, kv = SIZES[size]
        config = Config(c0.vocab_size, c0.max_len, l, d, h, kv, c0.dropout)
    c1 = config
    d, D, hd, HD = c0.d_model, c1.d_model, c0.d_model // c0.n_head, c1.d_model // c1.n_head
    if D < d or c1.n_layer < c0.n_layer or HD < hd or HD % hd or c1.vocab_size != c0.vocab_size:
        raise ValueError("can only grow into a size at least as wide and deep, with the same game language")
    new = GameAI(c1).to(next(old.parameters()).device)
    o, n = old.state_dict(), new.state_dict()
    out = {k: v.clone() for k, v in n.items()}
    s = math.sqrt(d / D)  # RMS over D dims, D-d of them ~0, is sqrt(d/D) times the RMS over d dims
    slot, src = _place_heads(c0.n_head, c0.n_kv_head, c1.n_head, c1.n_kv_head)
    dmap, k = _dim_map(hd, HD)
    dropped = sum(x is None for x in slot)

    with torch.no_grad():
        for name in ("emb.weight", "head.weight"):
            w = _rand(c1.vocab_size, D, std=NOISE, like=o[name]); w[:, :d] = o[name]; out[name] = w
        nw = torch.full((D,), s, dtype=o["norm.weight"].dtype, device=o["norm.weight"].device); nw[:d] = o["norm.weight"] * s
        out["norm.weight"] = nw
        # spread the old layers evenly; the rest start as pass-through
        L0, L1 = c0.n_layer, c1.n_layer
        where = [round(i * (L1 - 1) / max(1, L0 - 1)) if L0 > 1 else 0 for i in range(L0)]
        hid0, hid1 = o["blocks.0.mlp.gate.weight"].shape[0], n["blocks.0.mlp.gate.weight"].shape[0]
        for j in range(L1):
            p = f"blocks.{j}."
            if j not in where:  # pass-through layer: normal start, zero output
                out[p + "attn.o.weight"].zero_(); out[p + "mlp.down.weight"].zero_()
                continue
            q = f"blocks.{where.index(j)}."
            like = o[q + "n1.weight"]
            for nn_ in ("n1", "n2"):
                w = torch.full((D,), s, dtype=like.dtype, device=like.device); w[:d] = o[q + nn_ + ".weight"] * s
                out[p + nn_ + ".weight"] = w
            # attention: queries
            Wq = _rand(c1.n_head * HD, D, std=NOISE, like=like)
            for i, sl in enumerate(slot):
                if sl is None: continue
                for a, b in enumerate(dmap):
                    Wq[sl * HD + b, :d] = o[q + "attn.q.weight"][i * hd + a] * math.sqrt(k)  # same scores with the wider head
            for t in range(c1.n_head):  # brand-new heads (their output starts at zero) get normal starting values
                if t not in slot: Wq[t * HD:(t + 1) * HD] = _rand(HD, D, like=like)
            out[p + "attn.q.weight"] = Wq
            # keys and values, per group
            for name in ("k", "v"):
                W = _rand(c1.n_kv_head * HD, D, std=NOISE, like=like)
                for g, sg in enumerate(src):
                    if sg is None: W[g * HD:(g + 1) * HD] = _rand(HD, D, like=like); continue
                    for a, b in enumerate(dmap): W[g * HD + b, :d] = o[q + f"attn.{name}.weight"][sg * hd + a]
                out[p + f"attn.{name}.weight"] = W
            Wo = torch.zeros(D, c1.n_head * HD, dtype=like.dtype, device=like.device)
            for i, sl in enumerate(slot):
                if sl is None: continue
                for a, b in enumerate(dmap): Wo[:d, sl * HD + b] = o[q + "attn.o.weight"][:, i * hd + a]
            out[p + "attn.o.weight"] = Wo
            # feed-forward: extra neurons switched off
            for name in ("gate", "up"):
                W = _rand(hid1, D, like=like); W[:, d:] *= NOISE / INIT; W[:hid0, :d] = o[q + f"mlp.{name}.weight"]; W[:hid0, d:] = _rand(hid0, D - d, std=NOISE, like=like)
                out[p + f"mlp.{name}.weight"] = W
            W = torch.zeros(D, hid1, dtype=like.dtype, device=like.device); W[:d, :hid0] = o[q + "mlp.down.weight"]
            out[p + "mlp.down.weight"] = W
    new.load_state_dict(out)
    info = {"from": {"layers": L0, "width": d, "heads": c0.n_head}, "to": {"layers": L1, "width": D, "heads": c1.n_head},
            "params": [old.num_params(), new.num_params()], "exact": dropped == 0, "heads_dropped": dropped}
    return new, info


def describe(info):
    a, b = info["params"]
    s = f"Started the bigger AI from your current one ({a / 1e6:.1f}M -> {b / 1e6:.1f}M parameters): it begins already knowing what the current AI knows"
    if not info["exact"]: s += f" (an approximate start: {info['heads_dropped']} of its attention heads didn't fit the new shape and were left out)"
    return s
