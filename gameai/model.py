"""A decoder-only transformer, written from scratch.
Modern parts (doc section 1): RoPE positions, RMSNorm, SwiGLU, grouped-query attention,
fused attention through PyTorch SDPA (FlashAttention on GPU)."""
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    max_len: int
    n_layer: int = 6
    d_model: int = 256
    n_head: int = 8
    n_kv_head: int = 4
    dropout: float = 0.0


SIZES = {  # name: (layers, width, heads, kv heads)  -- parameters printed at start of training
    "tiny": (4, 128, 4, 2),
    "small": (6, 256, 8, 4),
    "base": (8, 384, 6, 2),
    "large": (12, 512, 8, 4),
}


def make_config(size, vocab_size, max_len, dropout=0.0):
    l, d, h, kv = SIZES[size]
    return Config(vocab_size, max_len, l, d, h, kv, dropout)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(max_len, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: (B, H, T, D); cos/sin: (B, 1, T, D/2) gathered per token position
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv, self.hd = c.n_head, c.n_kv_head, c.d_model // c.n_head
        self.q = nn.Linear(c.d_model, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, c.d_model, bias=False)
        self.drop = c.dropout

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        hid = int(math.ceil(8 * c.d_model / 3 / 64) * 64)
        self.gate = nn.Linear(c.d_model, hid, bias=False)
        self.up = nn.Linear(c.d_model, hid, bias=False)
        self.down = nn.Linear(hid, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = RMSNorm(c.d_model), Attention(c), RMSNorm(c.d_model), SwiGLU(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, mask):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, mask))
        return x + self.drop(self.mlp(self.n2(x)))


class GameAI(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        cos, sin = rope_cache(c.max_len, c.d_model // c.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():  # scale residual outputs down with depth
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, pad_id=0):
        """Left padding is invisible: pads are masked out of attention and positions
        start at the first real token, so a prompt gives the same answer alone or in a batch."""
        T = idx.size(1)
        real = idx != pad_id                                        # (B, T)
        pos = (real.long().cumsum(1) - 1).clamp(min=0)              # (B, T)
        cos, sin = self.cos[pos][:, None], self.sin[pos][:, None]   # (B, 1, T, D/2)
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        eye = torch.eye(T, dtype=torch.bool, device=idx.device)    # pads attend to themselves only (avoids empty rows)
        mask = (causal[None] & real[:, None, :]) | eye[None]        # (B, T, T)
        mask = mask[:, None]
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin, mask)
        return self.head(self.norm(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.config)
