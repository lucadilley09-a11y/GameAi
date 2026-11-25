"""Muon optimizer (Keller Jordan, 2024), written out here so the project has no
outside dependency for it. Used for 2D hidden weight matrices; embeddings, norms
and the output head stay on AdamW (doc section 2)."""
import torch


def orthogonalize(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("buf", torch.zeros_like(p.grad))
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                upd = orthogonalize(upd, g["ns_steps"]).to(p.dtype)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd, alpha=-g["lr"] * scale)


def split_params(model):
    """Muon for 2D weights inside blocks; AdamW for everything else."""
    muon, adam = [], []
    for n, p in model.named_parameters():
        (muon if p.ndim == 2 and n.startswith("blocks.") else adam).append(p)
    return muon, adam
