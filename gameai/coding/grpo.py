"""The reinforcement-learning trial for code (plan: Phase 4, "GRPO trial"). The AI learns straight from the sandbox's
verdicts instead of from copied solutions: for one task it writes a group of answers, each is checked (1 if accepted,
0 if not, -0.5 for a cheating attempt), and it is nudged towards the answers that did better than the group's average
and away from the worse ones (Group Relative Policy Optimization). A small pull back towards the starting model (KL)
keeps it from drifting.

It's a trial: it starts from the code adapter in use, runs for a set time, and the result becomes a new version of the
code skill only if it solves more validation tasks than the version in use (adapters.decide), like any other version.

Only tasks the AI sometimes solves are used (from practice: some of its tries passed, some didn't). A task it always or
never solves gives every answer the same reward, so there is nothing to learn from it.

    python -m gameai.coding.grpo --base models/base/<key>/hf --minutes 90
"""
import argparse, json, os, random, time

GROUP = 6          # answers per task
MAX_NEW = 700      # tokens per answer
LR = 5e-6
BETA = 0.04        # strength of the pull back towards the starting model
CHEAT = -0.5


def reward(task, text):
    from .verify import verify
    r = verify(task, text)
    if any(x.startswith(("not allowed", "looks like")) for x in r["reasons"]): return CHEAT
    return 1.0 if r["accepted"] else 0.0


def pick_tasks():
    """training tasks the AI solved sometimes but not always in practice"""
    from . import tasks as T
    from .selfteach import STATS
    seen = {}
    if os.path.exists(STATS):
        for l in open(STATS, encoding="utf-8"):
            r = json.loads(l)
            if r.get("source", "self") != "self": continue
            a = seen.setdefault(r["task"], [0, 0]); a[0] += r["passed"]; a[1] += r["tries"]
    mid = {k for k, (p, n) in seen.items() if 0 < p < n}
    return [t for t in T.load("train") if t["id"] in mid]


def advantages(rewards):
    """each answer's reward relative to its group: (r - mean) / std; all equal -> nothing to learn"""
    n = len(rewards); m = sum(rewards) / n
    sd = (sum((r - m) ** 2 for r in rewards) / n) ** 0.5
    return [0.0] * n if sd < 1e-6 else [(r - m) / (sd + 1e-4) for r in rewards]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--minutes", type=float, default=90); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from .. import adapters
    from ..lora_train import base_precision, validate
    from ..phase0.qlora import load, add_lora, _cap_gpu, chat_ids
    from . import prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda": _cap_gpu(torch)
    random.seed(a.seed); torch.manual_seed(a.seed)
    tasks = pick_tasks()
    if len(tasks) < int(os.environ.get("GAMEAI_GRPO_MIN_TASKS", 10)): raise SystemExit(f"only {len(tasks)} tasks are in the useful middle (solved sometimes); practise more first")
    four = base_precision(a.base) != "bf16"
    print(f"Loading the base model ({'4-bit' if four else '16-bit'}) and the code adapter in use...", flush=True)
    model, tok = load(a.base, four_bit=four)
    if device == "cuda": model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    act = adapters.active("code")
    if not act and not os.environ.get("GAMEAI_GRPO_FROM_SCRATCH"):
        raise SystemExit("train the code skill first: the GRPO trial starts from the code adapter in use")
    model = PeftModel.from_pretrained(model, adapters._dir("code", act["version"]), is_trainable=True) if act else add_lora(model, four_bit=four)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    end, step, t0, rewards_log = time.time() + a.minutes * 60, 0, time.time(), []
    print(f"GRPO trial: {len(tasks)} tasks, {GROUP} answers each, {a.minutes:.0f} minutes", flush=True)

    def logprobs(ids, n_prompt):
        out = model(input_ids=ids).logits[:, :-1].float().log_softmax(-1)
        return out.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, n_prompt - 1:]

    while time.time() < end:
        task = random.choice(tasks)
        prompt = chat_ids(tok, prompts.messages(task)).to(device)
        n_prompt = prompt.shape[1]
        model.eval()
        with torch.no_grad():
            outs = model.generate(prompt.repeat(GROUP, 1), max_new_tokens=MAX_NEW, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
        texts = [tok.decode(o[n_prompt:], skip_special_tokens=True) for o in outs]
        rs = [reward(task, x) for x in texts]
        adv = advantages(rs)
        rewards_log.append(sum(r > 0 for r in rs) / GROUP)
        if not any(adv): continue
        model.train()
        for o, ad in zip(outs, adv):  # one answer at a time, to fit in 8 GB
            if ad == 0: continue
            ans = o[n_prompt:]
            keep = (ans != tok.pad_token_id).sum().item()
            ids = o[:n_prompt + keep].unsqueeze(0)
            lp = logprobs(ids, n_prompt)
            with torch.no_grad(), model.disable_adapter():
                ref = logprobs(ids, n_prompt)
            kl = (torch.exp(ref - lp) - (ref - lp) - 1).mean()   # an unbiased estimate of the drift from the starting model
            loss = (-(ad * lp.mean()) + BETA * kl) / GROUP
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 5 == 0:
            recent = rewards_log[-20:]
            print(f"  step {step}, {(time.time() - t0) / 60:.0f} min: answers passing (last 20 groups) {sum(recent) / len(recent):.0%}", flush=True)
        if os.environ.get("GAMEAI_PROGRESS"):
            print("@@P" + json.dumps({"step": int(time.time() - t0), "total": int(a.minutes * 60), "loss": None}), flush=True)
    print("Scoring the trial on validation...", flush=True)
    scores = validate(model, tok, "code", device)
    v, path = adapters.new_version("code")
    model.save_pretrained(path)
    adapters.finish("code", v, base=os.path.basename(os.path.dirname(os.path.normpath(a.base))), base_revision=None, scores=scores,
                    method="GRPO trial", steps=step, minutes=round((time.time() - t0) / 60, 1),
                    start_pass_rate=round(sum(rewards_log[:20]) / max(1, len(rewards_log[:20])), 3),
                    end_pass_rate=round(sum(rewards_log[-20:]) / max(1, len(rewards_log[-20:])), 3))
    ok, why = adapters.decide("code", v)
    print(f"GRPO trial -> code v{v}: {scores} -> {'now in use' if ok else 'kept, not used'} ({why})", flush=True)


if __name__ == "__main__":
    main()
