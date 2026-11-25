"""Train one skill adapter on the chosen base model (QLoRA), resumable at any point, then let validation decide.

    python -m gameai.lora_train --skill game_design --base models/base/hf --minutes 120 [--general 0.3] [--resume]

Training examples come from the skill's training split, mixed with the permanent general collection (share set by
--general; the plan tests 20/30/40/50%) so general ability isn't traded away. Every example goes through the guard
(no test questions). Every few minutes a full checkpoint is written (adapter, optimizer state, step, random state and
position in the data), so a stop, a crash or an update loses at most those minutes; --resume carries on.
At the end the new version is scored on every validation set and promoted only if it is better overall with no set
clearly worse (adapters.decide).
"""
import argparse, json, os, random, time

SKILLS = {  # skill -> (system prompt, training rows loader name). More skills arrive with their phases.
    "game_design": "recipes",
    "code": "verified first attempts (data/coding/first_pass.jsonl)",          # plan: first-pass and repair trained separately
    "code_repair": "verified fixes of failed code (data/coding/repair.jsonl)",
}


def chat_rows(skill):
    """-> list of {"system", "prompt", "answer"} for a skill's training split"""
    from .phase0.tasks import RECIPE_SYSTEM
    from .recipe import FIELD_NAMES
    if skill == "game_design":
        from . import splits
        rows = splits.rows("game_requests", "train") + splits.rows("real_talk", "train") + splits.rows("yours", "train")
        return [{"system": RECIPE_SYSTEM, "prompt": r["prompt"], "answer": "\n".join(f"{f}={r['recipe'].get(f, 'auto')}" for f in FIELD_NAMES)}
                for r in rows if "recipe" in r]
    if skill in ("code", "code_repair"):
        from .coding import selfteach
        return [{"system": r["system"], "prompt": r["prompt"], "answer": r["answer"]} for r in selfteach.rows("first" if skill == "code" else "repair")]
    raise KeyError(f"no training data defined for skill {skill!r} yet")


SELF_ANSWERS = os.path.join("data", "splits", "general", "self_answers.jsonl")


def general_rows(source="data"):
    """the general rows mixed into every batch: the datasets' own answers ("data"), or the base model's own answers to
    the same questions ("self", endogenous replay: research found it protects general ability best)"""
    from . import splits
    out = []
    rows = splits.read(SELF_ANSWERS) if source == "self" else splits.rows("general", "train")
    for r in rows:
        if "messages" in r: out.append({"messages": r["messages"]})
        elif "prompt" in r and "answer" in r: out.append({"system": r.get("system", ""), "prompt": r["prompt"], "answer": r["answer"]})
    return out


def encode(tok, row):
    """-> (ids, labels): only the answer (the assistant's turns) counts in the loss"""
    msgs = row.get("messages") or ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + \
        [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["answer"]}]
    ids, labels = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant": continue
        head = _tpl(tok, msgs[:i], True)
        full = _tpl(tok, msgs[:i + 1], False)
        a = tok(head, add_special_tokens=False)["input_ids"]
        b = tok(full, add_special_tokens=False)["input_ids"]
        if b[:len(a)] != a:  # template didn't extend cleanly: fall back to answer + end token
            b = a + tok(m["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        ids, labels = b, labels + [-100] * (len(a) - len(labels)) + b[len(a):]
        labels = labels[:len(ids)]
    return ids, labels


def _tpl(tok, msgs, gen):
    try: return tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False, enable_thinking=False)
    except Exception: return tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=False)


def save_ckpt(model, opt, path, state):
    import torch
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({"opt": opt.state_dict(), **state}, os.path.join(path, "trainer_state.pt.new"))
    os.replace(os.path.join(path, "trainer_state.pt.new"), os.path.join(path, "trainer_state.pt"))


def validate(model, tok, skill, device, n=300):
    """validation sets for this skill + the general collection: {set: {"score", "n"}} (score: 1 / (1 + loss), higher is better;
    for game requests also fully-right answers)"""
    from . import splits
    from .phase0.qlora import val_loss, val_exact
    from .phase0.tasks import RECIPE_SYSTEM
    out = {}
    if skill == "game_design":
        for name in ("game_requests", "real_talk", "yours"):
            rows = (splits.rows(name, "anchor") + splits.rows(name, "rotating"))[:n]
            rows = [r for r in rows if "recipe" in r]
            if len(rows) < 10: continue
            out[name] = {"score": round(val_exact(model, tok, rows[:min(len(rows), 100)], RECIPE_SYSTEM, device)["fully_right"], 4), "n": min(len(rows), 100)}
    if skill in ("code", "code_repair"):  # the real measure: validation tasks solved, checked in the sandbox
        from .coding import evaluate
        def generate(messages):
            import torch
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(device)
            model.eval()
            with torch.no_grad(): out_ids = model.generate(ids, max_new_tokens=900, do_sample=False)
            model.train()
            return tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
        r = evaluate.pass_rate(generate, repair=skill == "code_repair")
        out["coding_tasks"] = {"score": r["pass_rate"], "n": r["tasks"]}
    gen = (splits.rows("general", "anchor") + splits.rows("general", "rotating"))[:n]
    if gen:
        import torch
        tot = cnt = 0
        model.eval()
        with torch.no_grad():
            for r in gen:
                ids, lab = encode(tok, r if "messages" in r else {"system": r.get("system", ""), "prompt": r["prompt"], "answer": r["answer"]})
                ids_t = torch.tensor([ids], device=device); lab_t = torch.tensor([lab], device=device)
                k = int((lab_t != -100).sum())
                if k: tot += model(input_ids=ids_t, labels=lab_t).loss.item() * k; cnt += k
        model.train()
        out["general"] = {"score": round(1 / (1 + tot / max(1, cnt)), 4), "n": len(gen)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True); ap.add_argument("--base", required=True, help="folder with the base model (Hugging Face format)")
    ap.add_argument("--minutes", type=float, default=120); ap.add_argument("--general", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--accum", type=int, default=2); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--general-source", choices=["data", "self"], default="data")
    ap.add_argument("--ckpt-minutes", type=float, default=10); ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from . import adapters
    from .guard import filter_training
    from .phase0.qlora import load, add_lora, collate, _cap_gpu
    from .phase0.common import progress
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda": _cap_gpu(torch)
    work = os.path.join("models", "adapters", a.skill, "training")
    state = torch.load(os.path.join(work, "trainer_state.pt"), weights_only=False) if a.resume and os.path.exists(os.path.join(work, "trainer_state.pt")) else None
    if state: a.__dict__.update({k: v for k, v in state["args"].items() if k not in ("resume", "minutes")})
    random.seed(a.seed); torch.manual_seed(a.seed)
    print(f"Loading the base model ({a.base})...", flush=True)
    model, tok = load(a.base)
    if state:
        from peft import PeftModel
        if torch.cuda.is_available():
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, work, is_trainable=True)
        print(f"Carrying on from step {state['step']}", flush=True)
    else:
        model = add_lora(model)
    skill_rows = filter_training(chat_rows(a.skill), label=a.skill)
    gen_rows = filter_training(general_rows(a.general_source), label="general") if a.general > 0 else []
    if a.general > 0 and not gen_rows:
        print("Note: the general collection is still empty, so this run can't protect general ability yet.", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    step, done_s, rng = 0, 0.0, random.Random(a.seed)
    if state:
        opt.load_state_dict(state["opt"]); step, done_s = state["step"], state["seconds"]
        rng.setstate(state["rng"]); torch.set_rng_state(state["torch_rng"])
    t0 = time.time() - done_s; end = t0 + (done_s + a.minutes * 60 if state else a.minutes * 60); last_ck = time.time()
    model.train()
    print(f"Training the {a.skill} adapter: {len(skill_rows)} skill examples, {len(gen_rows)} general ({a.general:.0%} of each batch)", flush=True)
    while time.time() < end:
        for _ in range(a.accum):
            batch = [rng.choice(gen_rows) if gen_rows and rng.random() < a.general else rng.choice(skill_rows) for _ in range(a.batch)]
            ids, lab, att = collate([encode(tok, r) for r in batch], tok.pad_token_id, device)
            (model(input_ids=ids, attention_mask=att, labels=lab).loss / a.accum).backward()
        for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / 20)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if os.environ.get("GAMEAI_PROGRESS") and step % 5 == 0:
            print("@@P" + json.dumps({"step": int(time.time() - t0), "total": int(end - t0), "loss": None}), flush=True)
        if time.time() - last_ck > a.ckpt_minutes * 60:
            save_ckpt(model, opt, work, {"step": step, "seconds": time.time() - t0, "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "args": vars(a)})
            last_ck = time.time()
    save_ckpt(model, opt, work, {"step": step, "seconds": time.time() - t0, "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "args": vars(a)})
    print("Scoring the new version on validation...", flush=True)
    scores = validate(model, tok, a.skill, device)
    v, path = adapters.new_version(a.skill)
    model.save_pretrained(path)
    rev = (json.load(open(os.path.join(a.base, "provenance.json"))) if os.path.exists(os.path.join(a.base, "provenance.json")) else {}).get("revision")
    adapters.finish(a.skill, v, base=os.path.basename(os.path.normpath(a.base)), base_revision=rev, scores=scores,
                    steps=step, general_share=a.general, general_source=a.general_source, minutes=round((time.time() - t0) / 60, 1))
    ok, why = adapters.decide(a.skill, v)
    print(f"{a.skill} v{v}: {scores} -> {'now in use' if ok else 'kept, not used'} ({why})", flush=True)
    import shutil
    shutil.rmtree(work, ignore_errors=True)  # the finished version is saved; the training checkpoint isn't needed


if __name__ == "__main__":
    main()
