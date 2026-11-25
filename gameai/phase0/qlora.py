"""The real training test for one candidate: QLoRA (the base frozen in 4-bit, small trainable adapters) on this PC.

Measures: GPU memory at several context lengths and batch sizes (the largest that fit), training speed, and how much a
30-minute run improves the candidate on the validation game requests (loss and fully-right answers), so candidates
can be compared on improvement per hour. Uses Transformers + PEFT + bitsandbytes (the training format); the
release format (GGUF) is separate.
"""
import json, math, os, random, time

from .common import progress, say

TRAIN_MINUTES = 30
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05)
EXCLUDE = r".*(vision|visual|audio|mm_proj|multi_modal|embed_tokens|lm_head).*"


def _cap_gpu(torch):
    """Same as training: never let Windows quietly spill into shared memory (that would make the numbers wrong)."""
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(0.95, (free + torch.cuda.memory_reserved() - 400 * 2**20) / total)))


def load(hf_dir, four_bit=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    kw = {"dtype": torch.bfloat16 if cuda else torch.float32}
    if cuda and four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    def attempt():
        tok = AutoTokenizer.from_pretrained(hf_dir)
        try:
            return AutoModelForCausalLM.from_pretrained(hf_dir, **kw), tok
        except (ValueError, KeyError):  # models that also see pictures load through the image-and-text class
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(hf_dir, **kw), tok
    try:
        model, tok = attempt()
    except (ValueError, KeyError, ImportError) as e:
        if "recogni" not in str(e) and "does not support" not in str(e) and "model type" not in str(e): raise
        say("  this model is newer than the installed Transformers: updating it once, then trying again...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft", "accelerate"], check=True)
        raise RuntimeError("Transformers was updated: run the bake-off again for this model (it carries on where it stopped)")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok


def add_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    if torch.cuda.is_available(): model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(target_modules="all-linear", exclude_modules=EXCLUDE, task_type="CAUSAL_LM", **LORA)
    return get_peft_model(model, cfg)


def example(tok, prompt, answer, system):
    """-> (ids, labels): the loss only counts the answer, not the request"""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    kw = {"enable_thinking": False}
    try: head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    except Exception: head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    a = tok(head, add_special_tokens=False)["input_ids"]; b = tok(answer + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    return a + b, [-100] * len(a) + b


def batches(rows, tok, system, size, rng):
    from .tasks import FIELD_NAMES
    while True:
        pick = rng.sample(rows, size)
        yield [example(tok, r["prompt"], "\n".join(f"{f}={r['recipe'][f]}" for f in FIELD_NAMES), system) for r in pick]


def collate(exs, pad, device):
    import torch
    L = max(len(i) for i, _ in exs)
    ids = torch.full((len(exs), L), pad); lab = torch.full((len(exs), L), -100); att = torch.zeros((len(exs), L), dtype=torch.long)
    for k, (i, l) in enumerate(exs):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def val_loss(model, tok, rows, system, device, bs=8):
    import torch
    model.eval(); tot = n = 0
    exs = [example(tok, r["prompt"], "\n".join(f"{f}={r['recipe'][f]}" for f in __import__("gameai.recipe", fromlist=["FIELD_NAMES"]).FIELD_NAMES), system) for r in rows]
    with torch.no_grad():
        for i in range(0, len(exs), bs):
            ids, lab, att = collate(exs[i:i + bs], tok.pad_token_id, device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            k = int((lab != -100).sum()); tot += out.loss.item() * k; n += k
    model.train()
    return tot / max(1, n)


def val_exact(model, tok, rows, system, device):
    """generate the settings for each validation request (greedy) and count fully-right answers"""
    import torch
    from .tasks import parse_recipe, score_recipes
    model.eval(); preds = []
    for r in rows:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]
        try: head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except Exception: head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(head, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=110, do_sample=False, pad_token_id=tok.pad_token_id)
        preds.append(parse_recipe(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))
    model.train()
    return score_recipes(rows, preds)


def memory_probe(model, tok, device, lengths=(512, 1024, 2048, 4096, 8192), batches_=(1, 2, 4, 8, 16, 32)):
    """the biggest context length (batch 1) and the biggest batch (at 512) that train without running out of memory"""
    import torch
    res, opt = {"context": {}, "batch_at_512": {}}, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)

    def try_step(B, L):
        ids = torch.randint(10, 1000, (B, L), device=device)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(2):
            out = model(input_ids=ids, labels=ids); out.loss.backward(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return round(torch.cuda.max_memory_allocated() / 2**30, 2), round((time.perf_counter() - t) / 2, 3)
    for L in lengths:
        try:
            gb, s = try_step(1, L); res["context"][L] = {"peak_gb": gb, "step_s": s}
        except torch.OutOfMemoryError:
            res["context"][L] = "out of memory"; break
        finally:
            torch.cuda.empty_cache()
    for B in batches_:
        try:
            gb, s = try_step(B, 512); res["batch_at_512"][B] = {"peak_gb": gb, "step_s": s}
        except torch.OutOfMemoryError:
            res["batch_at_512"][B] = "out of memory"; break
        finally:
            torch.cuda.empty_cache()
    ok_l = [L for L, v in res["context"].items() if isinstance(v, dict)]
    ok_b = [B for B, v in res["batch_at_512"].items() if isinstance(v, dict)]
    res["max_context"] = max(ok_l) if ok_l else 0
    res["max_batch_at_512"] = max(ok_b) if ok_b else 0
    return res


def run(hf_dir, minutes=TRAIN_MINUTES, seed=0, n_val_loss=200, n_val_exact=60, adapter_out=None):
    import torch
    from .tasks import RECIPE_SYSTEM, recipe_items
    from ..guard import filter_training
    cuda = torch.cuda.is_available(); device = "cuda" if cuda else "cpu"
    rng = random.Random(seed); torch.manual_seed(seed)
    if cuda: _cap_gpu(torch); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = load(hf_dir)
    res = {"load_s": round(time.time() - t0, 1), "four_bit": cuda}
    model = add_lora(model)
    res["trainable_params_m"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
    if cuda:
        say("  measuring memory at different context lengths and batch sizes...")
        res["memory"] = memory_probe(model, tok, device)
    val = recipe_items()
    vloss_rows, vex_rows = val[:n_val_loss], val[:n_val_exact]
    train_rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")] if os.path.exists("data/train.jsonl") else []
    train_rows = filter_training(train_rows[:50000], log=None)
    if not train_rows: return {**res, "error": "no training data (type /train once first)"}
    say("  checking it before training...")
    res["before"] = {"val_loss": round(val_loss(model, tok, vloss_rows, RECIPE_SYSTEM, device), 4), **val_exact(model, tok, vex_rows, RECIPE_SYSTEM, device)}
    say("   ", res["before"])
    B = min(8, (res.get("memory") or {}).get("max_batch_at_512", 8) or 1); accum = max(1, 16 // B)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    gen, curve, steps, toks, t_start = batches(train_rows, tok, RECIPE_SYSTEM, B, rng), [], 0, 0, time.time()
    end, next_check = t_start + minutes * 60, t_start + minutes * 60 / 5
    model.train()
    say(f"  training for {minutes} minutes (batch {B} x {accum})...")
    while time.time() < end:
        for _ in range(accum):
            ids, lab, att = collate(next(gen), tok.pad_token_id, device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
            loss.backward(); toks += int(att.sum())
        lr_mult = min(1.0, (steps + 1) / 20)
        for g in opt.param_groups: g["lr"] = 2e-4 * lr_mult
        opt.step(); opt.zero_grad(set_to_none=True); steps += 1
        progress(int(time.time() - t_start), minutes * 60)
        if time.time() >= next_check:
            vl = val_loss(model, tok, vloss_rows, RECIPE_SYSTEM, device)
            curve.append({"minutes": round((time.time() - t_start) / 60, 1), "steps": steps, "val_loss": round(vl, 4)}); say("   ", curve[-1])
            next_check += minutes * 60 / 5
    hours = (time.time() - t_start) / 3600
    res["train"] = {"steps": steps, "examples_per_s": round(steps * B * accum / (hours * 3600), 2), "tokens_per_s": round(toks / (hours * 3600), 1), "curve": curve}
    res["after"] = {"val_loss": round(val_loss(model, tok, vloss_rows, RECIPE_SYSTEM, device), 4), **val_exact(model, tok, vex_rows, RECIPE_SYSTEM, device)}
    say("   ", res["after"])
    res["improvement_per_hour"] = {"val_loss_drop": round((res["before"]["val_loss"] - res["after"]["val_loss"]) / hours, 4),
                                   "fully_right_gain": round((res["after"]["fully_right"] - res["before"]["fully_right"]) / hours, 4)}
    if cuda: res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if adapter_out: model.save_pretrained(adapter_out)  # adapters are small (tens of MB)
    del model, opt
    if cuda: torch.cuda.empty_cache()
    return res
