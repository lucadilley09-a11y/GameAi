"""The teacher: a free open AI (Qwen3.5, Apache 2.0) writes natural, varied game requests
for your AI to learn from. Your AI never needs the teacher to run.

    python -m gameai.teacher --n 2000        (or /teacher 2000 in the app)

1. our generator picks a game (so the right answer is always known)
2. the teacher writes how a real person might ask for it, in a random style
3. the teacher reads its own request back; the example is kept ONLY if it gets the same game
Kept examples go to data/teacher.jsonl; every 10th goes to data/test_teacher.jsonl (the "real talk" test).
First run downloads the teacher (about 6 GB) from Hugging Face automatically; no account needed.
"""
import argparse, json, os, random, re, sys, time
from .recipe import FIELDS, FIELD_NAMES, OUT_FIELDS, AUTO, rules_of, RULE_VALUES, is_valid_out
from .datagen import make_example

CANDIDATES = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]  # 4B first: about twice as fast; use --model Qwen/Qwen3.5-9B for the bigger one
STYLES = ["casual, all lowercase, like a text message", "an excited kid", "short and direct", "a longer description with a bit of story",
          "with one or two small spelling mistakes", "polite", "gamer slang", "a question, like 'can you make...'", "bossy and specific",
          "describing it to a friend", "very brief, few words", "rambling and informal"]
PLACE = {"forest": "a forest", "dungeon": "a dungeon", "space": "space", "desert": "a desert", "snow": "a snowy place", "ocean": "the ocean or a beach", "volcano": "a volcano"}
GOAL = {"coins": "collect coins", "gems": "collect gems", "exit": "find the exit", "survive": "survive for a while"}
TRIG = {"collect": "whenever the player collects something", "hit": "whenever the player gets hit", "every": "every {tn} seconds",
        "score": "after the player collects {tn} things", "powerup": "whenever the player grabs a power-up"}
ACT = {"spawn": "{an} more enemies appear", "faster": "the enemies get faster", "slower": "the enemies get slower", "boost": "the player gets a speed boost",
       "heal": "the player heals", "teleport": "the player teleports somewhere random", "open_exit": "the exit unlocks", "freeze": "the enemies freeze"}


def spec(label):
    """The game in plain words, only the parts that are set."""
    s = []
    if label["theme"] != AUTO: s.append(f"place: {PLACE[label['theme']]}")
    if label["player"] != AUTO: s.append(f"the player is a {label['player']}")
    if label["enemy"] != AUTO: s.append("no enemies at all (peaceful)" if label["enemy"] == "none" else f"enemies: {label['enemy']}")
    if label["count"] != AUTO: s.append({"few": "only a few enemies", "some": "a medium number of enemies", "many": "lots of enemies"}[label["count"]])
    if label["speed"] != AUTO: s.append(f"the player moves {label['speed']}")
    if label["goal"] != AUTO: s.append(f"goal: {GOAL[label['goal']]}")
    if label["difficulty"] != AUTO: s.append(f"difficulty: {label['difficulty']}")
    if label["size"] != AUTO: s.append(f"map size: {label['size']}")
    if label["powerup"] != AUTO: s.append("no power-ups" if label["powerup"] == "none" else f"power-ups: {label['powerup']}")
    if label.get("behavior", AUTO) != AUTO: s.append({"patrol": "the enemies patrol back and forth", "shoot": "the enemies shoot at you", "wander": "the enemies wander around instead of chasing"}[label["behavior"]])
    if label.get("doors", AUTO) != AUTO: s.append({"keys": "locked doors you open with keys", "switches": "gates you open with switches"}[label["doors"]])
    if label.get("levels", AUTO) != AUTO: s.append(f"{label['levels']} levels")
    if label.get("boss", AUTO) != AUTO: s.append("a boss on the last level")
    if label.get("timer", AUTO) != AUTO: s.append({"short": "a time limit of about a minute", "long": "a time limit of about two minutes"}[label["timer"]])
    for r in rules_of(label):
        s.append("rule: " + TRIG[r["trig"]].format(**r) + ", " + ACT[r["act"]].format(**r).replace("1 more enemies", "1 more enemy"))
    return s


def write_prompt(sp, style):
    return [{"role": "user", "content":
             "A player is asking a game-making AI for a top-down 2D game. Write exactly what the player types.\n"
             f"Style: {style}.\nThe game must have exactly these things:\n- " + "\n- ".join(sp) +
             "\nMention every one of them, add no other game details, and don't copy the wording above. "
             "Reply with only the player's message, one or two sentences, no quotes."}]


def check_prompt(text):
    """Short answer on purpose: only the settings the request mentions, one per line (about 5x less to write)."""
    vals = {f: [v for v in FIELDS[f] if v != AUTO] for f in FIELD_NAMES}
    return [{"role": "user", "content":
             "Read this request for a top-down game. List ONLY the settings it actually mentions, one per line as key=value.\n"
             f"Request: {text}\n\nKeys and allowed values:\n" +
             "\n".join(f"{k}: {'/'.join(v)}" for k, v in vals.items()) +
             "\n(count = how many enemies; size = map size; powerup=none only if it says no power-ups; enemy=none only if it says no enemies)"
             "\nRules (at most 2), one per line as: rule=<when> <number or none> <what> <number or none>"
             "\n  when: collect/hit/every/score/powerup (every needs seconds 5/10/20, score needs 3/5/10)"
             "\n  what: spawn/faster/slower/boost/heal/teleport/open_exit/freeze (spawn needs 1 or 3)"
             "\n  example: rule=every 10 spawn 3"
             "\nWrite nothing else. If it mentions nothing, write: none"}]


def parse_check(text):
    out = {f: AUTO for f in FIELD_NAMES}
    for i in (1, 2):
        for k in ("trig", "tn", "act", "an"): out[f"r{i}_{k}"] = "none"
    n_rules = 0
    for line in text.lower().splitlines():
        if "=" not in line: continue
        k, v = (x.strip().strip("-*` ") for x in line.split("=", 1))
        if k == "rule":
            parts = v.split()
            if len(parts) != 4 or n_rules == 2: return None
            n_rules += 1
            for key, val in zip(("trig", "tn", "act", "an"), parts): out[f"r{n_rules}_{key}"] = val
        elif k in out:
            out[k] = v
        else:
            return None  # an unknown key: the teacher didn't follow the format, don't trust it
    return out


LEARN_DIR = os.path.join("models", "teachers", "learning")
PARALLEL = 8   # requests the llama.cpp teacher answers at once


def load_teacher_gguf(log=print, parallel=PARALLEL):
    """the teacher through llama.cpp (Optimise phase: about 2.7 GB compressed instead of an ~8 GB download, and it
    answers several requests at once). -> (name, None, running server) or None if it can't be set up"""
    try:
        from .coding.teacher import CANDIDATES as CC, fetch
        from .phase0 import runtime as R
        if not R.exe("llama-server"): R.install(log)
        c = next(x for x in CC if x["key"] == "qwen3.5-4b")
        gguf = fetch(c, log, folder=LEARN_DIR)
        srv = R.Server(gguf, ctx=1024 * parallel, port=8793, parallel=parallel).__enter__()  # ~1k tokens per request is plenty here
        return "Qwen/Qwen3.5-4B (llama.cpp)", None, srv
    except Exception as e:
        log(f"  the llama.cpp teacher couldn't be set up ({str(e)[:200]}); using the standard one")
        return None


def load_teacher(name, parallel=PARALLEL):
    if not name or name == "Qwen/Qwen3.5-4B":  # the compressed copy of the same model
        got = load_teacher_gguf(lambda m: print(m, flush=True), parallel)
        if got: return got
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tries = [name] if name else CANDIDATES
    for repo in tries:
        try:
            print(f"Loading teacher {repo} (first time: downloading, about 3-6 GB)...", flush=True)
            tok = AutoTokenizer.from_pretrained(repo)
            tok.padding_side = "left"
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=q, device_map="cuda", dtype=torch.bfloat16)
            return repo, tok, model.eval()
        except Exception as e:
            print(f"  could not use {repo}: {str(e)[:200]}", flush=True)
            import gc; gc.collect(); torch.cuda.empty_cache()
    sys.exit("No teacher could be loaded. Send Claude the lines above.")


def chat(tok, model, conversations, max_new, sample=True):
    if hasattr(model, "chat"):  # the llama.cpp teacher: several at once
        from concurrent.futures import ThreadPoolExecutor
        def one(c):
            try: return model.chat(c, max_tokens=max_new, thinking=False, temperature=0.9 if sample else 0.0)[0]
            except Exception: return ""
        with ThreadPoolExecutor(8) as ex: res = list(ex.map(one, conversations))
        return [re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip() for r in res]
    import torch
    texts = []
    for c in conversations:
        try:
            texts.append(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            texts.append(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True))
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        kw = dict(do_sample=True, temperature=0.9, top_p=0.95) if sample else dict(do_sample=False)
        out = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
    res = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip() for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="examples to keep")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        sys.exit("The teacher needs your NVIDIA GPU.")
    repo, tok, model = load_teacher(a.model, parallel=8 if a.batch >= 32 else 4)  # fewer at once beside training
    os.makedirs("data", exist_ok=True)
    rng = random.Random(time.time())
    kept = tried = 0
    have = sum(1 for _ in open("data/teacher.jsonl")) if os.path.exists("data/teacher.jsonl") else 0
    t0 = time.time()
    print(f"=== Teacher ({repo}) writing natural examples ===", flush=True)
    with open("data/teacher.jsonl", "a") as ftrain, open("data/test_teacher.jsonl", "a") as ftest:
        from . import polite
        polite.lower_priority(); gate = polite.Gate(log=lambda m: print(m, flush=True))
        while kept < a.n:
            labels = []
            while len(labels) < a.batch:
                lab = make_example(rng)["recipe"]
                if spec(lab) and is_valid_out(lab) and not (lab["enemy"] == "none" and lab["goal"] == "survive"):
                    labels.append(lab)
            styles = [rng.choice(STYLES) for _ in labels]
            gate.after_step()  # slows down while you're using the PC (polite.py)
            prompts = chat(tok, model, [write_prompt(spec(l), s) for l, s in zip(labels, styles)], 70)
            checks = chat(tok, model, [check_prompt(p) for p in prompts], 70, sample=False)
            for lab, st, p, c in zip(labels, styles, prompts, checks):
                tried += 1
                got = parse_check(c)
                p = " ".join(p.strip().strip('"').split())
                if got != lab or not (8 <= len(p) <= 240):
                    continue  # the teacher misunderstood its own request: not trustworthy, throw it away
                row = {"prompt": p, "recipe": lab, "source": repo, "style": st}
                (ftest if (have + kept) % 10 == 9 else ftrain).write(json.dumps(row) + "\n")
                kept += 1
            ftrain.flush(); ftest.flush()
            rate = kept / max(1, time.time() - t0) * 60
            print("@@P" + json.dumps({"step": kept, "total": a.n, "loss": round(kept / max(1, tried), 3), "sec": round(time.time() - t0)}), flush=True)
            print(f"kept {kept}/{a.n}  ({kept / max(1, tried):.0%} passed the self-check, {rate:.0f} per minute)  e.g. \"{prompts[0][:90]}\"", flush=True)
    print(f"Done: {kept} natural examples saved. They are used automatically the next time you /train or /learn.")
    if hasattr(model, "__exit__"): model.__exit__(None, None, None)


if __name__ == "__main__":
    main()
