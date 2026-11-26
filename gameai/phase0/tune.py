"""Answer-speed tuning (Optimise phase, approved by the user 2026-11-25): measures llama.cpp settings for one model on
this PC and keeps the fastest one that doesn't change the answers for the worse.

    python -m gameai.phase0.tune models/base/<key>/model-q4_k_m.gguf

Settings tried (research: Optimise session 1):
    plain             the starting point
    flash attention   less memory, usually faster
    + 8-bit KV cache  half the memory for the context, nearly lossless (4-bit is never tried: a known quality cliff);
                      some builds silently fall back to slow CPU attention with it, so it's kept only if measured faster
    + n-gram speculation   drafts tokens from text already seen (no extra model); lossless; best on code edits
The workload is the AI's real kinds of work: game requests (with the recipe grammar), code tasks and chat.

Quality guard: flash attention and n-gram speculation must give (nearly) the same greedy answers as plain; the 8-bit
cache must solve at least as many code tasks and get as many game requests fully right as plain. Every setting is
also timed twice, and a setting must be at least 5% faster to be adopted (smaller differences are noise).
The result goes in runtime/tuning.json, keyed by the model file; Server() uses it from then on.
"""
import json, os, sys, time

from . import runtime as R

CONFIGS = [
    ("plain", []),
    ("flash attention", ["-fa", "on"]),
    ("flash attention + 8-bit KV cache", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]),
    ("n-gram speculation", ["--spec-type", "ngram-mod", "--spec-ngram-size-n", "24", "--draft-min", "48", "--draft-max", "64"]),
    ("flash attention + n-gram speculation", ["-fa", "on", "--spec-type", "ngram-mod", "--spec-ngram-size-n", "24", "--draft-min", "48", "--draft-max", "64"]),
    ("flash attention + 8-bit KV cache + n-gram speculation", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "ngram-mod",
                                                                "--spec-ngram-size-n", "24", "--draft-min", "48", "--draft-max", "64"]),
]
MIN_GAIN = 0.05
SAME = 0.9        # share of greedy answers that must match plain for the lossless settings


def workload(n_recipes=20, n_code=8):
    from . import tasks as T
    from ..coding import tasks as CT, prompts as CP
    items = [("recipe", it) for it in T.recipe_items()[:n_recipes]]
    items += [("code", t) for t in CT.load("val")[:n_code]]
    items += [("chat", [{"role": "user", "content": q}]) for q in (
        "Explain in three sentences how a platformer's jump should feel.", "Give me five names for a spooky dungeon game.",
        "What's the difference between a sprite and a tile?", "Write a short, friendly welcome message for a game menu.")]
    return items


def run(srv, items):
    """-> (seconds, answers, scores)"""
    from . import tasks as T
    from ..coding import prompts as CP
    from ..coding.verify import verify
    t0, answers, code_ok, r_items, r_preds = time.time(), [], 0, [], []
    for kind, it in items:
        if kind == "recipe":
            text, _ = srv.chat([{"role": "system", "content": T.RECIPE_SYSTEM}, {"role": "user", "content": it["prompt"]}],
                               max_tokens=200, grammar=T.recipe_grammar(), thinking=False)
            answers.append(text); r_items.append(it); r_preds.append(T.parse_recipe(text))
        elif kind == "code":
            text, _ = srv.chat(CP.messages(it), max_tokens=700, thinking=False, temperature=0.0)
            answers.append(text); code_ok += verify(it, text)["accepted"]
        else:
            text, _ = srv.chat(it, max_tokens=200, thinking=False, temperature=0.0)
            answers.append(text)
    secs = time.time() - t0
    fully = T.score_recipes(r_items, r_preds).get("fully_right", 0) * len(r_items) if r_items else 0
    return secs, answers, {"recipes_fully_right": round(fully), "code_tasks_solved": code_ok}


def tune(gguf, log=print, port=8795):
    items = workload()
    results, base = [], None
    for name, flags in CONFIGS:
        try:
            with R.Server(gguf, port=port, extra=flags, tuned=False, cache_prompt=True) as srv:
                run(srv, items[:3])                                   # warm-up
                times = []
                for _ in range(2):
                    s, answers, scores = run(srv, items); times.append(s)
        except Exception as e:
            log(f"  {name}: didn't work here ({str(e)[:120]})"); results.append({"name": name, "error": str(e)[:300]}); continue
        r = {"name": name, "flags": flags, "seconds": round(min(times), 2), "scores": scores}
        if base is None: base = dict(r, answers=answers)
        else:
            same = sum(a == b for a, b in zip(answers, base["answers"])) / len(answers)
            r["same_answers"] = round(same, 3)
            lossless = "-ctk" not in flags
            r["quality_ok"] = (same >= SAME) if lossless else all(scores[k] >= base["scores"][k] for k in scores)
            r["speedup"] = round(base["seconds"] / r["seconds"], 3)
        results.append(r)
        log(f"  {name}: {r['seconds']}s" + (f" ({r['speedup']}x, {'answers ok' if r['quality_ok'] else 'answers changed: rejected'})" if "speedup" in r else " (starting point)"))
    ok = [r for r in results if r.get("quality_ok") and r.get("speedup", 0) >= 1 + MIN_GAIN]
    best = max(ok, key=lambda r: r["speedup"]) if ok else None
    t = {}
    try: t = json.load(open(R.TUNING))
    except (OSError, ValueError): pass
    t[R._model_id(gguf)] = {"flags": best["flags"] if best else [], "chosen": best["name"] if best else "plain",
                             "speedup": best["speedup"] if best else 1.0, "measured": time.strftime("%Y-%m-%d %H:%M"),
                             "results": [{k: v for k, v in r.items() if k != "answers"} for r in results]}
    os.makedirs(os.path.dirname(R.TUNING), exist_ok=True)
    json.dump(t, open(R.TUNING, "w"), indent=1)
    log(f"chosen: {best['name']} ({best['speedup']}x faster)" if best else "chosen: plain (nothing was clearly faster without changing answers)")
    return t[R._model_id(gguf)]


if __name__ == "__main__":
    tune(sys.argv[1])
