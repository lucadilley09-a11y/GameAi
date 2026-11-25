"""Benchmark v1 (doc section 13) and promotion.

    python -m gameai.bench                         # tests checkpoints/latest.pt
    python -m gameai.bench --checkpoint path.pt

Measures, on frozen tests plus a FRESH set generated for this run:
  validity           output follows the grammar AND the engine builds a level that passes exact checks
  instruction_match  share of requested features actually present in the BUILT level
  field_acc / exact  recipe correctness against the labels
  typo_exact         exact match on the frozen one-typo set
  ms_per_prompt      speed
Phase 1 gate: validity == 100% and instruction_match >= 99% on the fresh set.
A model is promoted to models/current.pt only if it passes the gate and beats the
current model on frozen exact match without losing validity.
"""
import argparse, json, os, random, shutil, time
import torch
from .recipe import FIELD_NAMES, OUT_FIELDS, is_valid_out, resolve, rules_of
from .generate import decode, load
from .datagen import generate as gen_rows
from . import engine
from .track import Run


def score(model, tok, rows, device, seed):
    t0 = time.time()
    preds = decode(model, tok, [r["prompt"] for r in rows], device)
    ms = (time.time() - t0) * 1000 / len(rows)
    rng = random.Random(seed)
    valid = fields = exact = im_ok = im_total = 0
    failures = []
    for i, (p, r) in enumerate(zip(preds, rows)):
        rec = {f: p[f][0] for f in OUT_FIELDS}
        hits = [rec[f] == r["recipe"][f] for f in OUT_FIELDS]
        fields += sum(hits); exact += all(hits)
        good = is_valid_out(rec)
        if good:
            res = resolve({f: rec[f] for f in FIELD_NAMES}, rng); res["rules"] = rules_of(rec)
            lvl = engine.build(res, seed=i)
            ok, probs = engine.check(lvl)
            good = ok
            if not ok and len(failures) < 15:
                failures.append({"prompt": r["prompt"], "bug_report": probs})
            a, b = engine.instruction_match(r["recipe"], lvl)
            im_ok += a; im_total += b
        valid += good
        if not all(hits) and len(failures) < 15:
            failures.append({"prompt": r["prompt"], "want": r["recipe"], "got": rec})
    n = len(rows)
    return {"validity": valid / n, "field_acc": fields / (n * len(OUT_FIELDS)), "exact": exact / n,
            "instruction_match": im_ok / max(1, im_total), "ms_per_prompt": round(ms, 3)}, failures


def load_jsonl(p):
    """Examples saved before a part of the game language existed count as not mentioning it."""
    from .recipe import fill_defaults
    if not os.path.exists(p): return []
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    for r in rows:
        if "recipe" in r: fill_defaults(r["recipe"])
    return rows


GOLD = os.path.join(os.path.dirname(__file__), "gold.jsonl")
WEIGHT = {"frozen": 1, "typo": 1, "natural": 1, "gold": 1, "yours": 2}  # your own corrections count double


def test_sets():
    """Every test the AI must pass. The same sets are used for the new and the current version."""
    sets = {"frozen": load_jsonl("data/test_frozen.jsonl"), "typo": load_jsonl("data/test_typo.jsonl"), "gold": load_jsonl(GOLD)}
    if os.path.exists("data/test_teacher.jsonl"):  # "real talk": how people actually phrase things (frozen: first 500)
        t = load_jsonl("data/test_teacher.jsonl")[:500]
        if len(t) >= 50: sets["natural"] = t
    if os.path.exists("memory/corrections_test.jsonl"):  # requests you corrected, kept aside for testing
        y = load_jsonl("memory/corrections_test.jsonl")[-500:]
        if len(y) >= 10: sets["yours"] = y
    return sets


def total(res):
    w = {k: WEIGHT[k] for k in res if k in WEIGHT}
    return sum(res[k]["exact"] * w[k] for k in w) / sum(w.values())


def worse(new, old):
    """Sets where the new version is clearly worse: more than 1 point, or more than 1 example on small sets."""
    out = []
    for k in old:
        if k in new and k in WEIGHT:
            allow = max(0.01, 1.0 / max(1, new[k]["n"]))
            if new[k]["exact"] < old[k]["exact"] - allow: out.append(k)
    return out


def archive(result):
    """Keep every version that was ever in use, so you can go back (/versions, /rollback)."""
    os.makedirs("models/versions", exist_ok=True)
    have = sorted(f for f in os.listdir("models/versions") if f.endswith(".pt"))
    n = int(have[-1][1:4]) + 1 if have else 1
    shutil.copy("models/current.pt", f"models/versions/v{n:03d}.pt")
    json.dump(dict(result, version=n, time=time.time()), open(f"models/versions/v{n:03d}.json", "w"), indent=1)
    for old in have[:-19]:  # keep the last 20
        for ext in (".pt", ".json"):
            f = "models/versions/" + old[:-3] + ext
            if os.path.exists(f): os.remove(f)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--fresh", type=int, default=3000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, ck = load(a.checkpoint, device)
    seed = int(time.time())
    sets = test_sets()
    res = {}
    for i, (k, rows) in enumerate(sets.items()):
        res[k], _ = score(model, tok, rows, device, i + 1); res[k]["n"] = len(rows)
    fresh, fails = score(model, tok, gen_rows(a.fresh, seed, "none"), device, 99)
    gate = fresh["validity"] == 1.0 and fresh["instruction_match"] >= 0.99

    cur_path = "models/current.json"
    old, why, old_params = None, "", None
    if os.path.exists("models/current.pt"):
        try:
            cm, ctok, _ = load("models/current.pt", device)
            old_params = cm.num_params()
            old = {}
            for i, (k, rows) in enumerate(sets.items()):
                old[k], _ = score(cm, ctok, rows, device, i + 1); old[k]["n"] = len(rows)
        except ValueError:
            old = None  # the current AI is from an older game language: any passing new one replaces it
    if old is None:
        better, why = True, "no current AI to beat"
    else:
        drops, gain = worse(res, old), total(res) - total(old)
        better = gain > 0.0005 and not drops
        why = f"overall {gain*100:+.2f} points" + (f", but worse on: {', '.join(drops)}" if drops else "")
    promoted = gate and better
    result = {"checkpoint": a.checkpoint, "train_run": ck.get("run"), "fresh_seed": seed, **res,
              "fresh": fresh, "total": total(res), "params": model.num_params(), "old": old, "old_params": old_params, "old_total": total(old) if old else None, "why": why,
              "gate_passed": gate, "promoted": promoted}
    if promoted:
        os.makedirs("models", exist_ok=True)
        if old is not None and not os.path.isdir("models/versions"):
            archive({**old, "total": total(old), "note": "the AI before versions were kept"})
        shutil.copy(a.checkpoint, "models/current.pt")
        result["version"] = archive(result)
        json.dump(result, open(cur_path, "w"), indent=2)
    os.makedirs("bench", exist_ok=True)
    with open("bench/history.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    json.dump(fails, open("bench/last_failures.json", "w"), indent=2)
    run = Run("bench", {"checkpoint": a.checkpoint}, kind="bench")
    run.finish(exact=res["frozen"]["exact"], typo_exact=res["typo"]["exact"], instruction_match=fresh["instruction_match"],
               field_acc=res["frozen"]["field_acc"], promoted=promoted)

    pct = lambda v: f"{v:.2%}"
    names = {"frozen": "frozen", "typo": "typo", "gold": "gold", "natural": "real talk", "yours": "yours"}
    print(f"{'':12s}{'validity':>10s}{'fields':>10s}{'exact':>10s}{'instr':>10s}{'ms':>8s}{'in use':>10s}")
    for k, m in list(res.items()) + [("fresh", fresh)]:
        was = pct(old[k]["exact"]) if old and k in old else ""
        print(f"{names.get(k, k):12s}{pct(m['validity']):>10s}{pct(m['field_acc']):>10s}{pct(m['exact']):>10s}{pct(m['instruction_match']):>10s}{m['ms_per_prompt']:>8}{was:>10s}")
    print(f"Overall score {total(res):.2%} ({why})")
    print("Phase 1 gate:", "PASSED" if gate else "NOT passed", "| promoted:", promoted)
    print("wrong answers saved to bench/last_failures.json")


if __name__ == "__main__":
    main()
