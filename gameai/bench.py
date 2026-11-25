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
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--fresh", type=int, default=3000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, ck = load(a.checkpoint, device)
    seed = int(time.time())
    frozen, _ = score(model, tok, load_jsonl("data/test_frozen.jsonl"), device, 1)
    typo, _ = score(model, tok, load_jsonl("data/test_typo.jsonl"), device, 2)
    fresh, fails = score(model, tok, gen_rows(a.fresh, seed, "none"), device, 3)
    natural = None
    if os.path.exists("data/test_teacher.jsonl"):  # "real talk": how people actually phrase things (frozen: first 500)
        rows_t = load_jsonl("data/test_teacher.jsonl")[:500]
        if len(rows_t) >= 50:
            natural, _ = score(model, tok, rows_t, device, 4)
    gate = fresh["validity"] == 1.0 and fresh["instruction_match"] >= 0.99

    cur_path = "models/current.json"
    cur = json.load(open(cur_path)) if os.path.exists(cur_path) else None
    better = cur is None or (frozen["exact"] > cur["frozen"]["exact"] and frozen["validity"] >= cur["frozen"]["validity"])
    promoted = gate and better
    result = {"checkpoint": a.checkpoint, "train_run": ck.get("run"), "fresh_seed": seed,
              "frozen": frozen, "typo": typo, "fresh": fresh, "natural": natural, "gate_passed": gate, "promoted": promoted}
    if promoted:
        os.makedirs("models", exist_ok=True)
        shutil.copy(a.checkpoint, "models/current.pt")
        json.dump(result, open(cur_path, "w"), indent=2)
    os.makedirs("bench", exist_ok=True)
    with open("bench/history.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    json.dump(fails, open("bench/last_failures.json", "w"), indent=2)
    run = Run("bench", {"checkpoint": a.checkpoint}, kind="bench")
    run.finish(exact=frozen["exact"], typo_exact=typo["exact"], instruction_match=fresh["instruction_match"],
               field_acc=frozen["field_acc"], promoted=promoted)

    pct = lambda v: f"{v:.2%}"
    print(f"{'':12s}{'validity':>10s}{'fields':>10s}{'exact':>10s}{'instr':>10s}{'ms':>8s}")
    for name, m in (("frozen", frozen), ("typo", typo), ("fresh", fresh)) + ((("real talk", natural),) if natural else ()):
        print(f"{name:12s}{pct(m['validity']):>10s}{pct(m['field_acc']):>10s}{pct(m['exact']):>10s}{pct(m['instruction_match']):>10s}{m['ms_per_prompt']:>8}")
    print("Phase 1 gate:", "PASSED" if gate else "NOT passed", "| promoted:", promoted)
    print("wrong answers saved to bench/last_failures.json")


if __name__ == "__main__":
    main()
