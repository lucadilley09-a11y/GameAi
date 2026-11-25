"""The self-improving loop's lesson builder (runs at the start of every learning round).

    python -m gameai.improve            (the learning loop runs this for you)

1. Practice test: the current AI answers ~25,000 requests it has never seen (new practice requests,
   plus the teacher's real-person ones). The right answers are known, so every mistake is found.
2. Drill: each mistake becomes a lesson, plus new wordings of exactly the same game, so the AI
   practises the kinds of requests it gets wrong instead of ones it already knows.
3. Your corrections (/fix and answers to "I'm not sure" questions) count 5x; games you rated 4-5
   without corrections count 2x. Every 5th correction is kept aside as a test instead.
Writes data/extra.jsonl, which training adds to its practice data.
Nothing here can make the AI worse by itself: a new version is only used if it passes every test
(see bench.py: it must score better overall and not get worse on any test set).
"""
import argparse, json, os, random, time
from collections import Counter
import torch
from .datagen import generate as gen_rows, make_example
from .generate import load, decode
from .recipe import OUT_FIELDS, is_valid_out

MAX_LESSONS = 15000


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/current.pt")
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="data/extra.jsonl")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else int(time.time())
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load(a.model, device)

    print("=== Practice test: finding what it gets wrong ===", flush=True)
    pool = gen_rows(a.pool, seed, "mixed")
    teacher = load_jsonl("data/teacher.jsonl")
    pool += rng.sample(teacher, min(5000, len(teacher)))
    t0 = time.time()
    preds = decode(model, tok, [r["prompt"] for r in pool], device, gate=0.0)
    wrong, by_field = [], Counter()
    for r, p in zip(pool, preds):
        miss = [f for f in OUT_FIELDS if p[f][0] != r["recipe"][f]]
        if miss:
            wrong.append(r); by_field.update({("rules" if f[0] == "r" and f[1].isdigit() else f) for f in miss})
    rate = 1 - len(wrong) / max(1, len(pool))

    lessons = []
    for r in wrong:  # the mistake itself twice, plus 3 new ways of asking for the same game
        lessons += [r, r]
        for k in range(3):
            lessons.append(make_example(rng, rng.choice([0, 0, 1]), want=r["recipe"]))
    corr = [r for r in load_jsonl("memory/corrections.jsonl") if is_valid_out(r["recipe"])]
    conf = [r for r in load_jsonl("memory/confirmed.jsonl") if is_valid_out(r["recipe"])]
    rng.shuffle(lessons)
    lessons = lessons[:MAX_LESSONS] + corr * 5 + conf * 2
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in lessons:
            f.write(json.dumps({"prompt": r["prompt"], "recipe": r["recipe"]}) + "\n")
    top = ", ".join(f"{k} ({v})" for k, v in by_field.most_common(3)) or "nothing"
    info = {"practice_score": round(rate, 4), "mistakes": len(wrong), "tested": len(pool), "lessons": len(lessons),
            "corrections": len(corr), "confirmed": len(conf), "weakest": [k for k, _ in by_field.most_common(3)]}
    print("@@I" + json.dumps(info), flush=True)
    os.makedirs("learn", exist_ok=True); json.dump(info, open("learn/practice.json", "w"))
    print(f"Practice test: {rate:.2%} right on {len(pool)} new requests ({time.time() - t0:.0f} s). "
          f"Most mistakes in: {top}. Made {len(lessons)} lessons ({len(corr)} of your corrections).", flush=True)


if __name__ == "__main__":
    main()
