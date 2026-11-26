"""When to ask a clarifying question (Optimise change 9, approved by the user 2026-11-25; training data for Phase 3).

Research: models usually answer ambiguous requests by assuming one meaning; training on SIMULATED NEXT TURNS (what the
user would actually say back) teaches them to ask when the answer would differ and to answer directly otherwise.

Here a simulated user makes a request that leaves some parts out. For each part it left out, the user either has a
wish (with the probability that you care about that part, learned from your answers in the app: "Surprise me" means
you don't) or doesn't mind. Both choices are weighed by their expected outcome:
    answer now        the AI picks the missing parts itself; each part the user had a wish for is right only by luck
    ask first         one question with choices; the user's reply (their wish, or "you pick") fills that part in
asking is better when the expected gain (how often the user cares x how likely the AI's pick misses) is above
ASK_COST per question. The simulated user then really replies from its wishes. That gives
    data/clarify/sft.jsonl     whole conversations doing the better thing (ask, hear the reply, then answer; or answer)
    data/clarify/pairs.jsonl   the better first reply vs the worse one, for preference training (DPO)
The app uses the same rule today for its own questions (web_template.html: askWorth).

    python -m gameai.clarify --n 20000
"""
import argparse, json, os, random

from .recipe import FIELDS, AUTO

DIR = os.path.join("data", "clarify")
ASK_COST = 0.15
MAX_QUESTIONS = 2
CARE_PRIOR = {"goal": 0.5, "enemy": 0.4, "theme": 0.35, "player": 0.35}
QUESTION = {"goal": "What's the goal?", "enemy": "What should chase you?", "theme": "Where does it take place?", "player": "Who do you play as?"}
SYSTEM = ("You turn a player's request into a game. If the request leaves out something they probably care about and "
          "your guess would likely be wrong, ask one short question with choices first; otherwise make the game.")


def care(log=os.path.join("memory", "questions.jsonl"), k=4):
    """P(you care about a part you didn't mention), from your answers in the app (smoothed towards the priors)"""
    stats = {f: [0, 0] for f in QUESTION}
    if os.path.exists(log):
        for l in open(log, encoding="utf-8"):
            try: r = json.loads(l)
            except ValueError: continue
            if r.get("field") in stats: stats[r["field"]][0] += 1; stats[r["field"]][1] += bool(r.get("cared"))
    return {f: (c + CARE_PRIOR[f] * k) / (a + k) for f, (a, c) in stats.items()}


def worth(field, p_care, p_right=None):
    """the expected gain from asking about one part: the chance a wish exists times the chance the AI's pick misses it"""
    n = len([v for v in FIELDS[field] if v != AUTO]) or 1
    return p_care * (1 - (p_right if p_right is not None else 1 / n))


def choose_questions(recipe, cares):
    """which parts to ask about (best first), using only what the request itself left out"""
    cand = [(worth(f, cares[f]), f) for f in QUESTION if recipe.get(f) == AUTO]
    return [f for w, f in sorted(cand, reverse=True) if w > ASK_COST][:MAX_QUESTIONS]


def simulate(rng, cares):
    """one simulated request -> the conversation doing the better thing, and the pair (better first reply, worse one)"""
    from .datagen import make_example
    from .recipe import to_text
    ex = make_example(rng)
    rec = ex["recipe"]
    wishes = {f: rng.choice([v for v in FIELDS[f] if v != AUTO]) for f in QUESTION if rec.get(f) == AUTO and rng.random() < cares[f]}
    asks = choose_questions(rec, cares)
    # the label is the EXPECTED better choice (the user's wishes can't be seen in the request, so deciding from them
    # would teach the AI to guess at random); the simulated replies then come from this user's actual wishes
    ask_better = bool(asks)
    final = dict(rec)
    for f in asks if ask_better else []:
        if f in wishes: final[f] = wishes[f]
    user = {"role": "user", "content": ex["prompt"]}
    answer = {"role": "assistant", "content": to_text(final)}
    def question(f):
        opts = [v for v in FIELDS[f] if v != AUTO]
        return {"role": "assistant", "content": QUESTION[f] + " " + " / ".join(opts) + " / surprise me"}
    if ask_better:
        conv = [{"role": "system", "content": SYSTEM}, user]
        for f in asks:
            conv += [question(f), {"role": "user", "content": wishes.get(f, "surprise me")}]
        conv.append(answer)
    else:
        conv = [{"role": "system", "content": SYSTEM}, user, {"role": "assistant", "content": to_text(rec)}]
    pair = None
    if asks:  # only when asking was a real option is there a choice to learn
        ask_first, answer_first = question(asks[0])["content"], to_text(rec)
        pair = {"prompt": [{"role": "system", "content": SYSTEM}, user],
                "chosen": ask_first if ask_better else answer_first, "rejected": answer_first if ask_better else ask_first,
                "asked": ask_better, "wishes": sorted(wishes)}
    return {"messages": conv, "asked": ask_better, "prompt": ex["prompt"]}, pair


def generate(n, seed=0, cares=None):
    from . import guard
    cares = cares or care()
    rng, ix = random.Random(seed), guard.build_index()
    os.makedirs(DIR, exist_ok=True)
    kept = asked = 0
    with open(os.path.join(DIR, "sft.jsonl"), "w", encoding="utf-8") as fs, open(os.path.join(DIR, "pairs.jsonl"), "w", encoding="utf-8") as fp:
        while kept < n:
            row, pair = simulate(rng, cares)
            if ix.match(row["prompt"])[0]: continue  # never a twin of a test request
            fs.write(json.dumps(row) + "\n"); kept += 1; asked += row["asked"]
            if pair: fp.write(json.dumps(pair) + "\n")
    return {"conversations": kept, "asked_first": round(asked / max(1, kept), 3), "care": {k: round(v, 3) for k, v in cares.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(generate(a.n, a.seed), indent=1))
