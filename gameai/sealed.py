"""Sealed test sets: measure real ability, never used for training, tuning or feedback.

The tasks are written by someone other than Claude (you, ChatGPT, Gemini), because the developer can't be sealed
from what it writes. They are stored ENCRYPTED in sealed/. Only this evaluator opens them, in memory, and it only
ever reports TOTALS (one score per set), never which questions were right or wrong. Training never reads them: it
only gets their fingerprints (see guard.py), which can't be turned back into the text. The AI's own tools must never
read the sealed/ folder (PROTECTED below).

Each set keeps unseen spare tasks (the reserve). When you review some of a set's results, those tasks are retired
and replaced from the reserve, so the set stays unseen. Every score is kept with the set's version.

    python -m gameai.sealed add intent my_requests.jsonl   lock a file of tasks away (then delete your copy)
    python -m gameai.sealed run intent                      score the AI in use on that set (a total only)
    python -m gameai.sealed review intent 5                 show 5 tasks with the AI's answers, then retire them
    python -m gameai.sealed status                          sizes and past totals (no task text)

Task file format, one JSON object per line. For game requests:
    {"request": "a snowy level full of dragons and a boss at the end", "expect": {"theme": "snow", "enemy": "dragons", "boss": "yes"}}
"expect" lists only the parts the writer is sure about; only those are scored. {"request": ..., "ask": true} marks a
request so unclear that the right answer is a clarifying question.
"""
import json, os, random, sys, time

DIR = "sealed"
STORE, KEY, FPS, RESULTS = (os.path.join(DIR, f) for f in ("store.bin", "key", "fingerprints.npz", "results.jsonl"))
PROTECTED = [DIR]   # folders the AI's tools (sandboxes, file access, memory search) must never read
RESERVE_SHARE = 0.25
SCORERS = {}        # kind -> function(items, model_name) -> {"score": 0..1, ...totals only}


def _fernet():
    from cryptography.fernet import Fernet
    os.makedirs(DIR, exist_ok=True)
    if not os.path.exists(KEY):
        open(KEY, "wb").write(Fernet.generate_key())
    return Fernet(open(KEY, "rb").read())


def _open():
    if not os.path.exists(STORE): return {"sets": {}}
    return json.loads(_fernet().decrypt(open(STORE, "rb").read()))


def _save(store):
    data = _fernet().encrypt(json.dumps(store).encode())  # (also makes the folder on first use)
    tmp = STORE + ".new"
    open(tmp, "wb").write(data)
    os.replace(tmp, STORE)
    _write_fingerprints(store)


def _write_fingerprints(store):
    """Fingerprints of every sealed task, ever (active, spare and retired), for the training guard."""
    import numpy as np
    from .guard import exact_key, signature, PERMS
    keys, sigs = [], []
    for s in store["sets"].values():
        for it in s["active"] + s["reserve"] + s["retired"]:
            keys.append(exact_key(it["request"])); sigs.append(signature(it["request"]))
    np.savez_compressed(FPS, exact=np.array(keys), sigs=np.array(sigs, dtype=np.uint64).reshape(-1, PERMS))


def fingerprints():
    """-> [(exact_key, signature)] without any task text"""
    if not os.path.exists(FPS): return []
    import numpy as np
    d = np.load(FPS)
    return list(zip(map(str, d["exact"]), d["sigs"]))


def summary():
    """sizes only (safe to show anywhere)"""
    if not os.path.exists(STORE): return {}
    return {k: {"active": len(v["active"]), "reserve": len(v["reserve"]), "retired": len(v["retired"]), "version": v["version"]}
            for k, v in _open()["sets"].items()}


def add(set_name, path, kind=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        if not isinstance(r.get("request"), str) or not ("expect" in r or "ask" in r):
            raise ValueError("every line needs a \"request\" and either \"expect\" or \"ask\"")
    store = _open()
    s = store["sets"].setdefault(set_name, {"kind": kind or set_name, "version": 0, "active": [], "reserve": [], "retired": [], "added": []})
    random.Random(time.time_ns()).shuffle(rows)
    n_res = round(len(rows) * RESERVE_SHARE)
    s["reserve"] += rows[:n_res]; s["active"] += rows[n_res:]
    s["version"] += 1; s["added"].append({"time": time.time(), "rows": len(rows)})
    _save(store)
    return {"set": set_name, "added": len(rows), "active": len(s["active"]), "reserve": len(s["reserve"]), "version": s["version"]}


def run(set_name, model_name="current"):
    """Score one model on one sealed set. Only the total is returned and recorded."""
    store = _open()
    if set_name not in store["sets"]: raise KeyError(f"no sealed set called {set_name!r}")
    s = store["sets"][set_name]
    scorer = SCORERS.get(s["kind"])
    if scorer is None: raise KeyError(f"no scorer for {s['kind']!r} tasks yet")
    tot = scorer(s["active"], model_name)
    from .update import local_version
    row = {"set": set_name, "set_version": s["version"], "n": len(s["active"]), "model": model_name,
           "app_version": (local_version().get("sha") or "dev")[:7], "time": time.strftime("%Y-%m-%d %H:%M"), **tot}
    with open(RESULTS, "a") as f: f.write(json.dumps(row) + "\n")
    return row


def review(set_name, k, model_name="current", show=print):
    """Show k tasks and the AI's answers for a human to judge, then retire them and refill from the reserve."""
    store = _open(); s = store["sets"][set_name]
    k = min(k, len(s["active"]))
    picked = s["active"][:k]
    answers = SCORERS[s["kind"]](picked, model_name, details=True)["details"]
    for it, a in zip(picked, answers): show(json.dumps({"request": it["request"], "expected": it.get("expect") or "ask a question", "ai": a}))
    s["active"] = s["active"][k:]; s["retired"] += picked
    refill = s["reserve"][:k]; s["reserve"] = s["reserve"][k:]; s["active"] += refill
    s["version"] += 1
    _save(store)
    return {"retired": k, "refilled": len(refill), "reserve_left": len(s["reserve"]), "version": s["version"]}


# ---- scorers -------------------------------------------------------------------------------------------------

def _answers(requests, model_name):
    """The settings each model gives for each request: today's game AI ("current" or a .pt file), or a release
    ("release:<id>", run in llama.cpp exactly as it will be used)."""
    if not requests: return []
    if model_name.startswith("release:"):
        from . import release
        from .phase0 import tasks as T
        man = json.load(open(os.path.join(release.DIR, model_name[8:], "release.json")))
        with release._server_for(man, "game_design") as srv:
            return [T.parse_recipe(srv.chat([{"role": "system", "content": T.RECIPE_SYSTEM}, {"role": "user", "content": r}],
                                            max_tokens=200, grammar=T.recipe_grammar(), thinking=False)[0]) for r in requests]
    from .generate import load, decode
    from .recipe import OUT_FIELDS
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load("models/current.pt" if model_name == "current" else model_name, dev)
    return [{f: p[f][0] for f in OUT_FIELDS} for p in decode(model, tok, requests, dev)]


def score_game_requests(items, model_name, details=False):
    """Share of the expected fields the model got right; 'ask' tasks are reported apart until the orchestrator can
    ask questions (Phase 3)."""
    scored = [it for it in items if "expect" in it]
    preds = _answers([it["request"] for it in scored], model_name)
    hit = total = full = 0; out = []
    for it, rec in zip(scored, preds):
        h = sum(rec.get(f) == v for f, v in it["expect"].items())
        hit += h; total += len(it["expect"]); full += h == len(it["expect"])
        out.append({f: rec.get(f) for f in it["expect"]})
    res = {"score": round(hit / max(1, total), 4), "fully_right": round(full / max(1, len(scored)), 4),
           "scored": len(scored), "ask_tasks_not_scored_yet": len(items) - len(scored)}
    if details:
        it_out = iter(out)  # answers in the same order as the tasks
        res["details"] = [next(it_out) if "expect" in it else "(asks a question: not scorable yet)" for it in items]
    return res


SCORERS["game_requests"] = SCORERS["intent"] = score_game_requests


def main():
    a = sys.argv[1:]
    if not a or a[0] == "status":
        print(json.dumps(summary(), indent=1))
        if os.path.exists(RESULTS):
            for l in open(RESULTS): print(l.strip())
    elif a[0] == "add":
        print(add(a[1], a[2], a[3] if len(a) > 3 else None))
        print(f"Locked away. Now delete your own copy of {a[2]} (and don't paste it to Claude).")
    elif a[0] == "run":
        print(run(a[1], a[2] if len(a) > 2 else "current"))
    elif a[0] == "review":
        print(review(a[1], int(a[2]) if len(a) > 2 else 5))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
