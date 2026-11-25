"""Verified self-teaching for code (plan: Phase 4): the model tries each training task many times, the sandbox keeps
only the solutions that pass everything (verify.py), and those become training data. Three kinds, trained separately:

    data/coding/first_pass.jsonl   task -> a verified solution                 (first-attempt skill)
    data/coding/repair.jsonl       task + failed code + what went wrong -> a verified fix   (repair skill)
    data/coding/pairs.jsonl        same task: a verified solution vs a failed one (for preference training, DPO)
Unverified code is never written to any of them.

Guards (plan: guarding self-teaching against checker-gaming): at most CAP accepted solutions per task; near-identical
solutions are skipped once a task has two (novelty is a soft limit: a good solution isn't refused just for looking like
another); hidden tests and the cheat checks are in verify.py.

The same loop runs with the coding teacher (source="teacher") to fill gaps the AI can't solve yet.
"""
import hashlib, json, os, time

from . import prompts
from . import tasks as T
from .prompts import text_of_reply as _text
from .verify import verify, similarity

DIR = T.DIR
FILES = {"first": os.path.join(DIR, "first_pass.jsonl"), "repair": os.path.join(DIR, "repair.jsonl"), "pairs": os.path.join(DIR, "pairs.jsonl")}
CAP = 4              # accepted solutions kept per task
SAME = 0.95          # similarity at which two solutions count as the same
REPAIRS_PER_TASK = 3 # failed attempts that get a repair try


def _seen():
    out = set()
    for p in FILES.values():
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                try: out.add(json.loads(l)["hash"])
                except (ValueError, KeyError): pass
    return out


def _write(kind, row):
    os.makedirs(DIR, exist_ok=True)
    with open(FILES[kind], "a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hash(*parts):
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def answer_of(task, code):
    return f"```{prompts.FENCE[task['lang']]}\n{code.rstrip()}\n```"


def attempt(chat, task, n, temperature=0.8, code=None, report=None):
    """n answers from the model -> [verify results]"""
    out = []
    for i in range(n):
        try: text = _text(chat(prompts.messages(task, code, report), max_tokens=1500, thinking=False, temperature=temperature if n > 1 else 0.0))
        except Exception as e:  # a model error is a failed attempt, not a crash
            text = f"(no answer: {e})"
        out.append(verify(task, text))
    return out


def teach_task(chat, task, n=16, source="self", seen=None, log=print):
    """-> counts for this task"""
    seen = _seen() if seen is None else seen
    first_msgs = prompts.messages(task)
    results = attempt(chat, task, n)
    kept = []
    for r in results:
        if not r["accepted"] or len(kept) >= CAP: continue
        if len(kept) >= 2 and any(similarity(task["lang"], r["code"], k["code"]) >= SAME for k in kept): continue
        kept.append(r)
    stats = {"task": task["id"], "lang": task["lang"], "tries": n, "passed": sum(r["accepted"] for r in results), "kept": len(kept),
             "repairs_tried": 0, "repairs_ok": 0, "pairs": 0}
    for r in kept:
        h = _hash("first", task["id"], r["code"])
        if h in seen: continue
        seen.add(h)
        _write("first", {"hash": h, "task": task["id"], "lang": task["lang"], "source": source, "task_prompt": task["prompt"],
                         "system": first_msgs[0]["content"], "prompt": first_msgs[1]["content"], "answer": answer_of(task, r["code"]),
                         "ms": r["ms"], "lines": r["lines"]})
    # repairs: failed attempts that ran (not cheats), each with the visible tests' report
    failed = [r for r in results if not r["accepted"] and r["report"] and not any(x.startswith(("not allowed", "looks like")) for x in r["reasons"])]
    uniq = []
    for r in failed:
        if all(similarity(task["lang"], r["code"], u["code"]) < SAME for u in uniq): uniq.append(r)
    for bad in uniq[:REPAIRS_PER_TASK]:
        stats["repairs_tried"] += 1
        fix = next((f for f in attempt(chat, task, 2, code=bad["code"], report=bad["report"]) if f["accepted"]), None)
        if not fix: continue
        stats["repairs_ok"] += 1
        msgs = prompts.messages(task, bad["code"], bad["report"])
        h = _hash("repair", task["id"], bad["code"], fix["code"])
        if h not in seen:
            seen.add(h)
            _write("repair", {"hash": h, "task": task["id"], "lang": task["lang"], "source": source, "task_prompt": task["prompt"],
                              "system": msgs[0]["content"], "prompt": msgs[1]["content"], "answer": answer_of(task, fix["code"])})
    # preference pairs: same prompt, same context: a verified solution against a failed one
    for good, bad in list(zip(kept, uniq))[:2]:
        h = _hash("pair", task["id"], good["code"], bad["code"])
        if h in seen: continue
        seen.add(h); stats["pairs"] += 1
        _write("pairs", {"hash": h, "task": task["id"], "lang": task["lang"], "source": source, "task_prompt": task["prompt"],
                         "system": first_msgs[0]["content"], "prompt": first_msgs[1]["content"],
                         "chosen": answer_of(task, good["code"]), "rejected": answer_of(task, bad["code"])})
    return stats


def run(chat, task_list, n=16, source="self", log=print, stop=lambda: False):
    """self-teaching over many tasks; -> summary by language"""
    from ..guard import Index
    ix = Index()
    for t in T.load("val"): ix.add(t["prompt"])
    seen, summary, t0 = _seen(), {}, time.time()
    for i, task in enumerate(task_list):
        if stop(): break
        if task.get("split") == "val" or ix.match(task["prompt"])[0]: continue  # validation tasks are never trained on
        s = teach_task(chat, task, n=n, source=source, seen=seen, log=log)
        agg = summary.setdefault(task["lang"], {"tasks": 0, "solved": 0, "kept": 0, "repairs_tried": 0, "repairs_ok": 0, "pairs": 0})
        agg["tasks"] += 1; agg["solved"] += s["kept"] > 0
        for k in ("kept", "repairs_tried", "repairs_ok", "pairs"): agg[k] += s[k]
        if i % 10 == 0: log(f"  {i + 1}/{len(task_list)} tasks, {time.time() - t0:.0f}s: " + ", ".join(f"{k} {v['solved']}/{v['tasks']} solved" for k, v in summary.items()))
    return summary


def rows(kind, lang=None):
    """training rows of one kind ('first' | 'repair' | 'pairs'), guarded against validation tasks"""
    from ..guard import Index
    if not os.path.exists(FILES[kind]): return []
    ix = Index()
    for t in T.load("val"): ix.add(t["prompt"])
    out = []
    for l in open(FILES[kind], encoding="utf-8"):
        r = json.loads(l)
        if lang and r["lang"] != lang: continue
        if ix.match(r["task_prompt"])[0]: continue
        out.append(r)
    return out
