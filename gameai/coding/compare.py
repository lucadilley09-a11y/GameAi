"""Repair loop vs plain retries, at the same cost (plan: Phase 4; research: for small code models, budget-matched blind
resampling often beats self-repair, so repair is used only where it measurably wins).

For each validation task, both strategies get the same BUDGET of model answers:
    resample  up to BUDGET fresh attempts; solved as soon as one passes everything
    repair    BUDGET // 2 fresh attempts; each failure gets a repair (its code + what the visible tests said), one answer
              each, until the budget is used up
The winner per language goes in data/coding/strategy.json; a tie goes to resampling (simpler, cheaper per answer).
"""
import json, os, time

from . import tasks as T
from .selfteach import attempt

BUDGET = 8
OUT = os.path.join(T.DIR, "strategy.json")


def resample(chat, task):
    for i in range(BUDGET):
        if attempt(chat, task, 1, temperature=0.8)[0]["accepted"]: return True, i + 1
    return False, BUDGET


def repair(chat, task):
    used, failures = 0, []
    for _ in range(BUDGET // 2):
        r = attempt(chat, task, 1, temperature=0.8)[0]; used += 1
        if r["accepted"]: return True, used
        if r["report"]: failures.append(r)
    while used < BUDGET and failures:
        bad = failures.pop(0)
        f = attempt(chat, task, 1, temperature=0.3, code=bad["code"], report=bad["report"])[0]; used += 1
        if f["accepted"]: return True, used
        if f["report"]: failures.append(f)  # a failed repair can be repaired again
    return False, used


def run(chat, log=print, task_list=None):
    task_list = task_list or T.load("val")
    res, t0 = {}, time.time()
    for i, task in enumerate(task_list):
        a, _ = resample(chat, task); b, _ = repair(chat, task)
        r = res.setdefault(task["lang"], {"tasks": 0, "resample": 0, "repair": 0})
        r["tasks"] += 1; r["resample"] += a; r["repair"] += b
        if i % 5 == 0: log(f"  {i + 1}/{len(task_list)} tasks ({time.time() - t0:.0f}s)")
    out = {"budget": BUDGET, "measured": time.strftime("%Y-%m-%d %H:%M"), "by_language": res,
           "use": {lang: ("repair" if r["repair"] > r["resample"] else "resample") for lang, r in res.items()}}
    os.makedirs(T.DIR, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for lang, r in res.items():
        log(f"{lang}: plain retries solved {r['resample']}/{r['tasks']}, repair solved {r['repair']}/{r['tasks']} -> using {out['use'][lang]}")
    return out


def strategy(lang):
    """'repair' or 'resample' for a language (resample until measured)"""
    try: return json.load(open(OUT))["use"].get(lang, "resample")
    except (OSError, ValueError): return "resample"
