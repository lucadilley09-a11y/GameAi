"""Verified self-teaching for interfaces (plan: Phase 5): the AI builds each practice screen many times, verify.py keeps
only the ones that pass (right style, everything asked for on screen, GameUI parts, offline, clean layout at desktop and
phone size), and those become training data. Same guards as code: at most CAP kept per task, near-copies skipped.

    data/ui/first_pass.jsonl   request -> a verified screen
    data/ui/repair.jsonl       request + failed screen + what the checker said -> a verified fix
    data/ui/stats.jsonl        how many tries passed per task

Practice tasks come from the generator's training vocabulary; validation tasks (val_set) use separate words and phrasings
and are never trained on.
"""
import difflib, hashlib, json, os, random, re, time

from . import tasks as T, verify as V

DIR = os.path.join("data", "ui")
FILES = {"first": os.path.join(DIR, "first_pass.jsonl"), "repair": os.path.join(DIR, "repair.jsonl")}
STATS = os.path.join(DIR, "stats.jsonl")
CAP = 4
SAME = 0.95
REPAIRS_PER_TASK = 2

SYSTEM = (
    "You build game interface screens with the GameUI design system. Answer with one ```html block: a "
    '<div class="gui" data-theme="STYLE"> holding the screen, where STYLE is clean, pixel, neon or cozy (the closest to '
    "what was asked). Use only GameUI parts: gui-screen, gui-title, gui-subtitle, gui-menu, gui-btn (gui-primary, "
    "gui-danger), gui-panel, gui-row, gui-hud, gui-stat, gui-bar (a <progress>), gui-dialog, gui-speaker, gui-list, "
    "gui-item, gui-selected, gui-toggle (checkbox), gui-range (slider), gui-table, gui-grid, gui-slot, gui-toast, gui-muted. "
    "Real <button>, <input>, <table> elements. No scripts, no inline styles, no outside files. Every text the request names "
    "must appear exactly."
)


def messages(task, html=None, report=None):
    """one system + one user message (a repair puts the rejected screen and the checker's report in the same message, so
    training never learns from the rejected screen)"""
    user = task["prompt"]
    if html is not None:
        user += ("\n\nThis attempt was rejected:\n" + answer_of(html) + "\nWhat the checker said:\n" + report +
                 "\nFix it and send the whole screen again.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def answer_of(html):
    return f"```html\n{html.strip()}\n```"


def _text(reply):
    return reply[0] if isinstance(reply, tuple) else reply


def _hash(*parts):
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _norm(html):
    return re.sub(r"\s+", " ", html).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, _norm(a), _norm(b), autojunk=False).ratio()


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


_g = None
def _gate():
    global _g
    if _g is None:
        from .. import polite
        polite.lower_priority(); _g = polite.Gate(log=lambda m: print(m, flush=True))
    return _g


def attempt(chat, task, n, temperature=0.8, html=None, report=None):
    out, gate = [], _gate()
    for _ in range(n):
        try: text = _text(chat(messages(task, html, report), max_tokens=1500, thinking=False, temperature=temperature if n > 1 else 0.0))
        except Exception as e: text = f"(no answer: {e})"
        out.append(V.verify(task, text))
        gate.after_step()
    return out


def teach_task(chat, task, n=8, source="self", seen=None):
    seen = _seen() if seen is None else seen
    first = messages(task)
    results = attempt(chat, task, n)
    kept = []
    for r in results:
        if not r["accepted"] or len(kept) >= CAP: continue
        if len(kept) >= 2 and any(similarity(r["html"], k["html"]) >= SAME for k in kept): continue
        kept.append(r)
    stats = {"task": task["id"], "kind": task["kind"], "tries": n, "passed": sum(r["accepted"] for r in results), "kept": len(kept), "repairs_ok": 0}
    for r in kept:
        h = _hash("first", task["prompt"], r["html"])
        if h in seen: continue
        seen.add(h)
        _write("first", {"hash": h, "task": task["id"], "kind": task["kind"], "source": source, "system": first[0]["content"],
                         "prompt": first[1]["content"], "answer": answer_of(r["html"]), "score": r["score"]})
    # repairs: rejected screens that were real attempts (had some GameUI and HTML), each with the checker's report
    failed = [r for r in results if not r["accepted"] and "gui-" in r["html"]]
    uniq = []
    for r in failed:
        if all(similarity(r["html"], u["html"]) < SAME for u in uniq): uniq.append(r)
    for bad in uniq[:REPAIRS_PER_TASK]:
        fix = next((f for f in attempt(chat, task, 2, html=bad["html"], report=bad["report"]) if f["accepted"]), None)
        if not fix: continue
        stats["repairs_ok"] += 1
        msgs = messages(task, bad["html"], bad["report"])
        h = _hash("repair", task["prompt"], bad["html"], fix["html"])
        if h in seen: continue
        seen.add(h)
        _write("repair", {"hash": h, "task": task["id"], "kind": task["kind"], "source": source, "system": msgs[0]["content"],
                          "prompt": msgs[1]["content"], "answer": answer_of(fix["html"])})
    os.makedirs(DIR, exist_ok=True)
    with open(STATS, "a", encoding="utf-8") as f:
        f.write(json.dumps({**stats, "source": source, "time": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    return stats


def practice_tasks(count, seed=None):
    rng = random.Random(seed if seed is not None else time.time_ns())
    val = {t["prompt"] for t in T.val_set()}
    out = []
    while len(out) < count:
        t = T.generate(rng, "train")
        if t["prompt"] not in val: out.append(t)
    return out


def run(chat, task_list, n=8, source="self", log=print, stop=lambda: False):
    seen, summary, t0 = _seen(), {}, time.time()
    for i, task in enumerate(task_list):
        if stop(): break
        if task.get("split") == "val": continue
        s = teach_task(chat, task, n=n, source=source, seen=seen)
        a = summary.setdefault(task["kind"], {"tasks": 0, "solved": 0, "kept": 0, "repairs_ok": 0})
        a["tasks"] += 1; a["solved"] += s["kept"] > 0; a["kept"] += s["kept"]; a["repairs_ok"] += s["repairs_ok"]
        if i % 10 == 0: log(f"  {i + 1}/{len(task_list)} screens, {time.time() - t0:.0f}s: " + ", ".join(f"{k} {v['solved']}/{v['tasks']}" for k, v in sorted(summary.items())))
    return summary


def teach_from_references(count=400, seed=7):
    """the generator's own reference screens (all verified) as starter examples, so a model that has never seen GameUI
    has something to learn from before practice. Only training-split tasks; each reference is checked before it's kept."""
    seen, n = _seen(), 0
    for task in practice_tasks(count, seed):
        r = V.verify(task, task["reference"], sizes={"desktop": (1280, 800), "phone": (390, 844)})
        if not r["accepted"]: continue
        h = _hash("first", task["prompt"], r["html"])
        if h in seen: continue
        seen.add(h); n += 1
        first = messages(task)
        _write("first", {"hash": h, "task": task["id"], "kind": task["kind"], "source": "reference", "system": first[0]["content"],
                         "prompt": first[1]["content"], "answer": answer_of(r["html"]), "score": r["score"]})
    return n


def rows(kind="first"):
    p = FILES[kind]
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pass_rate(generate, task_list=None):
    """generate(messages) -> text. The UI skill's validation score: validation screens accepted on the first try."""
    task_list = task_list or T.val_set()
    ok, by = 0, {}
    for t in task_list:
        good = V.verify(t, generate(messages(t)))["accepted"]
        ok += good
        b = by.setdefault(t["kind"], [0, 0]); b[0] += good; b[1] += 1
    return {"pass_rate": round(ok / max(1, len(task_list)), 4), "tasks": len(task_list), "by_kind": {k: f"{a}/{c}" for k, (a, c) in by.items()}}
