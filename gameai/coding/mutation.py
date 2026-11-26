"""Mutation testing (plan: Phase 4): checks that a task's tests really catch broken code. Small bugs ("mutants") are put
into the correct solution one at a time: + becomes -, < becomes <=, a number changes by 1, true becomes false, and so on.
Good tests fail on (kill) most mutants; a mutant that survives shows a hole in the tests.

Plan: this runs offline, before a test suite is finalized or sealed, never in the live generation or training loop.

    r = mutation.score(task)      # {"score": 0.85, "killed": 17, "total": 20, "survivors": [...]}
"""
import random, re

from .verify import run_tests
from ..repair import repomap

THRESHOLD = 0.7   # a suite must catch this share of mutants to be finalized
MAX_MUTANTS = 24

SWAP = {"+": "-", "-": "+", "*": "/", "/": "*", "<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "~=", "===": "!==", "!==": "===",
        "!=": "==", "~=": "==", "&&": "||", "||": "&&", "and": "or", "or": "and", "true": "false", "false": "true",
        "True": "False", "False": "True", "%": "*", "+=": "-=", "-=": "+="}
LANG = {"js": "javascript", "py": "python", "lua": "lua"}


def mutants(lang, code, limit=MAX_MUTANTS, seed=0):
    """-> [(description, mutated code)]: one small change each, at real operator/literal tokens (not inside strings or comments)"""
    tree = repomap.parser(LANG[lang]).parse(code.encode("utf-8"))
    src = code.encode("utf-8")
    spots = []
    def walk(n):
        if n.type in ("comment", "string", "template_string", "string_content"): return
        if n.child_count == 0:
            tok = src[n.start_byte:n.end_byte].decode("utf-8", "replace")
            line = n.start_point[0] + 1
            if tok in SWAP and not (lang == "py" and tok in ("~=",)) and not (lang == "lua" and tok in ("===", "!==", "!=", "&&", "||")):
                new = SWAP[tok]
                if lang == "py" and new == "~=": new = "!="
                if lang == "lua" and new == "!=": new = "~="
                if lang == "js" and new == "~=": new = "!="
                spots.append((n.start_byte, n.end_byte, new, f"line {line}: {tok} -> {new}"))
            elif re.fullmatch(r"\d+(\.\d+)?", tok):
                v = float(tok) if "." in tok else int(tok)
                spots.append((n.start_byte, n.end_byte, str(v + 1), f"line {line}: {tok} -> {v + 1}"))
        for c in n.children: walk(c)
    walk(tree.root_node)
    random.Random(seed).shuffle(spots)
    out = []
    for a, b, new, why in spots[:limit]:
        m = (src[:a] + new.encode() + src[b:]).decode("utf-8", "replace")
        if m != code: out.append((why, m))
    return out


def score(task, limit=MAX_MUTANTS):
    """the share of mutants of the reference solution that the task's tests (visible and hidden) catch"""
    ms = mutants(task["lang"], task["reference"], limit)
    killed, survivors = 0, []
    for why, code in ms:
        r = run_tests(task, code)
        dead = r.get("failed", 0) > 0 or r.get("load_errors") or r.get("timed_out") or not r.get("tests")
        if dead: killed += 1
        else: survivors.append(why)
    # no mutants (code without operators or numbers): nothing to measure, so it doesn't count against the suite
    return {"score": round(killed / len(ms), 3) if ms else None, "killed": killed, "total": len(ms), "survivors": survivors}


def finalize(task_list, log=print):
    """for suites about to be finalized/sealed: -> (good tasks, weak tasks with their survivors)"""
    good, weak = [], []
    for t in task_list:
        r = score(t)
        ok = r["score"] is None or r["score"] >= THRESHOLD
        (good if ok else weak).append((t, r))
        if not ok: log(f"  {t['id']}: tests catch only {r['killed']}/{r['total']} broken versions")
    return good, weak
