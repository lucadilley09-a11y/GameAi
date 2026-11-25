"""Checking one solution to a coding task (plan: Phase 4, verified self-teaching and its guards against checker-gaming).

    r = verify(task, answer_text)
    r["accepted"]   passes everything below; only these are ever trained on
    r["report"]     what went wrong, from the VISIBLE tests only (it goes into repair prompts; hidden tests never leak)

A solution is accepted only if:
  1. it touches nothing but the task's file, and doesn't tamper with the test machinery (redefining test/expect/assert,
     reaching the sandbox's bridge, exiting early...);
  2. it doesn't copy answers out of the visible tests (a hard-coded table of the expected values);
  3. the checkers find no errors and no unjustified suppressions in it;
  4. every visible AND every hidden test passes (hidden tests check the same behaviour on inputs the AI never saw).
Passing makes a solution eligible; the separate quality numbers (time taken, length) are recorded alongside.
"""
import re

from .. import checks
from ..repair import debugger

FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)

CHEATS = {
    "js": [(r"\b(?:globalThis|window|self)\s*\.\s*(?:test|expect|assert|__\w+)\b", "changes the test functions"),
           (r"(?:^|[;{}\s])(?:test|expect|assert)\s*=[^=]", "redefines a test function"),
           (r"__host|__tests|__run|__list|__currentFile", "reaches into the sandbox"),
           (r"\bprocess\s*\.\s*exit|\bDeno\b", "exits early"),
           (r"\bObject\s*\.\s*defineProperty\s*\(\s*(?:globalThis|window)", "changes globals"),
           (r"\b(?:eval|Function)\s*\(", "builds code from text")],
    "py": [(r"\bbuiltins\b|__builtins__", "changes Python's built-ins"),
           (r"\bsys\s*\.\s*modules\b", "changes loaded modules"),
           (r"\b(?:os\s*\.\s*_exit|sys\s*\.\s*exit)|(?<![.\w])(?:quit|exit)\s*\(", "exits early"),
           (r"\bAssertionError\s*=|\bdef\s+test_", "changes or adds tests"),
           (r"\bunittest\s*\.\s*mock\b|\bmonkeypatch\b", "patches other code"),
           (r"\b(?:eval|exec|compile)\s*\(", "builds code from text"),
           (r"\b__import__\s*\(", "imports by name")],
    "lua": [(r"(?:^|[^.\w])(?:test|expect|assert)\s*=[^=]", "redefines a test function"),
            (r"\b_G\b|\b_ENV\b|\bsetfenv\b|\bgetfenv\b", "changes the global environment"),
            (r"(?<![.:\w])debug\s*\.|(?<![.:\w])os\s*\.\s*exit\b", "uses debug or exits"),
            (r"(?<![.:\w])load(?:string)?\s*\(", "builds code from text")],
}


def extract(answer, file=None):
    """the code from an answer: the code block (the last one that looks like the file), or the whole text"""
    blocks = FENCE.findall(answer or "")
    if not blocks: return (answer or "").strip() + "\n"
    return blocks[-1].rstrip() + "\n"


def _strings_blanked(code):
    return re.sub(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)""", lambda m: m.group(0)[0] * 2, code)


def cheat_reasons(task, code):
    no_str = _strings_blanked(code)
    no_comments = re.sub(r"//[^\n]*|/\*.*?\*/|#[^\n]*|--[^\n]*", "", no_str, flags=re.S) if task["lang"] != "py" else re.sub(r"#[^\n]*", "", no_str)
    return [why for rx, why in CHEATS[task["lang"]] if re.search(rx, no_comments, re.M)]


LITERAL = re.compile(r"""(?<![\w.])(-?\d+\.\d+|-?\d{2,})(?![\w.])|"([^"\n]{3,})"|'([^'\n]{3,})'""")


def copied_answers(task, code):
    """literals that appear in the visible tests' expectations but nowhere in the prompt or the starter files, and that the
    solution contains: a sign it hard-codes the tests' answers (2 or more of them)"""
    tests = "\n".join(task["tests"].values())
    expected = set()
    for m in re.finditer(r"(?:toBe|toEqual|==)\s*\(?\s*([^;\n)]+)", tests):
        for lit in LITERAL.finditer(m.group(1)):
            expected.add(next(g for g in lit.groups() if g))
    known = task["prompt"] + "\n".join(task.get("starter", {}).values())
    sus = [x for x in expected if x not in known and x in code]
    return sus if len(sus) >= 2 else []


def run_tests(task, code):
    files = dict(task.get("starter", {}))
    files[task["file"]] = code
    files.update(task["tests"]); files.update(task["hidden"])
    return debugger.run(files)


def _visible_only(task, r):
    vis = set(task["tests"])
    return {**r, "tests": [t for t in r["tests"] if t.get("file") in vis],
            "load_errors": [e for e in r["load_errors"] if e["file"] not in task["hidden"]],
            "passed": sum(1 for t in r["tests"] if t.get("file") in vis and t.get("passed")),
            "failed": sum(1 for t in r["tests"] if t.get("file") in vis and not t.get("passed"))}


def verify(task, answer):
    code = extract(answer, task["file"])
    out = {"code": code, "accepted": False, "reasons": [], "report": "", "visible_ok": False, "hidden_ok": False, "ms": 0, "lines": code.count("\n")}
    cheats = cheat_reasons(task, code)
    if cheats:
        out["reasons"] = [f"not allowed: {c}" for c in cheats]; out["report"] = "\n".join(out["reasons"]); return out
    copied = copied_answers(task, code)
    if copied:
        out["reasons"] = [f"looks like it copies the tests' answers ({', '.join(copied[:4])})"]; out["report"] = out["reasons"][0]; return out
    diags = [d for d in checks.check({task["file"]: code}, kit=task["lang"] == "js") if d["severity"] == "error"]
    sup = [s for s in checks._sup.suppressions({task["file"]: code}) if not s["justified"]]
    r = run_tests(task, code)
    vis = _visible_only(task, r)
    out["visible_ok"] = debugger.ok(vis) and vis["passed"] > 0
    hidden = [t for t in r["tests"] if t.get("file") in task["hidden"]]
    out["hidden_ok"] = bool(hidden) and all(t.get("passed") for t in hidden) and not any(e["file"] in task["hidden"] for e in r["load_errors"])
    out["ms"] = sum(t.get("ms", 0) for t in r["tests"])
    if diags: out["reasons"].append(f"{len(diags)} checker error(s)")
    if sup: out["reasons"].append(f"{len(sup)} unjustified suppression(s)")
    if not out["visible_ok"]: out["reasons"].append("visible tests fail")
    elif not out["hidden_ok"]: out["reasons"].append("hidden tests fail (fits the shown tests only)")
    out["accepted"] = not out["reasons"]
    lines = [debugger.report(vis)] if not out["visible_ok"] else []
    lines += [f"{d['file']}:{d['line']} [{d.get('code')}] {d['message']}" for d in diags[:8]]
    lines += [f"{s['file']}:{s['line']} {s['message']}" for s in sup[:4]]
    if out["visible_ok"] and not out["hidden_ok"]: lines.append("The shown tests pass, but the code fails on other inputs: make it work in general, not just for these tests.")
    out["report"] = "\n".join(lines)
    return out


# ---------- novelty: how alike two solutions are (a soft limit, never a rule) ----------
def tokens(lang, code):
    """the code's tokens with names and values replaced by their kind, so renaming a variable doesn't make it 'new'"""
    from ..repair import repomap
    name = {"js": "javascript", "py": "python", "lua": "lua"}[lang]
    try: tree = repomap.parser(name).parse(code.encode("utf-8"))
    except Exception: return code.split()
    out = []
    def walk(n):
        if n.child_count == 0:
            t = n.type
            out.append("ID" if "identifier" in t else "LIT" if any(k in t for k in ("number", "string", "integer", "float")) else t)
        for c in n.children: walk(c)
    walk(tree.root_node)
    return out


def similarity(lang, a, b, k=4):
    ta, tb = tokens(lang, a), tokens(lang, b)
    ga = {tuple(ta[i:i + k]) for i in range(max(1, len(ta) - k + 1))}
    gb = {tuple(tb[i:i + k]) for i in range(max(1, len(tb) - k + 1))}
    return len(ga & gb) / max(1, len(ga | gb))
