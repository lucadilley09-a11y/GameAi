"""Atomic multi-file changes (plan: Phase 4): related edits are staged in an isolated copy of the project. Every anchor
is verified first, then formatting, static checks, the relevant tests and the build all run. Only a fully passing
change set replaces the project; otherwise it is thrown away and the project is untouched. The active project never
holds a half-applied edit, and the previous state is kept for rollback.

    from gameai.repair import atomic
    r = atomic.change("my_game", [{"file": "src/player.js", "search": "speed = 3", "replace": "speed = 4"}, ...])
    r["applied"], print(r["report"])
    atomic.rollback("my_game")          # undo the last applied change

"Passing" means the change makes nothing worse: no new errors from the checkers, no new unjustified suppressions,
no test that passed before now failing, every script still loads, and every file index.html points to exists.
Problems the project already had are reported but don't block (otherwise one old error would block every fix).

The project's own working folder for this is <project>/.gameai (staging copies, history, a journal).
"""
import hashlib, json, os, re, shutil, time

from . import edits as E
from .repomap import read_dir, SKIP_DIRS

KEEP_HISTORY = 20
WORK = ".gameai"


def _work(project, *p):
    return os.path.join(project, WORK, *p)


def _ignore(d, names):
    return [n for n in names if n in SKIP_DIRS or n == WORK or n == ".git"]


def project_version(files):
    h = hashlib.sha256()
    for p in sorted(files): h.update(p.encode()); h.update(b"\0"); h.update(files[p].encode("utf-8")); h.update(b"\0")
    return h.hexdigest()[:16]


# ---------- what "passing" means ----------
def _key(d):
    return (d["file"], d.get("code") or d.get("kind"), d["message"], (d.get("line_text") or "").strip())


def _html_refs(files):
    """files that index.html (or any .html) points to but that don't exist"""
    missing = []
    for p, t in files.items():
        if not p.endswith(".html"): continue
        base = os.path.dirname(p)
        for ref in re.findall(r"""<(?:script|link|img)\b[^>]*?(?:src|href)\s*=\s*["']([^"'#?]+)""", t, re.I):
            if re.match(r"^[a-z]+:|^//|^data:", ref, re.I): continue
            target = os.path.normpath(os.path.join(base, ref)).replace("\\", "/").lstrip("./")
            if target not in files and not os.path.basename(target) in ("gamekit.js",): missing.append(f"{p} points to {ref}, which doesn't exist")
    return missing


def evaluate(root_or_files, run_tests=True, inspect=False):
    """all the checks on one version of a project -> {errors, unjustified, tests, test_result, load_errors, build}"""
    from .. import checks
    from . import debugger
    files = root_or_files if isinstance(root_or_files, dict) else read_dir(root_or_files)
    code = {p: t for p, t in files.items() if p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".py", ".lua"))}
    diags = checks.check(code) if code else []
    sup = checks._sup.suppressions(code)
    out = {"errors": [d for d in diags if d["severity"] == "error"], "warnings": [d for d in diags if d["severity"] == "warning"],
           "unjustified": [s for s in sup if not s["justified"]], "build": _html_refs(files), "tests": {}, "load_errors": []}
    if run_tests and any(debugger.is_test(p) for p in files):
        r = debugger.run(root_or_files if not isinstance(root_or_files, dict) else files, inspect=inspect)
        out["test_result"] = r
        out["tests"] = {t["name"]: bool(t.get("passed")) for t in r.get("tests", [])}
        out["load_errors"] = [e["file"] + ": " + e["error"] for e in r.get("load_errors", [])]
        if r.get("timed_out"): out["load_errors"].append("the tests timed out (an endless loop?)")
    return out


def _baseline(project, files):
    """the checks on the project as it is now; remembered per project version, so it only runs once per version"""
    path, ver = _work(project, "baseline.json"), project_version(files)
    try:
        b = json.load(open(path, encoding="utf-8"))
        if b.get("version") == ver: return b["result"]
    except (OSError, ValueError):
        pass
    r = evaluate(project)
    r.pop("test_result", None)
    os.makedirs(_work(project), exist_ok=True)
    json.dump({"version": ver, "result": r}, open(path, "w", encoding="utf-8"))
    return r


def compare(before, after):
    """-> list of reasons the change is worse (empty: it passes)"""
    old_err = {_key(d) for d in before["errors"]}
    old_sup = {(s["file"], s["kind"], s["line_text"].strip()) for s in before["unjustified"]}
    bad = [f"new error: {d['file']}:{d['line']} [{d.get('code')}] {d['message']}" for d in after["errors"] if _key(d) not in old_err]
    bad += [f"new unjustified {s['kind']}: {s['file']}:{s['line']}" for s in after["unjustified"] if (s["file"], s["kind"], s["line_text"].strip()) not in old_sup]
    bad += [f"test now fails: {name}" for name, ok in after["tests"].items() if not ok and before["tests"].get(name, False)]
    bad += [f"test disappeared: {name}" for name, ok in before["tests"].items() if ok and name not in after["tests"]]
    bad += [f"doesn't load: {e}" for e in after["load_errors"] if e not in before["load_errors"]]
    bad += [f"build: {m}" for m in after["build"] if m not in before["build"]]
    return bad


# ---------- applying ----------
def recover(project):
    """if the PC stopped in the middle of replacing files, put the project back as it was before that change"""
    j = _work(project, "journal.json")
    if os.path.exists(j):
        entry = json.load(open(j, encoding="utf-8"))["history"]
        _restore(project, entry); os.remove(j)
        return f"an unfinished change was undone ({entry})"
    return None


def _restore(project, entry):
    d = _work(project, "history", entry)
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    for rel in man["saved"]:
        dest = os.path.join(project, *rel.split("/")); os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy2(os.path.join(d, "files", *rel.split("/")), dest + ".tmp"); os.replace(dest + ".tmp", dest)
    for rel in man["created"]:
        p = os.path.join(project, *rel.split("/"))
        if os.path.exists(p): os.remove(p)


def change(project, edit_list, run_tests=True, inspect_on_fail=True):
    """-> {applied, stage, problems, done, diffs, report, history}"""
    from .. import checks
    from . import debugger
    project = os.path.abspath(project)
    note = recover(project)
    files = read_dir(project)
    res = {"applied": False, "stage": "edits", "problems": [], "done": [], "diffs": {}, "history": None, "recovered": note}
    # 1. every anchor verified, in memory
    try:
        new, done = E.apply(files, edit_list)
    except E.EditError as ex:
        res["problems"] = ex.problems; res["report"] = "NOT APPLIED (edits)\n" + "\n".join(ex.problems); return res
    res["done"] = done
    changed = sorted(p for p in new if new[p] != files.get(p))
    deleted = sorted(p for p in files if p not in new)
    # 2. formatting (shown, not silent)
    fmt_in = {p: new[p] for p in changed}
    formatted, diffs, fmt_err = checks.format_files(fmt_in) if fmt_in else ({}, {}, {})
    new.update(formatted); res["diffs"] = diffs
    # 3. an isolated copy with the change in it
    stage = _work(project, "staging", time.strftime("%Y%m%d-%H%M%S"))
    shutil.rmtree(_work(project, "staging"), ignore_errors=True)
    shutil.copytree(project, stage, ignore=_ignore)
    for p in changed:
        dest = os.path.join(stage, *p.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "w", encoding="utf-8", newline="").write(new[p])
    for p in deleted: os.remove(os.path.join(stage, *p.split("/")))
    try:
        # 4. checks, tests and build on the copy, compared with the project as it is now
        res["stage"] = "checks"
        before = _baseline(project, files)
        after = evaluate(stage, run_tests=run_tests, inspect=inspect_on_fail)
        bad = compare(before, after) + [f"{p} couldn't be formatted: {m}" for p, m in fmt_err.items()]
        tr = after.pop("test_result", None)
        if bad:
            res["problems"] = bad
            res["report"] = "NOT APPLIED: the change makes things worse\n" + "\n".join("- " + b for b in bad) + \
                            ("\n\n" + debugger.report(tr) if tr and not debugger.ok(tr) else "")
            return res
        # 5. it passes: keep the old versions for rollback, then replace the files (with a journal, in case the PC stops)
        res["stage"] = "apply"
        entry = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 10**9:09d}"  # sorts by time, never collides
        hist = _work(project, "history", entry)
        saved = [p for p in changed + deleted if p in files]
        for p in saved:
            dest = os.path.join(hist, "files", *p.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(os.path.join(project, *p.split("/")), dest)
        os.makedirs(hist, exist_ok=True)
        json.dump({"saved": saved, "created": [p for p in changed if p not in files], "edits": done, "time": entry},
                  open(os.path.join(hist, "manifest.json"), "w", encoding="utf-8"), indent=1)
        json.dump({"history": entry}, open(_work(project, "journal.json"), "w"))
        for p in changed:
            dest = os.path.join(project, *p.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest + ".tmp", "w", encoding="utf-8", newline="").write(new[p]); os.replace(dest + ".tmp", dest)
        for p in deleted: os.remove(os.path.join(project, *p.split("/")))
        os.remove(_work(project, "journal.json"))
        _prune(project)
        res.update(applied=True, history=entry)
        rest = [t for t, ok in after["tests"].items() if not ok]
        res["report"] = "APPLIED\n" + "\n".join(done) + (f"\nformatted: {', '.join(diffs)}" if diffs else "") + \
                        (f"\nstill failing (they failed before too): {', '.join(rest)}" if rest else "")
        return res
    finally:
        shutil.rmtree(_work(project, "staging"), ignore_errors=True)


def _prune(project):
    h = _work(project, "history")
    entries = sorted(os.listdir(h)) if os.path.isdir(h) else []
    for old in entries[:-KEEP_HISTORY]: shutil.rmtree(os.path.join(h, old), ignore_errors=True)


def history(project):
    h = _work(project, "history")
    out = []
    for e in sorted(os.listdir(h)) if os.path.isdir(h) else []:
        try: out.append(json.load(open(os.path.join(h, e, "manifest.json"), encoding="utf-8")))
        except (OSError, ValueError): pass
    return out


def rollback(project):
    """undo the last applied change -> what was undone, or None"""
    h = _work(project, "history")
    entries = sorted(os.listdir(h)) if os.path.isdir(h) else []
    if not entries: return None
    _restore(project, entries[-1])
    man = json.load(open(os.path.join(h, entries[-1], "manifest.json"), encoding="utf-8"))
    shutil.rmtree(os.path.join(h, entries[-1]))
    return man
