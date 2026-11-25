"""The component library (plan: Phase 4, "verified components are reused", searched locally and offline): small pieces
of code that have passed their tests, kept with those tests so they can be reused in new projects instead of being
written again (and gotten wrong again).

A component: {name, lang, file (its file name), code, tests: {path: text}, description, signatures, source, uses}
Stored in data/components/library.jsonl. A component is only added after its own tests pass in the sandbox, and a
near-copy of one already there isn't added twice.

    components.add(name, lang, file, code, tests, description, source="my_game")
    components.search("cooldown timer for abilities", lang="lua")   -> best matches first
    components.prompt_block(found)                                   -> text for the AI's prompt
"""
import json, math, os, re, time

DIR = os.path.join("data", "components")
LIB = os.path.join(DIR, "library.jsonl")
SAME = 0.9


def _words(text):
    parts = re.findall(r"[A-Za-z]+|\d+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text or ""))
    return [w.lower() for w in parts if len(w) > 1]


def load():
    if not os.path.exists(LIB): return []
    return [json.loads(l) for l in open(LIB, encoding="utf-8") if l.strip()]


def _save(items):
    os.makedirs(DIR, exist_ok=True)
    tmp = LIB + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for c in items: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, LIB)


def add(name, lang, file, code, tests, description, source=""):
    """-> (added: bool, why). Checked in the sandbox first; near-copies of existing components aren't added."""
    from ..repair import debugger, repomap
    from ..coding.verify import similarity
    if not tests: return False, "a component needs tests"
    items = load()
    for c in items:
        if c["lang"] == lang and similarity(lang, c["code"], code) >= SAME:
            return False, f"already in the library as {c['name']}"
    r = debugger.run({file: code, **tests})
    if not debugger.ok(r) or not r["passed"]: return False, "its tests don't pass:\n" + debugger.report(r)
    sigs = [s["signature"] for s in repomap.index_file(file, code)["symbols"] if s["exported"] or lang != "js"][:20]
    items.append({"name": name, "lang": lang, "file": file, "code": code, "tests": tests, "description": description,
                  "signatures": sigs, "source": source, "added": time.strftime("%Y-%m-%d"), "uses": 0})
    _save(items)
    return True, "added"


def search(query, lang=None, k=3):
    """the k components that best match the words of a request (TF-IDF over names, descriptions and signatures)"""
    items = [c for c in load() if not lang or c["lang"] == lang]
    if not items: return []
    docs = [_words(c["name"] + " " + c["description"] + " " + " ".join(c["signatures"])) for c in items]
    df = {}
    for d in docs:
        for w in set(d): df[w] = df.get(w, 0) + 1
    q = _words(query)
    scored = []
    for c, d in zip(items, docs):
        s = sum((d.count(w) / (1 + len(d))) * math.log(1 + len(items) / df[w]) for w in set(q) if w in df)
        if s > 0: scored.append((s, c))
    return [c for _, c in sorted(scored, key=lambda x: -x[0])[:k]]


def used(names):
    """count a reuse (the most-used components are the most trusted)"""
    items = load()
    for c in items:
        if c["name"] in names: c["uses"] += 1
    _save(items)


def prompt_block(found):
    """components as text for a prompt: name, what it does, its signatures and its code"""
    if not found: return ""
    out = ["Checked components you can copy into the project (they pass their tests):"]
    for c in found:
        out.append(f"\n{c['name']} ({c['file']}): {c['description']}\n```\n{c['code'].rstrip()}\n```")
    return "\n".join(out)


def harvest(project, plan, feat):
    """after a feature is finished: its new files with their tests become components (if they pass on their own)"""
    from ..repair.repomap import lang_of
    from ..repair.debugger import lang_of_test
    added = []
    tests = {t: open(os.path.join(project, t), encoding="utf-8").read() for t in feat["tests"] if os.path.exists(os.path.join(project, t))}
    for path in feat.get("files", []):
        if lang_of_test(path) or not os.path.exists(os.path.join(project, path)): continue
        lang = {"javascript": "js", "typescript": "js", "python": "py", "lua": "lua"}.get(lang_of(path) or "")
        if not lang: continue
        mod = os.path.splitext(os.path.basename(path))[0]
        own = {t: s for t, s in tests.items() if mod in s}
        if not own: continue
        code = open(os.path.join(project, path), encoding="utf-8").read()
        flat = {os.path.basename(t): s for t, s in own.items()}  # the component is stored as one file with its tests
        ok, _ = add(f"{plan['name']}/{mod}", lang, os.path.basename(path), code, flat, f"{feat['title']}: {feat['description']}", source=plan["name"])
        if ok: added.append(mod)
    return added
