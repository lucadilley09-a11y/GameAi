"""Skill adapters, each versioned on its own. The base model is never changed.

    models/adapters/<skill>/v001/      the adapter (PEFT files), its llama.cpp copy (adapter.gguf) and meta.json
    models/adapters/<skill>/active.json   which version is in use

A new version is used only if its validation total is better and no validation set is clearly worse (the same rule
as today's game AI). The last 20 versions of every skill are kept; rollback picks any of them.

    python -m gameai.adapters                 every skill, its versions and which one is in use
    python -m gameai.adapters rollback code 4 use version 4 of the code adapter again
"""
import json, os, shutil, time

ROOT = os.path.join("models", "adapters")
KEEP = 20


def _dir(skill, v=None):
    return os.path.join(ROOT, skill) if v is None else os.path.join(ROOT, skill, f"v{v:03d}")


def versions(skill):
    d = _dir(skill)
    if not os.path.isdir(d): return []
    out = []
    for f in sorted(os.listdir(d)):
        m = os.path.join(d, f, "meta.json")
        if f.startswith("v") and os.path.exists(m): out.append(json.load(open(m)))
    return out


def active(skill):
    p = os.path.join(_dir(skill), "active.json")
    return json.load(open(p)) if os.path.exists(p) else None


def new_version(skill):
    """-> (version number, folder to train into)"""
    have = [v["version"] for v in versions(skill)]
    if os.path.isdir(_dir(skill)):
        have += [int(f[1:]) for f in os.listdir(_dir(skill)) if f.startswith("v") and f[1:].isdigit()]
    n = max(have, default=0) + 1
    path = _dir(skill, n); os.makedirs(path, exist_ok=True)
    return n, path


def finish(skill, v, base, base_revision, scores, **meta):
    """Record a trained version (scores: {set: {"score": 0..1, "n": count}} on validation)."""
    m = {"skill": skill, "version": v, "base": base, "base_revision": base_revision, "scores": scores,
         "total": total(scores), "created": time.strftime("%Y-%m-%d %H:%M"), **meta}
    json.dump(m, open(os.path.join(_dir(skill, v), "meta.json"), "w"), indent=1)
    return m


def total(scores, weights=None):
    w = {k: (weights or {}).get(k, 1.0) for k in scores}
    return round(sum(scores[k]["score"] * w[k] for k in scores) / max(1e-9, sum(w.values())), 5) if scores else 0.0


def clearly_worse(new, old):
    """Sets where the new version is clearly worse: more than 1 point, or more than 1 task on small sets."""
    out = []
    for k, o in old.items():
        if k in new:
            allow = max(0.01, 1.0 / max(1, new[k].get("n", 1)))
            if new[k]["score"] < o["score"] - allow: out.append(k)
    return out


def decide(skill, v):
    """Promote version v if it beats the version in use on validation with no clearly-worse set. -> (promoted, why)"""
    new = next(x for x in versions(skill) if x["version"] == v)
    cur = active(skill)
    if cur:
        old = next((x for x in versions(skill) if x["version"] == cur["version"]), None)
        if old:  # validation sets are the same for every base, so a new base must beat the old one too
            bad = clearly_worse(new["scores"], old["scores"])
            if bad: return False, f"clearly worse on: {', '.join(bad)}"
            if new["total"] <= old["total"]: return False, f"not better overall ({new['total']:.4f} vs {old['total']:.4f})"
    use(skill, v, why="better on validation" if cur else "first version")
    prune(skill)
    return True, "better on validation" if cur else "first version"


def use(skill, v, why=""):
    if not os.path.exists(os.path.join(_dir(skill, v), "meta.json")): raise ValueError(f"{skill} version {v} isn't kept")
    p = os.path.join(_dir(skill), "active.json")
    hist = (active(skill) or {}).get("history", [])
    json.dump({"version": v, "since": time.strftime("%Y-%m-%d %H:%M"), "why": why,
               "history": (hist + [{"version": v, "time": time.strftime("%Y-%m-%d %H:%M"), "why": why}])[-50:]}, open(p + ".new", "w"), indent=1)
    os.replace(p + ".new", p)


def rollback(skill, v):
    use(skill, v, why="rollback")


def prune(skill):
    """Keep the last KEEP versions; never delete the one in use."""
    vs = [x["version"] for x in versions(skill)]
    cur = (active(skill) or {}).get("version")
    for v in vs[:-KEEP]:
        if v != cur: shutil.rmtree(_dir(skill, v), ignore_errors=True)


def skills():
    return sorted(f for f in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, f))) if os.path.isdir(ROOT) else []


def main():
    import sys
    a = sys.argv[1:]
    if a[:1] == ["rollback"] and len(a) == 3:
        rollback(a[1], int(a[2])); print(f"{a[1]}: using version {a[2]}"); return
    for s in skills():
        cur = (active(s) or {}).get("version")
        for v in versions(s):
            print(f"{s:14} v{v['version']:03d}{' (in use)' if v['version'] == cur else '         '}  total {v['total']:.4f}  base {v['base']}  {v['created']}")
    if not skills(): print("no adapters yet (they come with Phase 2 training)")


if __name__ == "__main__":
    main()
