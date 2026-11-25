"""The project planner (plan: Phase 4): large programs are built over many sessions from a design document and a feature
list, one tested feature at a time, with every finished feature's tests kept as regression tests.

Each project keeps its plan in <project>/.gameai/plan.json, plus two readable files the user can open:
    DESIGN.md     what the game is: goal, mechanics, controls, rules, look and sound, decisions made along the way
    FEATURES.md   the feature list with each one's status

A feature: {id, title, description, acceptance: [what must be true when it's done], depends: [feature ids],
status: todo | doing | done | blocked, tests: [its test files, protected once done], attempts, notes}

    p = project.create("my_game", plan_dict)     # from the planner model's JSON, checked first
    f = project.next_feature(p)                  # the first unfinished feature whose dependencies are done
    project.summary(p)                           # a compact text for the AI's context in a new session
"""
import json, os, time

FILE = os.path.join(".gameai", "plan.json")
STATUSES = ("todo", "doing", "done", "blocked")


class PlanError(Exception):
    pass


def check(plan):
    """-> list of problems with a plan (empty = fine)"""
    probs = []
    for k in ("name", "goal", "design", "features"):
        if not plan.get(k): probs.append(f"missing {k}")
    feats = plan.get("features") or []
    ids = [f.get("id") for f in feats]
    if len(set(ids)) != len(ids) or not all(ids): probs.append("every feature needs its own id")
    for f in feats:
        if not f.get("title"): probs.append(f"feature {f.get('id')}: no title")
        if not f.get("acceptance"): probs.append(f"feature {f.get('id')}: no acceptance criteria (how will we know it's done?)")
        for d in f.get("depends", []):
            if d not in ids: probs.append(f"feature {f.get('id')}: depends on unknown feature {d}")
    # no circles in the dependencies
    deps = {f.get("id"): set(f.get("depends", [])) for f in feats}
    done, changed = set(), True
    while changed:
        changed = False
        for i, d in deps.items():
            if i not in done and d <= done: done.add(i); changed = True
    if len(done) != len(deps) and not any("unknown feature" in p for p in probs):
        probs.append("the features depend on each other in a circle: " + ", ".join(sorted(set(deps) - done)))
    return probs


def _path(project): return os.path.join(project, FILE)


def load(project):
    try: return json.load(open(_path(project), encoding="utf-8"))
    except FileNotFoundError: return None


def save(project, plan):
    plan["updated"] = time.strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(_path(project)), exist_ok=True)
    tmp = _path(project) + ".tmp"
    json.dump(plan, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(tmp, _path(project))
    open(os.path.join(project, "DESIGN.md"), "w", encoding="utf-8").write(design_md(plan))
    open(os.path.join(project, "FEATURES.md"), "w", encoding="utf-8").write(features_md(plan))


def create(project, plan):
    """a new project from a plan (the planner model's JSON). Refused if the plan has problems."""
    probs = check(plan)
    if probs: raise PlanError("\n".join(probs))
    if load(project): raise PlanError("this project already has a plan (use update or add_feature)")
    plan = json.loads(json.dumps(plan))
    for f in plan["features"]:
        f.setdefault("description", ""); f.setdefault("depends", [])
        f.update(status="todo", tests=[], attempts=0, notes=[])
    plan.setdefault("decisions", [])
    plan["created"] = time.strftime("%Y-%m-%d %H:%M")
    os.makedirs(project, exist_ok=True)
    save(project, plan)
    return plan


def feature(plan, fid):
    return next((f for f in plan["features"] if f["id"] == fid), None)


def next_feature(plan):
    done = {f["id"] for f in plan["features"] if f["status"] == "done"}
    for f in plan["features"]:
        if f["status"] in ("todo", "doing") and set(f["depends"]) <= done: return f
    return None


def protected_tests(plan):
    """the test files of finished features: kept as regression tests, never changed or removed by later features"""
    return {t for f in plan["features"] if f["status"] == "done" for t in f["tests"]}


def add_feature(project, plan, feat, after=None):
    """a new feature (for example the user asked for more); checked like the plan"""
    feat = dict(feat); feat.setdefault("depends", []); feat.setdefault("description", "")
    feat.update(status="todo", tests=[], attempts=0, notes=[])
    trial = dict(plan, features=plan["features"] + [feat])
    probs = check(trial)
    if probs: raise PlanError("\n".join(probs))
    i = next((k for k, f in enumerate(plan["features"]) if f["id"] == after), len(plan["features"]) - 1) + 1
    plan["features"].insert(i, feat)
    save(project, plan)
    return feat


def decide(project, plan, text):
    """record a project decision (kept in DESIGN.md, and in every later session's summary)"""
    plan.setdefault("decisions", []).append({"time": time.strftime("%Y-%m-%d %H:%M"), "text": text})
    save(project, plan)


def summary(plan, max_chars=3000):
    """what a new session needs to know, compactly: the goal, the design in brief, what's done, what's next"""
    nxt = next_feature(plan)
    lines = [f"Project: {plan['name']}: {plan['goal']}"]
    d = plan["design"]
    lines += [f"- {k}: {v}" for k, v in (d.items() if isinstance(d, dict) else [("design", d)])]
    if plan.get("decisions"): lines.append("Decisions: " + "; ".join(x["text"] for x in plan["decisions"][-8:]))
    done = [f for f in plan["features"] if f["status"] == "done"]
    lines.append(f"Done ({len(done)}/{len(plan['features'])}): " + ", ".join(f["title"] for f in done) if done else "Nothing finished yet.")
    if nxt: lines.append(f"Next: {nxt['title']}: {nxt['description']} (done when: {'; '.join(nxt['acceptance'])})")
    blocked = [f["title"] for f in plan["features"] if f["status"] == "blocked"]
    if blocked: lines.append("Blocked (needs help): " + ", ".join(blocked))
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[:max_chars - 20] + "\n... (cut)"


MARK = {"todo": "[ ]", "doing": "[~]", "done": "[x]", "blocked": "[!]"}


def design_md(plan):
    d = plan["design"]
    out = [f"# {plan['name']}", "", plan["goal"], ""]
    if isinstance(d, dict):
        for k, v in d.items(): out += [f"## {k[:1].upper() + k[1:]}", "", v if isinstance(v, str) else json.dumps(v), ""]
    else:
        out += [str(d), ""]
    if plan.get("decisions"):
        out += ["## Decisions", ""] + [f"- {x['time']}: {x['text']}" for x in plan["decisions"]] + [""]
    return "\n".join(out)


def features_md(plan):
    out = [f"# Features: {plan['name']}", "", "[x] done, [~] in progress, [!] blocked, [ ] to do", ""]
    for f in plan["features"]:
        out.append(f"- {MARK[f['status']]} **{f['title']}** ({f['id']})" + (f", after {', '.join(f['depends'])}" if f["depends"] else ""))
        if f["description"]: out.append(f"  {f['description']}")
        out += [f"  - done when: {a}" for a in f["acceptance"]]
        if f["tests"]: out.append(f"  - tests: {', '.join(f['tests'])}")
        if f["status"] == "blocked" and f["notes"]: out.append(f"  - why: {f['notes'][-1]}")
    return "\n".join(out) + "\n"
