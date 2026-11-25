"""The autopilot: does every step of the plan by itself, in order, and asks you simple questions when it needs a
decision. Nothing for you to type: answers are buttons in the app. The long work runs only when you're away from
the app (or only at night, if you prefer), never in the middle of something you're doing.

Steps (each one only once; a failed step is tried again later, then it asks you):
  measure the PC -> the general collection -> the model bake-off -> choose the base (fixed rule, validation only)
  -> install it -> test how much general data to mix in -> train the first game-design adapter -> first release

    python -m gameai.autopilot status
    python -m gameai.autopilot run <step>     (the app does this)
"""
import json, os, sys, time

PATH = os.path.join("memory", "autopilot.json")
IDLE_MIN = float(os.environ.get("GAMEAI_AUTOPILOT_IDLE_MIN", 10))  # minutes away from the app before long work starts
NIGHT = (23, 7)        # "only at night": from 11 pm to 7 am
MAX_TRIES = 2

QUESTIONS = {
    "when": {"text": "I can do the remaining setup by myself: testing your PC, picking the best AI model and training it. "
                     "When should I do the long work?",
             "choices": ["When I'm away from the app", "Only at night", "Not now (pause the autopilot)"]},
    "general": {"text": "To keep the AI good at normal chat while it learns games, I mix in everyday conversations. Which should I use?",
                "choices": ["All four sets (about 1 GB download, some written by other AIs)", "Only the human-written ones (about 350 MB)"]},
    "bakeoff": {"text": "Next I test the candidate AI models one at a time to pick the best one for your PC: about 30 GB of downloads "
                        "spread over about 6 hours, using at most about 17 GB of disk at once (each is deleted after its test). OK?",
                "choices": ["Yes, test them all", "Only the two main ones (about 2-3 hours)", "Not now"]},
}

STEPS = [  # (id, plain label, what it needs first, the command it runs)
    ("hardware", "testing your PC's hardware", "when", ["gameai.phase0", "hardware"]),
    ("fullstack", "testing the whole app at once", "when", ["gameai.phase0", "fullstack"]),
    ("profile", "timing training and game making", "when", ["gameai.phase0", "profile"]),
    ("independence", "checking everything works offline", "when", ["gameai.phase0", "independence"]),
    ("checkers", "installing the code checkers (about 65 MB, a minute)", "when", ["gameai.checks", "install"]),
    ("general", "getting the everyday-conversation data", "general", ["gameai.autopilot", "run", "general"]),
    ("bakeoff", "testing the candidate AI models (hours)", "bakeoff", ["gameai.autopilot", "run", "bakeoff"]),
    ("choose", "choosing the best model", None, ["gameai.autopilot", "run", "choose"]),
    ("install", "installing the chosen model", None, ["gameai.autopilot", "run", "install"]),
    ("mix", "testing how much everyday data to mix in (about 2 hours)", None, ["gameai.autopilot", "run", "mix"]),
    ("adapter", "training the game-design skill (about 2 hours)", None, ["gameai.autopilot", "run", "adapter"]),
    ("release", "making and checking the first release", None, ["gameai.autopilot", "run", "release"]),
]


def load():
    if os.path.exists(PATH):
        try: return json.load(open(PATH, encoding="utf-8"))
        except Exception: pass
    return {"on": True, "answers": {}, "done": {}, "tries": {}, "news": [], "asked_at": {}}


def save(st):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".new"; json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1); os.replace(tmp, PATH)


def news(st, text, kind="d"):
    st["news"] = (st.get("news", []) + [{"id": int(time.time() * 1000), "text": text, "kind": kind}])[-30:]


def answer(qid, choice):
    st = load()
    q = QUESTIONS.get(qid) or (st.get("extra_questions") or {}).get(qid)
    if not q or choice not in q["choices"]: raise ValueError("that isn't one of the choices")
    st["answers"][qid] = {"choice": choice, "time": time.time()}
    if qid == "when": st["on"] = not choice.startswith("Not now")
    if qid.startswith("failed_") and choice.startswith("Try"): st["tries"].pop(qid[7:], None)
    if qid.startswith("failed_") and choice.startswith("Skip"): st["done"][qid[7:]] = {"skipped": True, "time": time.time()}
    save(st)
    return st


def pending_question(st):
    """the question to show now, if any"""
    for qid, q in (st.get("extra_questions") or {}).items():
        if qid not in st["answers"]: return {"id": qid, **q}
    for sid, _, need, _ in STEPS:
        if sid in st["done"]: continue
        if need and need not in st["answers"]: return {"id": need, **QUESTIONS[need]}
        if need and st["answers"][need]["choice"].startswith("Not now"):
            if time.time() - st["answers"][need]["time"] > 24 * 3600: st["answers"].pop(need); save(st); return {"id": need, **QUESTIONS[need]}
        return None
    return None


def next_step(st):
    """-> (step id, label, command) or (None, reason, None)"""
    if not st.get("on", True): return None, "paused", None
    for i, (sid, label, need, cmd) in enumerate(STEPS):
        if sid in st["done"]: continue
        if need and (need not in st["answers"] or st["answers"][need]["choice"].startswith("Not now")):
            return None, "waiting for your answer", None
        if st["tries"].get(sid, 0) >= MAX_TRIES: return None, f"'{label}' failed twice", None
        return sid, f"{label} (step {i + 1} of {len(STEPS)})", cmd
    return None, "all done", None


def may_start(st, idle_s, hour=None):
    when = (st["answers"].get("when") or {}).get("choice", "")
    if idle_s < IDLE_MIN * 60: return False
    if when.startswith("Only at night"):
        h = time.localtime().tm_hour if hour is None else hour
        return h >= NIGHT[0] or h < NIGHT[1]
    return True


def finished(sid, ok, tail=""):
    """called by the app when a step's job ends"""
    st = load()
    label = next(l for s, l, _, _ in STEPS if s == sid)
    if ok:
        st["done"][sid] = {"time": time.time()}
        news(st, f"Done: {label}.", "a")
        if sid == "choose":
            c = json.load(open(os.path.join("models", "base_choice.json")))
            if c.get("winner"):
                news(st, f"I chose {c['winner']} as the base model: the best total on the validation tests (game requests after training, "
                         f"learning speed, JavaScript, instructions, knowledge, languages, HTML, pictures) while staying quick with the playtesters running. "
                         f"To use a different one, tell Claude.", "a")
        if sid == "release":
            news(st, "The first release of your new AI is ready, tested and working offline. The setup steps are finished: "
                     "send Claude the files in bench\\phase0 and models\\releases so the next phase can start.", "a")
    else:
        st["tries"][sid] = st["tries"].get(sid, 0) + 1
        if st["tries"][sid] >= MAX_TRIES:
            st["answers"].pop(f"failed_{sid}", None)  # ask again, even if it was answered before
            st.setdefault("extra_questions", {})[f"failed_{sid}"] = {
                "text": f"'{label}' didn't work twice{(': ' + tail[-200:]) if tail else ''}. What should I do?",
                "choices": ["Try again", "Skip this step", "Wait for Claude to fix it"]}
        else:
            news(st, f"'{label}' didn't finish; I'll try it again later.", "w")
    save(st)


def status():
    st = load()
    sid, label, _ = next_step(st)
    q = pending_question(st)
    return {"on": st.get("on", True), "step": sid, "label": label, "question": q, "news": st.get("news", [])[-10:],
            "done": len(st["done"]), "total": len(STEPS)}


# ---------- the steps that aren't Phase 0 commands ----------

def run(sid):
    st = load()
    if sid == "general":
        from . import general_data
        print(general_data.run(human_only=st["answers"]["general"]["choice"].startswith("Only")))
    elif sid == "bakeoff":
        from .phase0 import bakeoff
        short = st["answers"]["bakeoff"]["choice"].startswith("Only")
        bakeoff.run(["qwen3.5-2b", "qwen3.5-4b"] if short else None, 30)
    elif sid == "choose":
        from .phase0 import choose
        c = choose.choose()
        print(json.dumps({k: c.get(k) for k in ("winner", "ranking", "why_not")}, indent=1)[:3000])
        if not c["winner"]:
            raise SystemExit("no candidate could be chosen: " + json.dumps(c["why_not"])[:300])
    elif sid == "install":
        from .phase0 import choose
        print(choose.install())
    elif sid == "mix":
        mix_sweep()
    elif sid == "adapter":
        best = (load().get("mix") or {}).get("best_arm") or {"share": 0.3, "source": "data"}
        _train("game_design", 120, best["share"], best["source"])
    elif sid == "release":
        from . import release
        a = release.build("A"); release.score(a)
        b = release.build("B"); release.score(b)
        pick, why = release.compare(a, b); print(why)
        m = release.finish(pick["id"])
        if m.get("blocked"): raise SystemExit(m["blocked"])
    else:
        raise SystemExit(f"unknown step {sid}")


def _train(skill, minutes, share, source="data"):
    import subprocess
    from . import registry
    hf = registry.load()["entries"]["core_hf"]["path"]
    r = subprocess.run([sys.executable, "-u", "-m", "gameai.lora_train", "--skill", skill, "--base", hf, "--minutes", str(minutes), "--general", str(share),
                        "--general-source", source]
                       + (["--resume"] if os.path.exists(os.path.join("models", "adapters", skill, "training", "trainer_state.pt")) else []))
    if r.returncode: raise SystemExit(f"training {skill} failed")


ARMS = [("data", 0.2), ("data", 0.3), ("data", 0.4), ("data", 0.5), ("self", 0.3)]


def make_self_answers(n=2000):
    """The base model's own answers to n everyday questions from the general training split (endogenous replay)."""
    from .lora_train import SELF_ANSWERS
    from . import registry, splits
    from .phase0 import runtime as R
    if os.path.exists(SELF_ANSWERS) and sum(1 for _ in open(SELF_ANSWERS, encoding="utf-8")) >= n * 0.9: return
    rows = [r for r in splits.rows("general", "train") if r.get("messages")][:n]
    q4 = registry.load()["entries"]["core"]["path"]
    with R.Server(q4) as srv, open(SELF_ANSWERS, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            msgs = [m for m in r["messages"] if m["role"] != "assistant"][:2]
            if not any(m["role"] == "user" for m in msgs): continue
            msgs = msgs[:[k for k, m in enumerate(msgs) if m["role"] == "user"][0] + 1]
            try: text, _ = srv.chat(msgs, max_tokens=400, thinking=False, temperature=0.7)
            except Exception: continue
            if text.strip(): f.write(json.dumps({"messages": msgs + [{"role": "assistant", "content": text}]}, ensure_ascii=False) + "\n")
            if i % 100 == 0: print(f"  base model answered {i} of {len(rows)} everyday questions", flush=True)


def mix_sweep():
    """The plan: short runs at 20, 30, 40 and 50% general data, plus one with the base model's own answers (research:
    endogenous replay protects general ability best), compared on validation; the best is used for the real run."""
    from . import adapters
    st = load(); res = st.get("mix", {}).get("runs", {})
    for source, share in ARMS:
        key = f"{source}:{share}"
        if key in res: continue
        if source == "self": make_self_answers()
        _train("mix_test", 20, share, source)
        v = adapters.versions("mix_test")[-1]
        res[key] = {"total": v["total"], "scores": v["scores"], "source": source, "share": share}
        st = load(); st["mix"] = {"runs": res}; save(st)
        print(f"{'own answers' if source == 'self' else 'dataset answers'} at {share:.0%}: validation total {v['total']}", flush=True)
    best = max(res.values(), key=lambda r: r["total"])
    st = load(); st["mix"] = {"runs": res, "best_arm": {"share": best["share"], "source": best["source"]}}; save(st)
    import shutil; shutil.rmtree(os.path.join("models", "adapters", "mix_test"), ignore_errors=True)  # only the result is kept
    print(f"best: {best['source']} answers at {best['share']:.0%}")


def main():
    a = sys.argv[1:]
    if a[:1] == ["run"] and len(a) > 1: run(a[1])
    else: print(json.dumps(status(), indent=1))


if __name__ == "__main__":
    main()
