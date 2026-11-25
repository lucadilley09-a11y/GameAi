"""The coding teacher (plan: Phase 4): "the strongest free coding model that runs here, every example checked in the
sandbox". It is chosen by measurement, not by name: each candidate solves the validation tasks once (first try, no
retries) and the best pass rate wins (ties: the faster one). If the AI itself is already far ahead of every candidate,
there is no teacher and self-teaching does it alone ("or base if far ahead").

Only one teacher is on disk at a time: each candidate is downloaded, measured and deleted unless it's the best so far.
Teachers help only while learning; the finished AI never needs them.

What the teacher does: writes new training tasks (each checked like the hand-written ones: its own solution must pass the
visible and hidden tests, and the tests must fail without a solution), and solves training tasks the AI can't yet; every
solution goes through the same verifier as the AI's own.

Candidates (permissive licences only; the licence is checked again before downloading):
    qwen2.5-coder-7b   a coding specialist, about 4.7 GB compressed
    qwen3.5-9b         a newer general model, strong at code for its size, about 5.5 GB compressed
    (if converting is needed and there isn't room for it, that candidate is skipped, never your other files)
    qwen3.5-4b         the current learning teacher, about 2.7 GB compressed
"""
import glob, json, os, random, re, shutil, time

from . import prompts
from . import tasks as T
from .prompts import text_of_reply as _text

DIR = os.path.join("models", "teachers", "coding")
RESULT = os.path.join(T.DIR, "teacher.json")
FAR_AHEAD = 0.10   # the AI itself beats every candidate by this much: no teacher
CANDIDATES = [  # a ready-compressed copy is used when one exists (saves converting, which briefly needs ~40 GB for the 9B)
    {"key": "qwen2.5-coder-7b", "gguf_repos": ["Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"], "gguf": "*q4_k_m*.gguf"},
    {"key": "qwen3.5-9b", "gguf_repos": ["Qwen/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF"], "gguf": "*q4_k_m*.gguf", "hf_repo": "Qwen/Qwen3.5-9B"},
    {"key": "qwen3.5-4b", "gguf_repos": ["Qwen/Qwen3.5-4B-GGUF", "unsloth/Qwen3.5-4B-GGUF"], "gguf": "*q4_k_m*.gguf", "hf_repo": "Qwen/Qwen3.5-4B"},
]


def _licence(repo):
    from ..phase0.candidates import record, ALLOWED
    rec = record({"key": repo, "repos": [repo], "role": "coding teacher"})
    return rec


def fetch(c, log=print):
    """download (and if needed convert and compress) one candidate -> path of its .gguf"""
    from ..phase0 import runtime as R
    from ..phase0.common import need_space
    dest = os.path.join(DIR, c["key"])
    have = sorted(glob.glob(os.path.join(dest, "*.gguf")))
    if have: return have[0]
    from huggingface_hub import snapshot_download, HfApi
    import fnmatch
    for repo in c.get("gguf_repos", []):  # a ready-compressed copy, if one exists with an allowed licence
        rec = _licence(repo)
        if not rec.get("allowed"): continue
        files = [s.rfilename for s in HfApi().model_info(repo, revision=rec["revision"]).siblings]
        pick = sorted(f for f in files if fnmatch.fnmatch(f.lower(), c["gguf"]))
        if not pick: continue
        need_space(8, c["key"])
        log(f"  downloading {c['key']} from {repo} ({len(pick)} file(s))...")
        snapshot_download(repo, revision=rec["revision"], local_dir=dest, allow_patterns=pick + ["LICENSE*"])
        json.dump(rec, open(os.path.join(dest, "provenance.json"), "w"), indent=1)
        return sorted(glob.glob(os.path.join(dest, "**", "*.gguf"), recursive=True))[0]
    if not c.get("hf_repo"): raise RuntimeError(f"{c['key']}: no compressed copy with an allowed licence was found")
    rec = _licence(c["hf_repo"])
    if not rec.get("allowed"): raise RuntimeError(f"{c['key']}: {rec.get('why_not')}")
    # Hugging Face weights -> 16-bit GGUF -> 4-bit; the big intermediate files are deleted
    need_space(rec.get("download_gb", 10) * 2.2 + 6, c["key"])
    from ..phase0.candidates import download
    hf = os.path.join(dest, "hf")
    log(f"  downloading {c['key']} ({rec.get('download_gb', '?')} GB, converted once, then the original is deleted)...")
    download(rec, hf)
    f16, _ = R.to_gguf(hf, dest, log=log)
    out = os.path.join(dest, f"{c['key']}-Q4_K_M.gguf")
    R.quantize(f16, out)
    os.remove(f16); shutil.rmtree(hf, ignore_errors=True)
    json.dump(rec, open(os.path.join(dest, "provenance.json"), "w"), indent=1)
    return out


def measure(chat, task_list, log=print):
    """first-try pass rate on the validation tasks, and answer speed"""
    from .verify import verify
    ok, secs, by_lang = 0, 0.0, {}
    for t in task_list:
        t0 = time.time()
        try: text = _text(chat(prompts.messages(t), max_tokens=1500, thinking=False, temperature=0.0))
        except Exception as e: text = f"(no answer: {e})"
        secs += time.time() - t0
        good = verify(t, text)["accepted"]
        ok += good
        b = by_lang.setdefault(t["lang"], [0, 0]); b[0] += good; b[1] += 1
    return {"pass_rate": round(ok / max(1, len(task_list)), 4), "seconds_per_task": round(secs / max(1, len(task_list)), 2),
            "by_language": {k: f"{a}/{n}" for k, (a, n) in by_lang.items()}}


def choose(core_chat=None, log=print):
    """measure every candidate (and the AI itself), keep the best on disk -> the saved result"""
    from ..phase0 import runtime as R
    val = T.load("val")
    res = {"measured": time.strftime("%Y-%m-%d %H:%M"), "tasks": len(val), "candidates": {}}
    if core_chat:
        log("measuring the AI itself on the coding tasks...")
        res["self"] = measure(core_chat, val, log)
    best = None
    for c in CANDIDATES:
        try:
            gguf = fetch(c, log)
            log(f"measuring {c['key']}...")
            with R.Server(gguf, ctx=8192, port=8792) as srv:
                m = measure(srv.chat, val, log)
            m["gguf"] = gguf
        except Exception as e:
            m = {"error": str(e)[:300]}
        res["candidates"][c["key"]] = m
        log(f"  {c['key']}: " + (f"{m['pass_rate']:.0%} solved first try, {m['seconds_per_task']}s per task" if "pass_rate" in m else m["error"]))
        if "pass_rate" in m and (best is None or (m["pass_rate"], -m["seconds_per_task"]) > (res["candidates"][best]["pass_rate"], -res["candidates"][best]["seconds_per_task"])):
            if best: shutil.rmtree(os.path.join(DIR, best), ignore_errors=True)  # one teacher on disk at a time
            best = c["key"]
        else:
            shutil.rmtree(os.path.join(DIR, c["key"]), ignore_errors=True)
    if best and "self" in res and res["self"]["pass_rate"] >= res["candidates"][best]["pass_rate"] + FAR_AHEAD:
        log("the AI itself is far ahead of every candidate: no coding teacher needed")
        shutil.rmtree(os.path.join(DIR, best), ignore_errors=True); best = None
    res["winner"] = best
    os.makedirs(T.DIR, exist_ok=True)
    json.dump(res, open(RESULT, "w"), indent=1)
    return res


def winner():
    try:
        r = json.load(open(RESULT))
        return r["candidates"][r["winner"]]["gguf"] if r.get("winner") else None
    except (OSError, ValueError, KeyError):
        return None


# ---------- writing new tasks ----------
TOPICS = {
    "js": ["enemy movement", "collision between boxes and circles", "a score and combo system", "an inventory", "a save/load format",
           "tile map helpers", "path finding on a grid", "a timer or cooldown", "random level generation with a seed", "a health and damage system",
           "camera following the player", "input handling with key states", "a state machine for an enemy", "particle effects data",
           "a GameKit game with a twist", "a GameKit rule that spawns things over time", "a GameKit custom enemy brain", "fixing a bug in game code",
           "a dialogue/quest list", "physics: velocity, gravity and jumping", "sorting and ranking high scores", "spawning waves of enemies"],
    "py": ["level file parsing", "a save format", "high-score tables", "a tile map", "path finding", "a damage formula with rules", "an inventory",
           "dice and random loot with a seed", "text adventure commands", "fixing a bug in game code", "a timer/scheduler", "statistics of a level"],
    "lua": ["a cooldown or timer object", "a state machine", "an inventory module", "weighted random loot", "vector maths helpers", "a simple class with methods",
            "an event system (subscribe/emit)", "string formatting for a HUD", "a queue or stack", "fixing a bug in Lua game code", "grid helpers"],
}
EXAMPLE_IDS = {"js": "js-inventory", "py": "py-high-scores", "lua": "lua-cooldown"}


def task_prompt(lang, topic, example):
    ex = {k: example[k] for k in ("prompt", "file", "starter", "tests", "hidden", "reference")}
    rules = {"js": "Tests use test(name, fn), expect(x).toBe(y) / toEqual / toBeTruthy / toBeGreaterThan / toContain / toThrow. "
                   "Use ES module exports and relative imports, or plain scripts that use the GameKit global.",
             "py": "Tests are test_* functions with plain assert, in files named test_<module>.py and test_<module>_hidden.py.",
             "lua": "Tests use test('name', function() expect(x).toBe(y) end) (also toEqual, toBeTruthy, toContain), in files <module>_test.lua and <module>_hidden_test.lua; modules return a table and are loaded with require('<module>')."}[lang]
    return [{"role": "system", "content": "You write small, precise programming exercises for game developers, as JSON."},
            {"role": "user", "content": f"Write one new {prompts.LANG_NAME[lang]} exercise about: {topic}.\n{rules}\n"
             "The visible tests show the idea; the hidden tests (at least 2) check the same behaviour on other inputs and edge cases, "
             "so code that only fits the visible tests fails. Give a correct reference solution. If the task is a bug fix, put the "
             "buggy file in starter. Reply with JSON only, with exactly these keys, like this example:\n" + json.dumps(ex, indent=1)}]


def parse_task(text, lang):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except ValueError: return None
    if not all(isinstance(d.get(k), (str, dict)) for k in ("prompt", "file", "tests", "hidden", "reference")): return None
    for k in ("starter", "tests", "hidden"):
        if not isinstance(d.get(k, {}), dict) or not all(isinstance(v, str) for v in d.get(k, {}).values()): return None
    d["starter"] = d.get("starter") or {}
    d["lang"] = lang
    ext = {"js": (".js", ".mjs"), "py": (".py",), "lua": (".lua",)}[lang]
    if not all(f.endswith(ext) for f in [d["file"], *d["tests"], *d["hidden"], *d["starter"]]): return None  # wrong language
    if any(".." in f or f.startswith(("/", "\\")) for f in [d["file"], *d["tests"], *d["hidden"], *d["starter"]]): return None
    return d


def make_tasks(chat, n, log=print, stop=lambda: False, rng=None):
    """the teacher writes up to n new tasks; only checked, non-duplicate ones are kept -> how many were kept"""
    from ..guard import Index
    from .verify import similarity
    seeds = {t["id"]: t for t in T.load("val")}
    existing = T.load("val") + T.load("train")
    ix = Index()
    for t in T.load("val"): ix.add(t["prompt"])
    rng, kept, tried = rng or random.Random(), 0, 0
    langs = ["js"] * 5 + ["py"] * 2 + ["lua"] * 3   # mostly the language the games use
    while kept < n and tried < n * 4 and not stop():
        tried += 1
        lang = rng.choice(langs); topic = rng.choice(TOPICS[lang])
        try: text = _text(chat(task_prompt(lang, topic, seeds[EXAMPLE_IDS[lang]]), max_tokens=3000, thinking=False, temperature=0.9))
        except Exception: continue
        task = parse_task(text, lang)
        if not task: continue
        if ix.match(task["prompt"])[0]: continue                       # too close to a validation task
        if any(similarity(lang, task["reference"], e["reference"]) > 0.9 for e in existing if e["lang"] == lang): continue
        task.update(id=f"t-{lang}-{int(time.time() * 1000) % 10**10}-{kept}", source="teacher",
                    split="val" if rng.random() < T.VAL_SHARE else "train")
        if T.check_task(task): continue                                 # the reference must pass, the tests must test
        T.add(task); existing.append(task); kept += 1
        if task["split"] == "val": ix.add(task["prompt"])
        if kept % 10 == 0: log(f"  {kept} new tasks ({tried} written)")
    return kept
