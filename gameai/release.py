"""Making a release of the AI (see docs/RELEASE.md). Steps, in order:

  1. build    configuration A: the 4-bit base + each skill's adapter as its own small file (the default), or
              configuration B: the adapters merged into the full-precision base, then compressed to 4-bit once
  2. compare  both on VALIDATION (the compressed files, in llama.cpp, exactly as they'll be used); B is kept only
              if it beats A. The choice is then locked.
  3. sealed   only now, the sealed sets measure the locked release (totals only); this never changes the choice
  4. offline  the independence test must pass
  5. bundle   weights/adapters, tokenizer, config, licence and change log with SHA-256 of every file;
              the last 3 releases are kept (lean storage)

    python -m gameai.release build A        (or B)
    python -m gameai.release compare
    python -m gameai.release finish <release id>
    python -m gameai.release list
"""
import glob, hashlib, json, os, shutil, sys, time

from . import adapters, registry
from .phase0 import runtime as R, tasks as T

DIR = os.path.join("models", "releases")
KEEP = 3


def _rid(cfg): return time.strftime("%Y%m%d-%H%M") + "-" + cfg


def core():
    """the base model entries: 4-bit GGUF for use, Hugging Face folder for training and merging"""
    e = registry.load()["entries"]
    q4 = e.get("core"); hf = e.get("core_hf")
    if not q4 or not hf: raise RuntimeError("no base model registered yet (it's chosen by the Phase 0 bake-off)")
    return q4, hf


def build(cfg, skills=None, log=print):
    q4, hf = core()
    skills = skills or [s for s in adapters.skills() if adapters.active(s)]
    rid = _rid(cfg); out = os.path.join(DIR, rid); os.makedirs(out, exist_ok=True)
    man = {"id": rid, "config": cfg, "base": {"path": q4["path"], "version": q4["version"], "licence": q4["licence"]}, "skills": {}, "created": time.strftime("%Y-%m-%d %H:%M")}
    if cfg == "A":
        for s in skills:
            v = adapters.active(s)["version"]; src = adapters._dir(s, v)
            g = os.path.join(src, "adapter.gguf")
            if not os.path.exists(g): R.lora_to_gguf(src, hf["path"], g)
            dst = os.path.join(out, f"{s}-v{v:03d}.gguf"); shutil.copy2(g, dst)
            man["skills"][s] = {"version": v, "file": os.path.basename(dst)}
        man["model"] = q4["path"]
    elif cfg == "B":
        from .phase0.common import need_space
        q4_gb = os.path.getsize(q4["path"]) / 2**30
        need_space(9 * q4_gb + 2, "merging the adapters")  # full-precision copy + its 16-bit GGUF, each about 4x the 4-bit file
        merged = os.path.join(out, "merged-hf")
        _merge(hf["path"], [adapters._dir(s, adapters.active(s)["version"]) for s in skills], merged)
        f16, _ = R.to_gguf(merged, out, log=log)
        model = R.quantize(f16, os.path.join(out, "merged-q4_k_m.gguf"))
        os.remove(f16); shutil.rmtree(merged, ignore_errors=True)  # lean storage: only the compressed file stays
        man["model"] = model
        man["skills"] = {s: {"version": adapters.active(s)["version"], "merged": True} for s in skills}
    else:
        raise ValueError("configuration must be A or B")
    json.dump(man, open(os.path.join(out, "release.json"), "w"), indent=1)
    return man


def _merge(hf_dir, adapter_dirs, out):
    """Adapters trained against the 4-bit base, merged into the full-precision base (so the result is measured,
    not assumed equal to what was trained)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float16, device_map={"": "cpu"})
    for d in adapter_dirs:
        model = PeftModel.from_pretrained(model, d).merge_and_unload()
    model.save_pretrained(out); AutoTokenizer.from_pretrained(hf_dir).save_pretrained(out)


def _server_for(man, skill=None):
    """How the release runs in llama.cpp: configuration A loads the base plus the skill's adapter (the router picks one
    per step); B is a single file."""
    extra = []
    if man["config"] == "A" and skill and skill in man["skills"]:
        extra = ["--lora", os.path.join(DIR, man["id"], man["skills"][skill]["file"])]
    return R.Server(man["model"], extra=extra)


def score(man):
    """Validation scores of the compressed release, in the runtime."""
    out = {}
    with _server_for(man, "game_design") as srv:
        r = T.test_recipes(srv.chat, T.recipe_items(), thinking=False)
        out["game_requests"] = {"score": r["fully_right"], "n": r["n"]}
        out["speed"] = {k: r[k] for k in ("answer_s_median", "answer_s_p95", "gen_tok_per_s")}
        out["instructions"] = {"score": T.test_instructions(srv.chat, thinking=False)["follows_instructions"], "n": len(T.INSTR)}
        out["knowledge"] = {"score": T.test_knowledge(srv.chat, thinking=False)["knowledge_and_reasoning"], "n": len(T.QA)}
    man["validation"] = out
    json.dump(man, open(os.path.join(DIR, man["id"], "release.json"), "w"), indent=1)
    return out


def compare(a, b):
    """-> the chosen manifest. B only if it is better overall on validation and not clearly worse anywhere."""
    sa = {k: v for k, v in a["validation"].items() if "score" in v}; sb = {k: v for k, v in b["validation"].items() if "score" in v}
    bad = adapters.clearly_worse(sb, sa)
    pick = b if not bad and adapters.total(sb) > adapters.total(sa) else a
    why = ("merged (B) is better on validation" if pick is b else
           f"separate adapters (A) kept: merged is {'clearly worse on ' + ', '.join(bad) if bad else 'not better overall'}")
    for m in (a, b):
        m["chosen"] = m is pick; m["why"] = why
        json.dump(m, open(os.path.join(DIR, m["id"], "release.json"), "w"), indent=1)
    if pick is not b: shutil.rmtree(os.path.join(DIR, b["id"]), ignore_errors=True)  # lean storage
    else: shutil.rmtree(os.path.join(DIR, a["id"]), ignore_errors=True)
    return pick, why


def finish(rid, log=print):
    """Lock, measure on the sealed sets (totals only), run the independence test, write the bundle list."""
    p = os.path.join(DIR, rid, "release.json"); man = json.load(open(p))
    if man.get("locked"): raise RuntimeError("this release is already finished")
    man["locked"] = time.strftime("%Y-%m-%d %H:%M")
    from . import sealed
    man["sealed"] = {}
    for name in sealed.summary():
        try: man["sealed"][name] = {k: v for k, v in sealed.run(name, model_name="release:" + rid).items() if k in ("score", "n", "set_version")}
        except Exception as e: man["sealed"][name] = {"not measured": str(e)[:200]}
    from .phase0 import independence
    man["independence_passed"] = independence.run()["passed"]
    if not man["independence_passed"]: man["blocked"] = "the independence test failed: this release can't be used"
    files = {}
    for f in [man["model"]] + [os.path.join(DIR, rid, s["file"]) for s in man["skills"].values() if s.get("file")]:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(2**20), b""): h.update(chunk)
        files[f] = h.hexdigest()
    man["files"] = files
    json.dump(man, open(p, "w"), indent=1)
    prune()
    return man


def prune():
    rel = sorted(d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d))) if os.path.isdir(DIR) else []
    for d in rel[:-KEEP]: shutil.rmtree(os.path.join(DIR, d), ignore_errors=True)


def releases():
    out = []
    for p in sorted(glob.glob(os.path.join(DIR, "*", "release.json"))):
        out.append(json.load(open(p)))
    return out


def main():
    a = sys.argv[1:]
    if a[:1] == ["build"]: m = build(a[1] if len(a) > 1 else "A"); print(m["id"], score(m))
    elif a[:1] == ["compare"]:
        rs = [r for r in releases() if not r.get("locked") and r.get("validation")]
        A = next((r for r in reversed(rs) if r["config"] == "A"), None); B = next((r for r in reversed(rs) if r["config"] == "B"), None)
        if not (A and B): print("build and score both A and B first"); return
        pick, why = compare(A, B); print(pick["id"], why)
    elif a[:1] == ["finish"]: print(json.dumps(finish(a[1]), indent=1)[:2000])
    else:
        for r in releases(): print(r["id"], r["config"], r.get("validation"), "locked" if r.get("locked") else "", r.get("blocked", ""))


if __name__ == "__main__":
    main()
