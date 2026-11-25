"""The base-model bake-off, one model at a time (lean storage):

  record licence + size  ->  download (pinned revision)  ->  30-minute QLoRA test (training format)  ->
  convert to GGUF + compress to 4-bit (the release format)  ->  every test in llama.cpp (the real offline runtime),
  also with thinking off/on, with the playtesters running at the same time, and with fast drafting (MTP) if the
  runtime supports it  ->  save the results  ->  delete the model  ->  next.

Results: bench/phase0/bakeoff/<model>.json and bench/phase0/bakeoff/summary.md. A finished model is skipped when you
run it again (delete its .json to redo it). Only validation material is used; the sealed sets never choose a model.
"""
import glob, json, os, shutil, time

from . import candidates as C, runtime as R, tasks as T
from .common import OUT, need_space, nvidia_smi, say, stage, progress

DIR = os.path.join(OUT, "bakeoff")
WORK = os.path.join("models", "bakeoff")  # models/ is never touched by updates


def thinking_supported(hf_dir):
    try:
        cfg = json.load(open(os.path.join(hf_dir, "tokenizer_config.json"), encoding="utf-8"))
        tpl = cfg.get("chat_template") or ""
        tpl += "".join(open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(hf_dir, "*.jinja")))
        return "enable_thinking" in tpl
    except Exception:
        return False


def game_ai_baseline():
    """Today's game AI on the same validation requests: the baseline every candidate is compared with."""
    import torch
    from ..generate import load, decode
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tok, _ = load("models/current.pt", dev, upgrade=True)
    except Exception as e:
        return {"note": f"no game AI to compare with ({e})"}
    items = T.recipe_items()
    t = time.perf_counter()
    preds = decode(model, tok, [it["prompt"] for it in items], dev)
    secs = time.perf_counter() - t
    from ..recipe import FIELD_NAMES
    out = T.score_recipes(items, [{f: p[f][0] for f in FIELD_NAMES} for p in preds])
    out["answer_s_avg"] = round(secs / len(items), 4)
    out["params_m"] = round(model.num_params() / 1e6, 2)
    return out


def under_load(srv, n_items=30):
    """The same requests while playtesters use most of the processor (the app at full load)."""
    import random
    from concurrent.futures import ProcessPoolExecutor
    from .fullstack import _playtest_job, _warm
    n = max(1, (os.cpu_count() or 2) - 2)
    items = T.recipe_items()[:n_items]
    with ProcessPoolExecutor(n) as pool:
        list(pool.map(_warm, range(n)))
        futs = [pool.submit(_playtest_job, k) for k in range(n * 50)]  # plenty of work for the whole test
        res = T.test_recipes(srv.chat, items, thinking=False)
        for f in futs: f.cancel()
    return {"playtesters_running": n, **{k: res[k] for k in ("answer_s_median", "answer_s_p95", "gen_tok_per_s", "fully_right")}}


def inference_tests(rec, hf_dir, gguf, mmproj, log=say):
    res = {}
    pipe = rec.get("pipeline") or ""
    with R.Server(gguf, mmproj=mmproj) as srv:
        time.sleep(2)
        smi = nvidia_smi("memory.used")
        res["vram_in_use_gb"] = round(float(smi[0]) / 1024, 2) if smi else None
        think = thinking_supported(hf_dir)
        tests = [("game requests (thinking off)", lambda: T.test_recipes(srv.chat, thinking=False if think else None))]
        if think: tests.append(("game requests (thinking on)", lambda: T.test_recipes(srv.chat, T.recipe_items()[:40], thinking=True)))
        tests += [("JavaScript", lambda: T.test_code(srv.chat)), ("instructions", lambda: T.test_instructions(srv.chat, thinking=False if think else None)),
                  ("languages", lambda: T.test_languages(srv.chat)), ("knowledge and reasoning", lambda: T.test_knowledge(srv.chat, thinking=False if think else None)),
                  ("interfaces (HTML)", lambda: T.test_ui(srv.chat))]
        if mmproj: tests.append(("pictures", lambda: T.test_vision(srv.chat)))
        if "audio" in json.dumps(rec).lower() or "gemma" in rec["key"]: tests.append(("sound", lambda: T.test_audio(srv.chat)))
        tests.append(("with the playtesters running", lambda: under_load(srv)))
        for i, (name, fn) in enumerate(tests):
            progress(i, len(tests)); log(f"  testing: {name}...")
            try: res[name] = fn()
            except Exception as e: res[name] = {"not measured": str(e)[:300]}
            log("   ", res[name])
    try:  # fast drafting with the model's own extra prediction heads (multi-token prediction), where supported
        with R.Server(gguf, extra=["--spec-type", "draft-mtp"]) as srv:
            res["fast drafting (MTP)"] = T.test_recipes(srv.chat, T.recipe_items()[:30], thinking=False)
    except Exception as e:
        res["fast drafting (MTP)"] = {"not supported": str(e)[:200]}
    return res


def one(rec, minutes, log=say):
    key = rec["key"]; work = os.path.join(WORK, key); hf = os.path.join(work, "hf")
    res = {"candidate": rec, "started": time.strftime("%Y-%m-%d %H:%M")}
    try:
        need_space(rec["download_gb"] * 2.2 + 1, f"testing {key}")  # the weights, plus their 16-bit converted copy for a moment
        stage(f"{key}: downloading {rec['download_gb']} GB")
        C.download(rec, hf)
        cand = C.by_key(key)
        import torch
        if cand["train"] and torch.cuda.is_available():
            stage(f"{key}: QLoRA training test ({minutes} minutes)")
            try: res["qlora"] = __import__("gameai.phase0.qlora", fromlist=["run"]).run(hf, minutes=minutes, adapter_out=os.path.join(work, "adapter"),
                                                                                         precision=cand.get("precision", "qlora"))
            except Exception as e: res["qlora"] = {"failed": str(e)[:500]}
        elif not cand["train"]:
            res["qlora"] = {"skipped": "measured as a teacher or reference only (QLoRA doesn't fit on 8 GB)"}
        else:
            res["qlora"] = {"skipped": "no NVIDIA GPU found"}
        stage(f"{key}: converting to the runtime's format and compressing to 4-bit")
        vision = "image" in (rec.get("pipeline") or "") or "vision" in (rec.get("pipeline") or "")
        f16, mm = R.to_gguf(hf, work, log=log, vision=vision)
        q4 = R.quantize(f16, os.path.join(work, "model-q4_k_m.gguf"))
        os.remove(f16)
        res["q4_gb"] = round(os.path.getsize(q4) / 2**30, 2)
        stage(f"{key}: tests in llama.cpp")
        res["tests"] = inference_tests(rec, hf, q4, mm, log)
        ad = os.path.join(work, "adapter")
        if os.path.isdir(ad):  # runtime + adapter compatibility: does llama.cpp load the trained adapter, and does it help there too?
            stage(f"{key}: the trained adapter in llama.cpp")
            try:
                lg = R.lora_to_gguf(ad, hf, os.path.join(work, "adapter.gguf"))
                with R.Server(q4, extra=["--lora", lg]) as srv:
                    think = thinking_supported(hf)
                    res["adapter_in_runtime"] = {"loads": True, "adapter_mb": round(os.path.getsize(lg) / 2**20, 1),
                                                 **T.test_recipes(srv.chat, T.recipe_items()[:60], thinking=False if think else None)}
                    # does it keep its general abilities after training? (research: models differ a lot here)
                    before = res["tests"]
                    ins = T.test_instructions(srv.chat, thinking=False if think else None)["follows_instructions"]
                    kno = T.test_knowledge(srv.chat, thinking=False if think else None)["knowledge_and_reasoning"]
                    b_ins = (before.get("instructions") or {}).get("follows_instructions"); b_kno = (before.get("knowledge and reasoning") or {}).get("knowledge_and_reasoning")
                    res["adapter_in_runtime"]["general_after_training"] = {"instructions": ins, "knowledge": kno}
                    if b_ins and b_kno:
                        res["adapter_in_runtime"]["general_kept"] = round(min(1.0, (ins + kno) / max(1e-9, b_ins + b_kno)), 4)
            except Exception as e:
                res["adapter_in_runtime"] = {"loads": False, "why": str(e)[:400]}
            log("   ", res["adapter_in_runtime"])
    except Exception as e:
        res["error"] = str(e)[:800]
    finally:
        shutil.rmtree(work, ignore_errors=True)  # lean storage: nothing of the model stays on disk
    res["finished"] = time.strftime("%Y-%m-%d %H:%M")
    os.makedirs(DIR, exist_ok=True)
    json.dump(res, open(os.path.join(DIR, f"{key}.json"), "w"), indent=1)
    return res


def g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d: return None
        d = d[p]
    return d


def summary():
    rows = []
    for p in sorted(glob.glob(os.path.join(DIR, "*.json"))):
        if p.endswith("baseline.json"): continue
        r = json.load(open(p)); t = r.get("tests") or {}; q = r.get("qlora") or {}
        gr = t.get("game requests (thinking off)") or {}
        rows.append([r["candidate"]["key"], r["candidate"].get("license"), r.get("q4_gb"),
                     g(q, "before", "fully_right"), g(q, "after", "fully_right"), g(q, "improvement_per_hour", "fully_right_gain"),
                     g(q, "memory", "max_context"), g(q, "memory", "max_batch_at_512"), g(q, "peak_gpu_gb"),
                     gr.get("fully_right"), gr.get("gen_tok_per_s"), gr.get("answer_s_median"),
                     g(t, "with the playtesters running", "answer_s_p95"), g(t, "JavaScript", "passes_sandbox"),
                     g(t, "instructions", "follows_instructions"), g(t, "knowledge and reasoning", "knowledge_and_reasoning"),
                     g(t, "languages", "languages"), g(t, "interfaces (HTML)", "ui_html"), g(t, "pictures", "vision"), g(t, "sound", "audio"),
                     t.get("vram_in_use_gb"), "yes" if g(t, "fast drafting (MTP)", "fully_right") is not None else "no",
                     ("loads, " + str(g(r, "adapter_in_runtime", "fully_right")) + " right") if g(r, "adapter_in_runtime", "loads") else ("no" if r.get("adapter_in_runtime") else ""),
                     (r.get("error") or g(q, "failed") or "")[:60]])
    head = ["model", "licence", "4-bit GB", "trained: right before", "after 30 min", "gain per hour", "max context", "max batch", "training GPU GB",
            "untrained: right", "words/s", "answer s", "answer s under load (slowest 5%)", "JS passes", "instructions", "knowledge", "languages",
            "UI", "pictures", "sound", "VRAM GB", "MTP", "trained adapter in llama.cpp", "problem"]
    base = json.load(open(os.path.join(DIR, "baseline.json"))) if os.path.exists(os.path.join(DIR, "baseline.json")) else {}
    md = ["# Base-model bake-off", "", f"Today's game AI on the same requests: {base.get('fully_right')} fully right, {base.get('answer_s_avg')} s per answer.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    md += ["", "All scores are 0 to 1 on validation tasks (never the sealed sets). \"right\" = all settings of a game request right."]
    os.makedirs(DIR, exist_ok=True)
    open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return "\n".join(md)


def run(keys=None, minutes=30, redo=False):
    stage("Bake-off: licences and sizes (nothing is downloaded yet)")
    recs = C.record_all()
    stage("Bake-off: getting the llama.cpp runtime")
    say("  ", R.install())
    os.makedirs(DIR, exist_ok=True)
    if redo or not os.path.exists(os.path.join(DIR, "baseline.json")):
        stage("Bake-off: today's game AI as the baseline")
        b = game_ai_baseline(); say("  ", b); json.dump(b, open(os.path.join(DIR, "baseline.json"), "w"), indent=1)
    todo = [r for r in recs if r["allowed"] and (not keys or r["key"] in keys)]
    for i, rec in enumerate(todo):
        if not redo and os.path.exists(os.path.join(DIR, f"{rec['key']}.json")):
            say(f"{rec['key']}: already tested, skipping (delete its file in {DIR} to redo it)"); continue
        say(f"\n[{i + 1}/{len(todo)}] {rec['key']}")
        one(rec, minutes)
        summary()
    stage("Bake-off finished")
    print(summary())
