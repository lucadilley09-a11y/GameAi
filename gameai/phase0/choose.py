"""Choosing the base model from the bake-off, automatically, with one fixed rule (validation results only; the
sealed sets never choose). You can always overrule it: /base <model>.

A candidate can win only if:
  - it trained on this PC (the 30-minute QLoRA test worked) and its trained adapter loads in llama.cpp,
  - it stays responsive with the playtesters running (slowest 5% of answers under RESPONSIVE_S seconds).
Score (0 to 1), weights written here so anyone can check them:
"""
import glob, json, os

from .common import OUT, say

RESPONSIVE_S = 3.0
WEIGHTS = {  # what matters most for this AI: reading game requests after training, then learning speed, coding, the rest
    "game requests after 30 min training": 0.30,
    "learning speed (vs the best candidate)": 0.15,
    "JavaScript": 0.15,
    "general ability kept after training": 0.10,  # research: models forget very differently when fine-tuned
    "instructions": 0.08,
    "knowledge and reasoning": 0.08,
    "languages": 0.05,
    "interfaces (HTML)": 0.05,
    "pictures": 0.04,
}
TIE = 0.02  # closer than this: the faster, smaller one wins
CHOICE = os.path.join("models", "base_choice.json")


def _g(d, *p):
    for k in p:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d


def load_results():
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, "bakeoff", "*.json"))):
        if os.path.basename(p) in ("baseline.json",): continue
        out.append(json.load(open(p)))
    return out


def choose(results=None):
    results = results if results is not None else load_results()
    rows, why_not = [], {}
    gains = [max(0.0, _g(r, "qlora", "improvement_per_hour", "fully_right_gain") or 0.0) for r in results]
    best_gain = max(gains + [1e-9])
    for r, gain in zip(results, gains):
        k, t = r["candidate"]["key"], r.get("tests") or {}
        after = _g(r, "qlora", "after", "fully_right")
        p95 = _g(t, "with the playtesters running", "answer_s_p95")
        if r.get("error"): why_not[k] = "the test didn't finish: " + r["error"][:80]; continue
        if after is None: why_not[k] = "it couldn't be trained on this PC"; continue
        if not _g(r, "adapter_in_runtime", "loads"): why_not[k] = "its trained adapter didn't load in llama.cpp"; continue
        if p95 is None or p95 > RESPONSIVE_S: why_not[k] = f"too slow with the playtesters running ({p95} s)"; continue
        parts = {"game requests after 30 min training": after, "learning speed (vs the best candidate)": gain / best_gain,
                 "JavaScript": _g(t, "JavaScript", "passes_sandbox") or 0, "instructions": _g(t, "instructions", "follows_instructions") or 0,
                 "knowledge and reasoning": _g(t, "knowledge and reasoning", "knowledge_and_reasoning") or 0,
                 "languages": _g(t, "languages", "languages") or 0, "interfaces (HTML)": _g(t, "interfaces (HTML)", "ui_html") or 0,
                 "pictures": _g(t, "pictures", "vision") or 0,
                 "general ability kept after training": _g(r, "adapter_in_runtime", "general_kept") or 0}
        rows.append({"key": k, "score": round(sum(WEIGHTS[n] * v for n, v in parts.items()), 4), "parts": parts,
                     "speed_s": _g(t, "game requests (thinking off)", "answer_s_median") or 99, "q4_gb": r.get("q4_gb") or 99, "record": r["candidate"]})
    if not rows: return {"winner": None, "why_not": why_not, "rule": WEIGHTS}
    rows.sort(key=lambda x: -x["score"])
    top = [x for x in rows if x["score"] >= rows[0]["score"] - TIE]
    win = min(top, key=lambda x: (x["speed_s"], x["q4_gb"]))
    res = {"winner": win["key"], "record": win["record"], "ranking": [{k: v for k, v in x.items() if k != "record"} for x in rows],
           "why_not": why_not, "rule": WEIGHTS, "tie_rule": f"within {TIE} of the best, the faster then smaller one wins",
           "how": "chosen automatically on validation results"}
    os.makedirs(os.path.dirname(CHOICE), exist_ok=True)
    json.dump(res, open(CHOICE, "w"), indent=1)
    return res


def install(key=None, log=say):
    """Download the chosen base again (pinned revision), keep its training format (Hugging Face) and its
    4-bit release format (GGUF), and register both. About 2x the model's size on disk."""
    from . import candidates as C, runtime as R
    from .. import registry
    from .common import need_space
    choice = json.load(open(CHOICE)) if os.path.exists(CHOICE) else {}
    key = key or choice.get("winner")
    if not key: raise RuntimeError("no base model chosen yet")
    rec = choice.get("record") if choice.get("winner") == key else C.record(C.by_key(key))
    if not rec.get("allowed", True): raise RuntimeError(rec.get("why_not"))
    need_space(rec["download_gb"] * 2.3 + 2, "installing the base model")
    R.install(log)
    base = os.path.join("models", "base", key); hf = os.path.join(base, "hf")
    log(f"Downloading {key} ({rec['download_gb']} GB)..."); C.download(rec, hf)
    vision = "image" in (rec.get("pipeline") or "")
    f16, mm = R.to_gguf(hf, base, log=log, vision=vision)
    q4 = R.quantize(f16, os.path.join(base, "model-q4_k_m.gguf")); os.remove(f16)
    res = next((r for r in load_results() if r["candidate"]["key"] == key), {})
    vram = _g(res, "tests", "vram_in_use_gb") or round(os.path.getsize(q4) / 2**30 + 1.2, 2)
    registry.register("core", "base", "core model: chat, understanding, planning, code", q4, vram, rec["license"], rec["revision"][:10], "gguf",
                      repo=rec["repo"], mmproj=mm)
    registry.register("core_hf", "base", "core model, training format (not loaded for use)", hf, 0, rec["license"], rec["revision"][:10], "hf", repo=rec["repo"],
                      key=key, precision=C.by_key(key).get("precision", "qlora"))
    for old in glob.glob(os.path.join("models", "base", "*")):
        if os.path.basename(old) != key: __import__("shutil").rmtree(old, ignore_errors=True)  # lean storage: one base at a time
    return {"installed": key, "gguf": q4, "vision_file": mm, "vram_gb": vram}
