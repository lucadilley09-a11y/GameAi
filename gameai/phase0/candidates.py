"""The bake-off candidates (from the plan's table), and their licence and size, recorded BEFORE anything is downloaded.

A candidate is only downloaded when its licence is on the allowed list (permissive: Apache 2.0 or MIT), and never
if it needs to run its own code (trust_remote_code): that would mean running code from the internet on your PC.
"""
import json, os

from .common import OUT, say

ALLOWED = {"apache-2.0", "mit"}

# role: "base" candidates are fine-tuned; "teacher" ones are measured but not trained (they don't fit QLoRA on 8 GB)
CANDIDATES = [
    # precision: how it's trained. Unsloth's Qwen3.5 guide advises against 4-bit QLoRA for Qwen3.5 (larger quantization
    # differences than usual), so the 2B trains with 16-bit LoRA, which fits on 8 GB; the 4B doesn't fit in 16-bit, so it
    # can only be tested with QLoRA, and that's noted in its results.
    {"key": "qwen3.5-2b", "repos": ["Qwen/Qwen3.5-2B"], "role": "base (preferred start)", "train": True, "precision": "bf16"},
    {"key": "minicpm5-2b", "repos": ["openbmb/MiniCPM5-2B"], "role": "base", "train": True},
    {"key": "gemma-4-e2b", "repos": ["google/gemma-4-E2B-it", "google/gemma-4-E2B"], "role": "base", "train": True},
    {"key": "granite-4.2-3b", "repos": ["ibm-granite/granite-4.2-3b"], "role": "coding teacher, or base if far ahead", "train": True},
    {"key": "qwen3.5-4b", "repos": ["Qwen/Qwen3.5-4B"], "role": "challenger", "train": True, "precision": "qlora",
     "note": "QLoRA isn't recommended for Qwen3.5 and 16-bit LoRA doesn't fit on 8 GB: its training result may understate it"},
    {"key": "gemma-4-e4b", "repos": ["google/gemma-4-E4B-it", "google/gemma-4-E4B"], "role": "teacher or reference (QLoRA needs about 9-10 GB)", "train": False},
]
WEIGHT_FILES = ("*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "tokenizer*", "LICENSE*", "*.tiktoken")


def by_key(key):
    for c in CANDIDATES:
        if c["key"] == key: return c
    raise KeyError(f"no candidate called {key!r}; choose from {', '.join(c['key'] for c in CANDIDATES)}")


def record(c):
    """Look up one candidate on Hugging Face (a few KB): which repo, licence, download size, whether it can be used."""
    from huggingface_hub import HfApi
    api, last_err = HfApi(), None
    for repo in c["repos"]:
        try:
            info = api.model_info(repo, files_metadata=True)
        except Exception as e:
            last_err = e; continue
        card = getattr(info, "card_data", None) or {}
        lic = (card.get("license") if hasattr(card, "get") else getattr(card, "license", None)) or ""
        tags = {t.lower() for t in (info.tags or [])}
        if not lic: lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "")
        import fnmatch
        files = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
        want = [f for f in files if any(fnmatch.fnmatch(os.path.basename(f[0]), p) for p in WEIGHT_FILES)]
        remote_code = any(f[0].endswith(".py") for f in files) and "custom_code" in tags
        rec = {"key": c["key"], "repo": repo, "revision": info.sha, "license": lic.lower(), "role": c["role"],
               "download_gb": round(sum(s for _, s in want) / 2**30, 2), "gated": bool(getattr(info, "gated", False)),
               "needs_its_own_code": remote_code, "pipeline": getattr(info, "pipeline_tag", None)}
        rec["allowed"] = rec["license"] in ALLOWED and not remote_code
        rec["why_not"] = (None if rec["allowed"] else
                          f"licence '{rec['license'] or 'unknown'}' isn't on the allowed list" if rec["license"] not in ALLOWED else
                          "it needs to run its own code from the internet (trust_remote_code), which the plan doesn't allow")
        return rec
    return {"key": c["key"], "repo": c["repos"][0], "allowed": False, "why_not": f"couldn't look it up on Hugging Face ({last_err})"}


def record_all():
    out = []
    for c in CANDIDATES:
        r = record(c); out.append(r)
        say(f"  {c['key']:16} {r.get('license', '?'):12} {r.get('download_gb', '?')!s:>6} GB  {'ok' if r['allowed'] else 'skipped: ' + r['why_not']}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "licences.json"), "w"), indent=1)
    return out


def download(rec, dest):
    """Weights, tokenizer, config and the licence text, pinned to the exact revision that was checked."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(rec["repo"], revision=rec["revision"], local_dir=dest, allow_patterns=list(WEIGHT_FILES))
    lic = [f for f in os.listdir(path) if f.upper().startswith("LICENSE")]
    json.dump({**rec, "licence_files": lic}, open(os.path.join(path, "provenance.json"), "w"), indent=1)
    return path
