"""The model registry: one list of every model the AI can use (the core model, its skill adapters, specialists),
what each is for, its format, how much GPU memory it needs, its licence and version.

Swapping in a stronger base later means changing the registry and re-training the adapters on the same data;
the data, checkers and benchmarks carry over. A bigger GPU only changes the budget.

    python -m gameai.registry            show everything registered
    python -m gameai.registry plan core,game_design,pixel_art    what fits in the GPU together, and what must wait
"""
import json, os, time

PATH = os.path.join("models", "registry.json")
KINDS = ("base", "adapter", "specialist", "tool")
SAFETY_GB = 0.8  # left free for Windows, the screen and the KV cache growing


def load():
    if not os.path.exists(PATH): return {"entries": {}, "budget_gb": None}
    return json.load(open(PATH, encoding="utf-8"))


def save(reg):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".new"; json.dump(reg, open(tmp, "w", encoding="utf-8"), indent=1); os.replace(tmp, PATH)


def register(name, kind, role, path, vram_gb, licence, version, fmt, base=None, **extra):
    """Add or update one entry. kind: base | adapter | specialist | tool. base: the base model an adapter belongs to."""
    if kind not in KINDS: raise ValueError(f"kind must be one of {KINDS}")
    if kind == "adapter" and not base: raise ValueError("an adapter must name the base model it was trained on")
    reg = load()
    reg["entries"][name] = {"kind": kind, "role": role, "path": path, "vram_gb": float(vram_gb), "licence": licence, "version": version,
                            "format": fmt, "base": base, "updated": time.strftime("%Y-%m-%d %H:%M"), **extra}
    save(reg)
    return reg["entries"][name]


def remove(name):
    reg = load(); reg["entries"].pop(name, None); save(reg)


def gpu_budget():
    """GPU memory the AI may use: the card's total minus a safety margin (or the saved budget)."""
    reg = load()
    if reg.get("budget_gb"): return reg["budget_gb"]
    try:
        import torch
        if torch.cuda.is_available(): return round(torch.cuda.get_device_properties(0).total_memory / 2**30 - SAFETY_GB, 2)
    except Exception:
        pass
    return 0.0


def plan(wanted, budget=None, loaded=()):
    """Which of the wanted entries can be on the GPU together. Adapters ride on their base (the base loads first);
    the rest of the wanted list waits and loads one at a time when something is unloaded.
    -> {"load": [...], "wait": [...], "unload": [...], "used_gb", "budget_gb"}"""
    reg, budget = load()["entries"], gpu_budget() if budget is None else budget
    need, missing = [], []
    for n in wanted:
        e = reg.get(n)
        if not e: missing.append(n); continue
        if e["kind"] == "adapter" and e["base"] not in need: need.append(e["base"])
        if n not in need: need.append(n)
    used, load_, wait = 0.0, [], []
    for n in need:
        v = reg[n]["vram_gb"] if n in reg else 0
        if used + v <= budget: load_.append(n); used += v
        else: wait.append(n)
    unload = [n for n in loaded if n not in load_]
    return {"load": load_, "wait": wait, "unload": unload, "missing": missing, "used_gb": round(used, 2), "budget_gb": budget}


def main():
    import sys
    a = sys.argv[1:]
    if a and a[0] == "plan":
        print(json.dumps(plan(a[1].split(",") if len(a) > 1 else []), indent=1)); return
    reg = load()
    if not reg["entries"]: print("nothing registered yet"); return
    for n, e in sorted(reg["entries"].items()):
        print(f"{n:24} {e['kind']:10} {e['vram_gb']:5.2f} GB  {e['format']:8} {e['licence']:12} v{e['version']}  {e['role']}")
    print(f"GPU budget: {gpu_budget()} GB")


if __name__ == "__main__":
    main()
