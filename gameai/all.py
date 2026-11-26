"""One command for the whole Phase 1 pipeline: data -> train -> test -> export.

    python -m gameai.all                # small model
    python -m gameai.all --size base
"""
import argparse, json, os, subprocess, sys


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    from .sub import run
    if run(args):
        print("\nStopped: this step failed. Copy the text above and send it to Claude.")
        sys.exit(1)


def smaller_current(size):
    """models/current.pt if it's a smaller AI that can grow into `size`, else None"""
    path = "models/current.pt"
    if not os.path.exists(path): return None
    try:
        import torch
        from .model import SIZES
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = ck["config"]; l, d, h, kv = SIZES[size]
        if (l, d) == (c["n_layer"], c["d_model"]) or l < c["n_layer"] or d < c["d_model"] or (d // h) % (c["d_model"] // c["n_head"]): return None
        return path
    except Exception as e:
        print(f"(starting from random: couldn't read the current AI: {e})"); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--steps", default=None, help="default depends on the size")
    ap.add_argument("--init", default=None, help="upgrade this AI (keeps what it learned) instead of starting from scratch")
    ap.add_argument("--fresh", action="store_true", help="a bigger size starts from random numbers instead of from your current AI")
    a = ap.parse_args()
    from .recipe import GRAMMAR_VERSION
    old = json.load(open("data/manifest.json")) if os.path.exists("data/manifest.json") else None
    if old and old.get("grammar_version", 2) != GRAMMAR_VERSION:
        print(f"1/4  The practice data is from an older game language (version {old.get('grammar_version', 2)}), so making new data")
        old = None
    if not old:
        step("1/4  Making training data", ["gameai.datagen"])
    else:
        print("1/4  Training data already exists, reusing it")
    if a.init:
        step("2/4  Teaching your AI the new parts (it keeps what it learned)", ["gameai.train", "--init", a.init, "--steps", a.steps or "1500", "--lr", "1.5e-3", "--muon-lr", "0.01"])
    else:
        base = smaller_current(a.size) if not a.fresh else None
        if base:  # training speed-ups 1 and 2: start from the current AI and learn from its answers
            step(f"2/4  Training a {a.size} AI, starting from your current one", ["gameai.train", "--size", a.size, "--grow-from", base, "--teacher", base]
                 + (["--steps", a.steps] if a.steps else []))
        else:
            step(f"2/4  Training the AI from scratch ({a.size})", ["gameai.train", "--size", a.size] + (["--steps", a.steps] if a.steps else []))
    step("3/4  Testing it", ["gameai.bench"])
    if os.path.exists("models/current.pt"):
        step("4/4  Making the browser/Chromebook version", ["gameai.export_onnx"])
    cur = json.load(open("models/current.json")) if os.path.exists("models/current.json") else None
    print("\n=== Done ===")
    if cur:
        f = cur["fresh"]
        print(f"Current best AI: {f['exact']:.1%} of new prompts fully right, {f['instruction_match']:.1%} of requests in the built game.")
        print("Try it: start.bat, option 2.  Send Claude the table printed in step 3.")
    else:
        print("No model has passed the test yet. Send Claude the table printed in step 3.")


if __name__ == "__main__":
    main()
