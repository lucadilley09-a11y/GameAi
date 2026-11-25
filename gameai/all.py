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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--steps", default=None, help="default depends on the size")
    ap.add_argument("--init", default=None, help="upgrade this AI (keeps what it learned) instead of starting from scratch")
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
