"""One command for the whole Phase 1 pipeline: data -> train -> test -> export.

    python -m gameai.all                # small model
    python -m gameai.all --size base
"""
import argparse, json, os, subprocess, sys


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    if subprocess.run([sys.executable, "-m", *args]).returncode:
        print("\nStopped: this step failed. Copy the text above and send it to Claude.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--steps", default="4000")
    a = ap.parse_args()
    if not os.path.exists("data/manifest.json"):
        step("1/4  Making training data", ["gameai.datagen"])
    else:
        print("1/4  Training data already exists, reusing it")
    step(f"2/4  Training the AI from scratch ({a.size})", ["gameai.train", "--size", a.size, "--steps", a.steps])
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
