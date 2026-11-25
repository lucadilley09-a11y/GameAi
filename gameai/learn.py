"""Always-on learning: keeps improving the current AI in rounds until stopped.

    python -m gameai.learn            (or /learn in the app)

Each round: fresh practice data it has never seen -> continue training from the current
model (warm start, gentle learning rate) -> benchmark. The new version replaces the current
one only if it scores better, so the AI can only get better. Progress and the score after
every round go to learn/history.jsonl (the app draws them as the "how smart" graph).
Stop: /learn off in the app, Ctrl+C here, or create the file learn/STOP."""
import json, os, subprocess, sys, time

STEPS = int(os.environ.get("GAMEAI_LEARN_STEPS", 800))  # training steps per round
PRACTICE = int(os.environ.get("GAMEAI_LEARN_DATA", 40_000))  # new practice prompts per round


def run(args):
    return subprocess.run([sys.executable, "-m", *args]).returncode == 0


def main():
    if not os.path.exists("models/current.pt"):
        print("Train a first AI before always-on learning (/train)."); sys.exit(1)
    os.makedirs("learn", exist_ok=True)
    if os.path.exists("learn/STOP"): os.remove("learn/STOP")
    hist = "learn/history.jsonl"
    rnd = sum(1 for _ in open(hist)) if os.path.exists(hist) else 0
    if rnd == 0 and os.path.exists("models/current.json"):  # starting point for the graph
        cur = json.load(open("models/current.json"))
        with open(hist, "a") as f:
            f.write(json.dumps({"round": 0, "time": time.time(), "exact": cur["frozen"]["exact"], "typo": cur["typo"]["exact"], "promoted": True}) + "\n")
        rnd = 1
    while not os.path.exists("learn/STOP"):
        print(f"=== Round {rnd}: new practice data ===", flush=True)
        if not run(["gameai.datagen", "--out", "data/round", "--train", str(PRACTICE), "--val", "1000", "--test", "10",
                    "--seed", str(1000 + rnd)]): break
        print(f"=== Round {rnd}: learning ===", flush=True)
        if not run(["gameai.train", "--init", "models/current.pt", "--data", "data/round", "--steps", str(STEPS),
                    "--lr", "1e-3", "--muon-lr", "0.006", "--eval-every", "400", "--name", f"round{rnd}"]): break
        print(f"=== Round {rnd}: testing ===", flush=True)
        if not run(["gameai.bench", "--fresh", "2000"]): break
        last = json.loads(open("bench/history.jsonl").readlines()[-1])
        best = json.load(open("models/current.json"))  # the AI actually in use after this round
        row = {"round": rnd, "time": time.time(), "exact": best["frozen"]["exact"], "typo": best["typo"]["exact"],
               "natural": (best.get("natural") or {}).get("exact"),
               "candidate": last["frozen"]["exact"], "promoted": last["promoted"]}
        with open(hist, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"@@L{json.dumps(row)}", flush=True)
        print(f"Round {rnd}: candidate {row['candidate']:.2%}, in use {row['exact']:.2%} -> {'NEW BEST, now in use' if row['promoted'] else 'not better, kept the old one'}", flush=True)
        rnd += 1
    print("=== Always-on learning stopped ===", flush=True)


if __name__ == "__main__":
    main()
