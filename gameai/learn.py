"""Always-on learning: keeps improving the current AI in rounds until stopped.

    python -m gameai.learn            (or /learn in the app)

Each round: fresh practice data it has never seen -> continue training from the current
model (warm start, gentle learning rate) -> benchmark. The new version replaces the current
one only if it scores better, so the AI can only get better. Progress and the score after
every round go to learn/history.jsonl (the app draws them as the "how smart" graph).
The teacher AI runs at the same time (if you have an NVIDIA GPU): it keeps writing natural,
real-person examples into data/teacher.jsonl, and every round learns from all of them so far.
Turn that off with GAMEAI_LEARN_TEACHER=0.
Stop: /learn off in the app, Ctrl+C here, or create the file learn/STOP."""
import json, os, subprocess, sys, time

STEPS = int(os.environ.get("GAMEAI_LEARN_STEPS", 800))  # training steps per round
PRACTICE = int(os.environ.get("GAMEAI_LEARN_DATA", 40_000))  # new practice prompts per round


def run(args):
    return subprocess.run([sys.executable, "-m", *args]).returncode == 0


TEACHER_MODEL = os.environ.get("GAMEAI_LEARN_TEACHER_MODEL", "Qwen/Qwen3.5-4B")  # the smaller teacher fits beside training on 8 GB


def count(path):
    return sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0


def start_teacher():
    """The teacher writes examples non-stop in the background while the rounds run."""
    if os.environ.get("GAMEAI_LEARN_TEACHER", "1") == "0": return None
    try:
        import torch
        if not torch.cuda.is_available(): return None
    except Exception:
        return None
    log = open("learn/teacher.log", "a", encoding="utf-8")
    print(f"=== Starting the teacher ({TEACHER_MODEL}); it writes examples while the AI learns ===", flush=True)
    return subprocess.Popen([sys.executable, "-m", "gameai.teacher", "--n", "100000000", "--model", TEACHER_MODEL, "--batch", "16"],
                            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


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
    teacher, warned = start_teacher(), False
    try:
        loop(rnd, hist, teacher)
    finally:
        if teacher and teacher.poll() is None: teacher.terminate()
    print("=== Always-on learning stopped ===", flush=True)


def loop(rnd, hist, teacher):
    warned = False
    while not os.path.exists("learn/STOP"):
        if teacher and teacher.poll() is not None and not warned:
            warned = True
            print("The teacher stopped (see learn/teacher.log); learning carries on without new examples.", flush=True)
        n_teacher = count("data/teacher.jsonl")
        if n_teacher: print(f"Teacher examples so far: {n_teacher}", flush=True)
        print(f"=== Round {rnd}: new practice data ===", flush=True)
        if not run(["gameai.datagen", "--out", "data/round", "--train", str(PRACTICE), "--val", "1000", "--test", "10",
                    "--seed", str(1000 + rnd)]): break
        print(f"=== Round {rnd}: practice test ===", flush=True)
        practice = None
        if run(["gameai.improve", "--seed", str(5000 + rnd), "--pool", os.environ.get("GAMEAI_LEARN_POOL", "20000")]) and os.path.exists("learn/practice.json"):
            practice = json.load(open("learn/practice.json"))
        print(f"=== Round {rnd}: learning ===", flush=True)
        if not run(["gameai.train", "--init", "models/current.pt", "--data", "data/round", "--steps", str(STEPS),
                    "--lr", "1e-3", "--muon-lr", "0.006", "--eval-every", "400", "--name", f"round{rnd}"]): break
        print(f"=== Round {rnd}: testing ===", flush=True)
        if not run(["gameai.bench", "--fresh", "2000"]): break
        last = json.loads(open("bench/history.jsonl").readlines()[-1])
        best = json.load(open("models/current.json")) if os.path.exists("models/current.json") else last  # the AI in use now
        row = {"round": rnd, "time": time.time(), "exact": best.get("total", best["frozen"]["exact"]), "typo": best["typo"]["exact"],
               "frozen": best["frozen"]["exact"], "gold": (best.get("gold") or {}).get("exact"), "version": best.get("version"),
               "practice": practice and practice["practice_score"], "lessons": practice and practice["lessons"],
               "weakest": practice and practice["weakest"], "why": last.get("why"),
               "natural": (best.get("natural") or {}).get("exact"),
               "candidate": last.get("total", last["frozen"]["exact"]), "promoted": last["promoted"], "teacher": n_teacher}
        with open(hist, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"@@L{json.dumps(row)}", flush=True)
        print(f"Round {rnd}: candidate {row['candidate']:.2%}, in use {row['exact']:.2%} -> {'NEW BEST, now in use' if row['promoted'] else 'not better, kept the old one'}", flush=True)
        rnd += 1


if __name__ == "__main__":
    main()
