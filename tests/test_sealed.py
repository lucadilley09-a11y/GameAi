"""Data-integrity tests: test questions never reach training, sealed tasks stay unreadable.  python -m tests.test_sealed"""
import json, os, sys, tempfile

results = []


def case(name, ok, why=""):
    results.append((name, bool(ok), why))


here = os.getcwd()
os.chdir(tempfile.mkdtemp())
os.makedirs("data"); os.makedirs("gameai")
sys.path.insert(0, here)
from gameai import guard, sealed  # noqa: E402

tests = ["snowy level with lots of dragons and a boss", "ninja collecting gems in space", "a castle full of bats at night"]
with open("data/test_frozen.jsonl", "w") as f:
    for t in tests[:2]: f.write(json.dumps({"prompt": t}) + "\n")
with open("my_tasks.jsonl", "w") as f:
    f.write(json.dumps({"request": tests[2], "expect": {"enemy": "bats"}}) + "\n")
    f.write(json.dumps({"request": "something fun", "ask": True}) + "\n")
sealed.add("game_requests", "my_tasks.jsonl")

store = open("sealed/store.bin", "rb").read()
case("sealed tasks are encrypted on disk", b"castle" not in store and b"bats" not in store)
case("only sizes come out of the summary", "castle" not in json.dumps(sealed.summary()))
try:
    open("bad.jsonl", "w").write(json.dumps({"request": "no answer given"}) + "\n"); sealed.add("x", "bad.jsonl"); case("bad task files are refused", False)
except ValueError:
    case("bad task files are refused", True)

ix = guard.build_index()
train = [{"prompt": tests[0]},                                   # the same as a test
         {"prompt": "Snowy level, with lots of dragons and a boss!"},  # a reworded copy
         {"prompt": "A castle full of bats, at night"},             # a reworded sealed task
         {"prompt": "a desert with slow slimes and coins"}]         # unrelated: must stay
kept = guard.filter_training(train, ix=ix, log=None)
case("drops a test question from training", train[0] not in kept)
case("drops a reworded copy of a test question", train[1] not in kept)
case("drops a reworded copy of a sealed task (without reading it)", train[2] not in kept)
case("keeps unrelated training examples", train[3] in kept)

os.chdir(here)
for name, ok, why in results: print(("PASS " if ok else "FAIL ") + name + (f": {why}" if why else ""))
print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
