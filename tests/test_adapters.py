"""Model registry and adapter versions: what fits on the GPU, and when a new version may be used.  python -m tests.test_adapters"""
import os, sys, tempfile

here = os.getcwd(); os.chdir(tempfile.mkdtemp()); sys.path.insert(0, here)
from gameai import registry as R, adapters as A  # noqa: E402

results = []
def case(name, ok): results.append((name, bool(ok)))

R.register("core", "base", "core", "x.gguf", 2.6, "apache-2.0", "r1", "gguf")
R.register("design", "adapter", "design", "d", 0.05, "ours", 1, "gguf", base="core")
R.register("vision", "specialist", "vision", "v", 0.6, "apache-2.0", 1, "gguf")
p = R.plan(["design", "vision"], budget=3.0)
case("an adapter brings its base model along", p["load"][:2] == ["core", "design"])
case("what doesn't fit waits instead of overflowing the GPU", p["wait"] == ["vision"] and p["used_gb"] <= 3.0)
try: R.register("bad", "adapter", "x", "x", 0, "x", 1, "gguf"); case("an adapter must name its base", False)
except ValueError: case("an adapter must name its base", True)

S = lambda a, b: {"game": {"score": a, "n": 200}, "general": {"score": b, "n": 300}}
def add(a, b):
    v, _ = A.new_version("code"); A.finish("code", v, "core", "r1", S(a, b)); return v, A.decide("code", v)[0]
v1, ok1 = add(.50, .70); v2, ok2 = add(.60, .60); v3, ok3 = add(.55, .70); v4, ok4 = add(.55, .695)
case("the first version is used", ok1)
case("better overall but clearly worse on general ability: refused", not ok2)
case("better overall and nothing clearly worse: used", ok3 and A.active("code")["version"] == v3)
case("not better overall: refused", not ok4)
A.rollback("code", v1); case("rollback to any kept version", A.active("code")["version"] == v1)
A.KEEP = 2; A.prune("code"); kept = [v["version"] for v in A.versions("code")]
case("pruning keeps the version in use", v1 in kept and v2 not in kept)

os.chdir(here)
for name, ok in results: print(("PASS " if ok else "FAIL ") + name)
print(f"{sum(ok for _, ok in results)}/{len(results)} passed")
sys.exit(0 if all(ok for _, ok in results) else 1)
