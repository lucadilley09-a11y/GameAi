"""GameKit tests, run in V8 (mini-racer) like the game page.  python -m tests.test_gamekit"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(HERE, "gameai", "kit")


def main():
    from py_mini_racer import MiniRacer
    ctx = MiniRacer()
    try:
        ex = {os.path.basename(p)[:-3]: open(p, encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(KIT, "examples", "*.js")))}
        ctx.eval(open(os.path.join(KIT, "gamekit.js"), encoding="utf-8").read())
        ctx.eval("globalThis.EXAMPLES = " + json.dumps(ex))
        res = json.loads(ctx.eval(open(os.path.join(HERE, "tests", "gamekit_tests.js"), encoding="utf-8").read(), timeout_sec=600))
    finally:
        ctx.close()
    for name, ok, why in res: print(("PASS " if ok else "FAIL ") + name + (f": {why}" if why else ""))
    print(f"{sum(1 for r in res if r[1])}/{len(res)} passed")
    return all(r[1] for r in res)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
