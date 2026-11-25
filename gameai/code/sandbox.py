"""Runs code the AI wrote, safely, and checks it does what its spec says.

1. safety.check: only allowed words (no loops, no access to anything outside the maths).
2. Run it inside a separate JavaScript engine (V8 through mini-racer) with a time limit and a memory limit.
3. Compare with the reference for the spec on 40 random situations (near and far, any time):
   the direction it moves in must match in every one."""
import json, math, os, random
from . import safety, specs

_TOOLKIT = open(os.path.join(os.path.dirname(__file__), "toolkit.js"), encoding="utf-8").read()
_ctx = None


def _engine():
    global _ctx
    if _ctx is None:
        from py_mini_racer import MiniRacer
        _ctx = MiniRacer()
        _ctx.eval(_TOOLKIT)
        import atexit
        atexit.register(_ctx.close)  # without this the program waits forever when it ends
    return _ctx


def run(code, states, timeout_ms=500):
    """-> list of [dx, dy] for each (e, p, t), or raises ValueError with a plain reason"""
    ok, why = safety.check(code)
    if not ok: raise ValueError(why)
    js = ("(function(){let brain; try { brain = new Function('e','p','t','dist','dir', %s); } catch (err) { return 'ERR' + err.message; }"
          "const S = %s, out = []; for (const [ex, ey, px, py, t] of S) { let r; try { r = brain({x: ex, y: ey, mem: {}}, {x: px, y: py}, t, dist, dir); }"
          " catch (err) { return 'ERR' + err.message; } if (!Array.isArray(r) || r.length !== 2 || !isFinite(r[0]) || !isFinite(r[1])) return 'BAD';"
          " out.push([+r[0], +r[1]]); } return JSON.stringify(out); })()") % (json.dumps(code), json.dumps(states))
    try:
        res = _engine().eval(js, timeout=timeout_ms, max_memory=32 * 1024 * 1024)
    except Exception as e:
        raise ValueError(f"it took too long or crashed ({type(e).__name__})")
    if res == "BAD": raise ValueError("it has to return two numbers, like [dx, dy]")
    if isinstance(res, str) and res.startswith("ERR"): raise ValueError("the code has a mistake: " + res[3:])
    return json.loads(res)


def situations(spec, rng, n=40):
    """random (enemy x, y, player x, y, time); for 'near' specs, half are close to the edge of the distance"""
    out = []
    for i in range(n):
        px, py = rng.uniform(-300, 300), rng.uniform(-300, 300)
        if spec["kind"] == "near" and i % 2:
            r = spec["R"] * rng.choice([rng.uniform(.3, .92), rng.uniform(1.08, 2)])
        else:
            r = rng.uniform(20, 420)
        a = rng.uniform(0, 2 * math.pi)
        out.append([px + r * math.cos(a), py + r * math.sin(a), px, py, round(rng.uniform(0.05, 30), 3)])
    return out


def same_way(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nv < 1e-9: return nu < 1e-6
    if nu < 1e-9: return False
    return (u[0] * v[0] + u[1] * v[1]) / (nu * nv) > 0.999


def verify(spec_str, code, seed=0):
    """-> (ok, reason). Does `code` move exactly like the reference for `spec_str`?"""
    sp = specs.parse(spec_str)
    if not sp: return False, f"I don't know the behaviour '{spec_str}'"
    rng = random.Random(seed)
    st = situations(sp, rng)
    try:
        got = run(code, st)
    except ValueError as e:
        return False, str(e)
    ref = specs.reference(sp)
    for (ex, ey, px, py, t), g in zip(st, got):
        if not same_way(g, ref((ex, ey), (px, py), t)):
            return False, f"it moves the wrong way (for example with the enemy at {ex:.0f},{ey:.0f}, you at {px:.0f},{py:.0f})"
    return True, "works"
