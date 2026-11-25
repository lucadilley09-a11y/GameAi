"""What enemy brains can do, in three forms:
  - spec: the AI's short note of what it understood, e.g. "near 100: flee else circle"
  - a reference (here, in Python) that defines exactly what that spec means
  - many correct ways to write it in JavaScript, and many ways people ask for it
A brain the AI writes is kept only if it moves exactly like the reference for the spec it wrote."""
import math, random

ATOMS = ["chase", "flee", "circle", "keep", "zigzag", "pace_x", "pace_y", "still", "spiral"]
RADII = [100, 150, 200]


def _dir(e, p):
    dx, dy = p[0] - e[0], p[1] - e[1]; d = math.hypot(dx, dy) or 1
    return dx / d, dy / d


def _sign(v): return (v > 0) - (v < 0)


def atom_ref(name, R, e, p, t):
    d0, d1 = _dir(e, p)
    if name == "chase": return d0, d1
    if name == "flee": return -d0, -d1
    if name == "circle": return -d1, d0
    if name == "keep": k = 1 if math.hypot(p[0] - e[0], p[1] - e[1]) > R else -1; return d0 * k, d1 * k
    if name == "zigzag": s = math.sin(t * 6) * 0.8; return d0 - d1 * s, d1 + d0 * s
    if name == "pace_x": return _sign(math.sin(t * 2)), 0
    if name == "pace_y": return 0, _sign(math.sin(t * 2))
    if name == "still": return 0, 0
    if name == "spiral": return -d1 + d0 * 0.5, d0 + d1 * 0.5
    raise ValueError(name)


def parse(spec):
    """'chase' | 'keep 150' | 'near 100: flee else circle'  ->  dict, or None if it isn't a valid spec"""
    spec = spec.strip()
    try:
        if spec.startswith("near "):
            head, rest = spec[5:].split(":", 1)
            a, b = rest.strip().split(" else ")
            R = int(head); A, B = parse(a), parse(b)
            if R not in RADII or not A or not B or "near" in (A["kind"], B["kind"]) or A == B: return None
            return {"kind": "near", "R": R, "a": A, "b": B}
        if spec.startswith("keep "):
            R = int(spec[5:]); return {"kind": "keep", "R": R} if R in RADII else None
        return {"kind": spec} if spec in ATOMS and spec != "keep" else None
    except (ValueError, TypeError):
        return None


def reference(spec):
    """spec dict -> function(e, p, t) -> (dx, dy)"""
    if spec["kind"] == "near":
        A, B, R = reference(spec["a"]), reference(spec["b"]), spec["R"]
        return lambda e, p, t: A(e, p, t) if math.hypot(p[0] - e[0], p[1] - e[1]) < R else B(e, p, t)
    return lambda e, p, t: atom_ref(spec["kind"], spec.get("R", 150), e, p, t)


def spec_text(s):
    if s["kind"] == "near": return f"near {s['R']}: {spec_text(s['a'])} else {spec_text(s['b'])}"
    return f"keep {s['R']}" if s["kind"] == "keep" else s["kind"]


# ---------- how people ask for it ----------
SAY = {
    "chase": ["chase you", "run at you", "come straight for you", "hunt you down", "follow you", "rush toward you", "go after you", "charge at you"],
    "flee": ["run away from you", "flee from you", "stay away from you", "avoid you", "run off", "escape from you", "keep running away", "back away from you"],
    "circle": ["circle around you", "orbit you", "go around you in circles", "circle you", "spin around you", "move in circles around you"],
    "keep": ["keep their distance", "stay at a distance", "hover at a distance", "keep away but not too far", "stay out of reach"],
    "keep_n": ["stay about {R} away", "keep {R} pixels away", "stay {R} away from you", "hover {R} pixels away"],
    "zigzag": ["zigzag toward you", "weave toward you", "snake toward you", "come at you in a zigzag", "wiggle toward you"],
    "pace_x": ["walk left and right", "pace back and forth sideways", "move side to side", "patrol left to right", "go left and right"],
    "pace_y": ["walk up and down", "pace up and down", "move up and down", "patrol up and down", "go up and down"],
    "still": ["stand still", "don't move", "stay put", "wait in place", "freeze in place", "stay where they are"],
    "spiral": ["spiral toward you", "spiral in", "circle closer and closer", "close in while circling", "spiral around you getting closer"],
}
NEAR = {100: ["when you get close", "if you come close", "when you're within 100", "when you're really close", "up close"],
        150: ["when you're nearby", "when you're within 150", "if you get near", "when you come near"],
        200: ["when you're within 200", "when you're in range", "once you're within 200", "when you come into range"]}
SUBJ = ["", "enemies that ", "make the enemies ", "monsters that ", "i want enemies to ", "the enemies should ", "make them ", "enemies ", "bad guys that "]
FILL = ["", " please", " lol", " like in pacman", " that would be cool"]


def say(s, rng):
    if s["kind"] == "near":
        A, B, near = say_atom(s["a"], rng), say_atom(s["b"], rng), rng.choice(NEAR[s["R"]])
        form = rng.choice(["{A} {N}, otherwise {B}", "{B}, but {A} {N}", "{N} they {A}, otherwise they {B}", "{B} until you're close, then {A}" if s["R"] == 100 else "{B}, and {A} {N}",
                           "{A} {N} and {B} the rest of the time", "{B}; {A} {N}"])
        txt = form.format(A=A, B=B, N=near)
    else:
        txt = say_atom(s, rng)
    return (rng.choice(SUBJ) + txt + rng.choice(FILL)).strip()


def say_atom(s, rng):
    if s["kind"] == "keep":
        return rng.choice(SAY["keep_n"]).format(R=s["R"]) if s["R"] != 150 or rng.random() < .5 else rng.choice(SAY["keep"])
    return rng.choice(SAY[s["kind"]])


# ---------- how to write it ----------
NAMES = {"v": ["d", "to", "aim", "way", "toward"], "a": [("ax", "ay"), ("vx", "vy"), ("gx", "gy")], "k": ["k", "side", "sgn"], "s": ["s", "wob", "w"], "dd": ["dd", "far", "r"]}


def code_atom(s, rng):
    v, (a, b), k, w = rng.choice(NAMES["v"]), rng.choice(NAMES["a"]), rng.choice(NAMES["k"]), rng.choice(NAMES["s"])
    R = s.get("R", 150)
    T = {
        "chase": ["return dir(e, p);", f"const {v} = dir(e, p);\nreturn [{v}[0], {v}[1]];", f"const {a} = p.x - e.x, {b} = p.y - e.y;\nreturn [{a}, {b}];"],
        "flee": [f"const {v} = dir(e, p);\nreturn [-{v}[0], -{v}[1]];", "return [e.x - p.x, e.y - p.y];", f"const {a} = e.x - p.x, {b} = e.y - p.y;\nreturn [{a}, {b}];"],
        "circle": [f"const {v} = dir(e, p);\nreturn [-{v}[1], {v}[0]];", f"const {a} = p.x - e.x, {b} = p.y - e.y;\nreturn [-{b}, {a}];"],
        "keep": [f"const {v} = dir(e, p);\nconst {k} = dist(e, p) > {R} ? 1 : -1;\nreturn [{v}[0] * {k}, {v}[1] * {k}];",
                 f"const {v} = dir(e, p);\nif (dist(e, p) > {R}) return {v};\nreturn [-{v}[0], -{v}[1]];"],
        "zigzag": [f"const {v} = dir(e, p);\nconst {w} = Math.sin(t * 6) * 0.8;\nreturn [{v}[0] - {v}[1] * {w}, {v}[1] + {v}[0] * {w}];"],
        "pace_x": ["return [Math.sign(Math.sin(t * 2)), 0];"],
        "pace_y": ["return [0, Math.sign(Math.sin(t * 2))];"],
        "still": ["return [0, 0];"],
        "spiral": [f"const {v} = dir(e, p);\nreturn [-{v}[1] + {v}[0] * 0.5, {v}[0] + {v}[1] * 0.5];"],
    }
    return rng.choice(T[s["kind"]])


def code(s, rng):
    if s["kind"] != "near": return code_atom(s, rng)
    A, B, R = code_atom(s["a"], rng), code_atom(s["b"], rng), s["R"]
    ind = lambda c: "\n".join("  " + l for l in c.split("\n"))
    dd = rng.choice(NAMES["dd"])
    forms = [f"if (dist(e, p) < {R}) {{\n{ind(A)}\n}}\n{B}",
             f"const {dd} = dist(e, p);\nif ({dd} < {R}) {{\n{ind(A)}\n}}\n{B}",
             f"if (dist(e, p) >= {R}) {{\n{ind(B)}\n}}\n{A}"]
    return rng.choice(forms)


def random_spec(rng, composite=None):
    def atom():
        k = rng.choice(ATOMS)
        return {"kind": "keep", "R": rng.choice(RADII)} if k == "keep" else {"kind": k}
    if composite is None: composite = rng.random() < .55
    if not composite: return atom()
    while True:
        A, B = atom(), atom()
        if A != B and "keep" not in (A["kind"], B["kind"]): return {"kind": "near", "R": rng.choice(RADII), "a": A, "b": B}


def describe(s):
    """plain words, for the terminal"""
    W = {"chase": "chase you", "flee": "run away from you", "circle": "circle around you", "zigzag": "zigzag toward you",
         "pace_x": "walk left and right", "pace_y": "walk up and down", "still": "stand still", "spiral": "spiral in toward you"}
    if s["kind"] == "near": return f"{describe(s['a'])} when you're within {s['R']}, otherwise {describe(s['b'])}"
    return f"stay about {s['R']} away from you" if s["kind"] == "keep" else W[s["kind"]]
