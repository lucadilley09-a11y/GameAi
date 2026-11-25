"""An independent check of what you asked for, separate from the coder's own reading of it.

The coder writes "spec: ..." (what it understood) and code that is tested against that spec. But the spec itself
could be a misunderstanding: valid code for the wrong behaviour. This reads your words directly (which behaviours
you named, any distance, and order words like "until ... then" or "otherwise") and says whether the coder's spec
agrees. It is deliberately simple and only says "agrees" or "disagrees" when your words are clear; otherwise
it says it isn't sure.
"""
import re

WORDS = {  # longest phrases are matched first
    "spiral": ["spiral", "circle closer and closer", "closer and closer", "close in while circling", "circling closer", "spiral around you getting closer"],
    "zigzag": ["come at you in a zigzag", "zigzag", "zig zag", "weave", "snake toward", "wiggle"],
    "pace_x": ["left and right", "side to side", "sideways", "left to right", "back and forth sideways", "horizontally"],
    "pace_y": ["up and down", "vertically"],
    "still": ["stand still", "don't move", "dont move", "stay put", "wait in place", "freeze in place", "stay where", "not move", "sit still", "stop moving"],
    "circle": ["circle", "orbit", "go around you", "spin around", "around you in circles", "in circles"],
    "keep": ["keep their distance", "at a distance", "keep away", "out of reach", "pixels away", "stay about", "keep distance", "away but not too far"],
    "flee": ["keep running away", "running away", "run away", "flee", "avoid", "escape", "back away", "stay away", "run off", "get away"],
    "chase": ["chase", "follow", "hunt", "rush", "charge", "come straight", "come at you", "run at you", "go after", "come for you", "attack you"],
}


def mentioned(text):
    """-> list of (position, behaviour) in the order they appear in the text"""
    t = " " + text.lower().replace("’", "'") + " "
    found, taken = [], [False] * len(t)
    phrases = sorted(((p, k) for k, ps in WORDS.items() for p in ps), key=lambda x: -len(x[0]))
    for p, k in phrases:
        for m in re.finditer(re.escape(p), t):
            if any(taken[m.start():m.end()]): continue
            for i in range(m.start(), m.end()): taken[i] = True
            found.append((m.start(), k))
    found.sort()
    out = []
    for pos, k in found:  # the same behaviour twice counts once
        if k not in [x[1] for x in out]: out.append((pos, k))
    return out


def kinds(spec):
    if spec is None: return []
    return [spec["a"]["kind"], spec["b"]["kind"]] if spec["kind"] == "near" else [spec["kind"]]


def check(text, spec):
    """-> (True, why) agrees / (False, why) disagrees / (None, why) not sure"""
    if spec is None: return False, "it didn't say what it understood"
    found = mentioned(text)
    said = {k for _, k in found}
    want = set(kinds(spec))
    if not said: return None, "I couldn't tell from your words which movement you meant"
    if said != want:
        return False, f"you mentioned {', '.join(sorted(said))} but it understood {', '.join(sorted(want))}"
    nums = [int(n) for n in re.findall(r"\b(100|150|200)\b", text)]
    R = spec.get("R") if spec["kind"] == "near" else spec.get("R") if spec["kind"] == "keep" else None
    if nums and R is not None and nums[0] != R:
        return False, f"you said {nums[0]} but it used {R}"
    if spec["kind"] == "near" and len(found) == 2:
        t = text.lower()
        first, second = found[0][1], found[1][1]
        u = t.find("until")
        if u >= 0 and found[0][0] - 1 < u < found[1][0] - 1:  # "B until you're close, then A"
            if (spec["b"]["kind"], spec["a"]["kind"]) != (first, second):
                return False, "the 'until ... then' order is the other way round"
        o = max(t.find("otherwise"), t.find("the rest of the time"), t.find("if not"))
        if o >= 0 and u < 0:  # "A when close, otherwise B": what comes after "otherwise" is what they do when far
            after = [k for p, k in found if p - 1 > o]
            if after and after[0] != spec["b"]["kind"]:
                return False, "it mixed up what happens near and far"
    return True, "matches your words"
