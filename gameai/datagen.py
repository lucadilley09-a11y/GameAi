"""Synthetic data generator: made-up prompts paired with the correct recipe.

    python -m gameai.datagen                # default sizes
    python -m gameai.datagen --train 400000

Writes data/train.jsonl, data/val.jsonl, data/test_frozen.jsonl, data/test_typo.jsonl
and data/manifest.json (with a hash, so every run records exactly which data it used).
The two test files use fixed seeds and never change: they are the frozen benchmark.
"""
import argparse, hashlib, json, os, random
from .recipe import FIELDS, FIELD_NAMES, AUTO, OUT_FIELDS, RULE_FIELDS, TN_BY_TRIG

SYN = {
    "theme": {"forest": ["forest", "woods", "jungle", "grassland", "nature", "meadow", "woodland", "enchanted forest", "garden"],
              "dungeon": ["dungeon", "castle", "cave", "crypt", "spooky dungeon", "haunted castle", "creepy cave", "dark dungeon", "tomb", "haunted house", "spooky", "creepy"],
              "space": ["space", "alien planet", "galaxy", "spaceship", "space station", "moon base", "outer space", "sci-fi", "mars", "asteroid field"],
              "desert": ["desert", "sandy desert", "pyramid", "egypt", "dunes", "canyon", "wasteland", "sahara", "wild west", "oasis"],
              "snow": ["snow", "ice cave", "winter", "frozen land", "arctic", "snowy mountain", "tundra", "icy", "north pole", "glacier", "christmas"],
              "ocean": ["ocean", "underwater", "sea", "beach", "coral reef", "island", "deep sea", "pirate island", "lagoon", "tropical"],
              "volcano": ["volcano", "lava", "lava world", "fire land", "magma cave", "molten", "inferno"]},
    "player": {"knight": ["knight", "warrior", "hero", "soldier", "paladin", "swordsman"], "cat": ["cat", "kitty", "kitten"],
               "robot": ["robot", "bot", "android", "mech", "cyborg"], "wizard": ["wizard", "mage", "witch", "sorcerer", "magician"],
               "dog": ["dog", "puppy", "doggo", "pup"], "ninja": ["ninja", "assassin", "samurai"], "frog": ["frog", "toad", "froggy"]},
    "enemy": {"bats": ["bats", "vampire bats", "flying bats"], "slimes": ["slimes", "blobs", "goo monsters", "jellies"],
              "skeletons": ["skeletons", "zombies", "undead", "mummies"], "aliens": ["aliens", "martians", "ufos", "space invaders"],
              "ghosts": ["ghosts", "spirits", "phantoms", "spooky ghosts"], "spiders": ["spiders", "giant spiders", "creepy crawlies", "bugs"],
              "dragons": ["dragons", "fire dragons", "drakes", "wyverns"],
              "none": ["no enemies", "peaceful", "no monsters", "cozy", "nothing to fight", "without enemies"]},
    "count": {"few": ["a few", "a couple of", "two or three", "not many", "fewer", "less", "only a few"],
              "some": ["some", "several", "a handful of"],
              "many": ["lots of", "tons of", "hordes of", "a swarm of", "hundreds of", "loads of", "full of", "more", "way more", "a lot of", "an army of"]},
    "speed": {"slow": ["slow", "sluggish", "chill", "relaxed", "calm", "slower", "slow down", "slow motion"],
              "fast": ["fast", "speedy", "quick", "zippy", "rapid", "super fast", "faster", "speed it up", "lightning fast", "hyper"]},
    "goal": {"coins": ["collect coins", "grab coins", "get the coins", "gather gold", "collect gold", "coins", "treasure", "money", "loot"],
             "gems": ["collect gems", "find gems", "grab diamonds", "collect crystals", "jewels", "gems", "rubies"],
             "exit": ["find the exit", "escape", "escaping", "reach the door", "find the way out", "get out of the maze", "exit", "maze", "labyrinth", "find the stairs", "reach the portal"],
             "survive": ["survive", "stay alive", "last as long as you can", "survive the waves", "survival", "endless waves", "hold out"]},
    "difficulty": {"easy": ["easy", "simple", "beginner", "casual", "for kids", "easier", "not too hard", "relaxing", "baby mode"],
                   "hard": ["hard", "difficult", "challenging", "brutal", "insane", "tough", "very hard", "harder", "more difficult", "not too easy", "nightmare", "extreme"]},
    "size": {"small": ["small map", "tiny map", "small level", "compact", "short level", "smaller map", "smaller"],
             "big": ["big map", "huge map", "large world", "giant level", "open world", "big level", "huge world", "bigger map", "bigger", "larger map"]},
    "powerup": {"hearts": ["health pickups", "hearts", "healing hearts", "extra lives", "health potions"],
                "boots": ["speed boots", "speed boosts", "speed powerups", "boots"],
                "shield": ["shields", "shield powerups", "invincibility", "bubble shields"],
                "none": ["no powerups", "no power ups", "no items"]},
    "behavior": {"patrol": ["enemies that patrol", "patrolling guards", "guards on patrol routes", "enemies walking back and forth", "sentries", "patrolling monsters"],
                 "shoot": ["enemies that shoot", "shooting enemies", "monsters that shoot at you", "enemies that fire lasers", "turrets", "enemies that throw fireballs", "shooters"],
                 "wander": ["enemies that wander around", "wandering monsters", "enemies that don't chase you", "roaming enemies", "lazy enemies", "aimless monsters"]},
    "doors": {"keys": ["locked doors and keys", "doors you open with keys", "key and door puzzles", "locked rooms", "keys to unlock doors", "locked doors"],
              "switches": ["switches that open gates", "levers", "pressure plates", "buttons that open doors", "switch puzzles", "gates and levers"]},
    "levels": {"3": ["3 levels", "three levels", "three stages", "3 stages", "a few levels", "multiple levels"],
               "5": ["5 levels", "five levels", "five stages", "lots of levels", "a whole campaign", "many levels"]},
    "boss": {"yes": ["a boss", "a boss fight", "a final boss", "a boss battle", "a big boss", "a boss at the end"]},
    "timer": {"short": ["a time limit", "against the clock", "a 60 second timer", "a timer", "one minute to win", "speedrun style"],
              "long": ["a 2 minute timer", "two minutes to win", "a long timer", "120 seconds", "a relaxed time limit"]},
}
assert all(set(SYN[f]) == set(FIELDS[f]) for f in FIELDS), "synonyms must cover every grammar value"

GENERIC = ["enemies", "monsters", "bad guys", "baddies", "foes", "creatures"]
FILLER = ["please", "pls", "awesome", "cool", "epic", "like zelda", "pixel art", "retro style", "fun", "really", "super",
          "kind of", "i think", "with nice graphics", "for my friend", "top down", "2d", "n stuff", "lol"]
TPL = {
    "theme": ["in a W", "W themed", "set in a W", "W level", "W world", "in the W", "set it in W", "W"],
    "player": ["play as a W", "you are a W", "a W hero", "W player", "starring a W", "a W", "switch to a W", "make me a W"],
    "enemy": ["fighting W", "with W", "dodge the W", "W chase you", "avoid W", "W everywhere", "watch out for W", "add W"],
    "speed": ["W paced", "W player", "make it W", "W movement", "W"],
    "goal": ["W", "goal is to W", "you have to W", "W to win"],
    "difficulty": ["W mode", "make it W", "W difficulty", "W", "W please"],
    "size": ["W", "on a W", "W please", "with a W"],
    "powerup": ["with W", "W to pick up", "W", "add W"],
    "behavior": ["W", "with W", "add W", "make it W"],
    "doors": ["W", "with W", "add W", "puzzles with W"],
    "levels": ["W", "with W", "make it W", "W long"],
    "boss": ["W", "with W", "add W", "and W"],
    "timer": ["W", "with W", "add W", "beat it with W"],
}
PROB = {"theme": .7, "player": .65, "enemy": .65, "count": .45, "speed": .4, "goal": .6, "difficulty": .4, "size": .3, "powerup": .3,
        "behavior": .22, "doors": .2, "levels": .2, "boss": .15, "timer": .15}


RTRIG = {
    "collect": ["whenever you grab a {i1}", "each time you collect a {i1}", "when you pick up a {i1}", "every time you get a {i1}"],
    "hit": ["when you get hit", "if an enemy touches you", "whenever you take damage", "when you get hurt"],
    "every": ["every {n} seconds", "each {n} seconds", "every {n} secs"],
    "score": ["after {n} {items}", "once you collect {n} {items}", "when you have {n} {items}", "after collecting {n} {items}"],
    "powerup": ["when you grab a power up", "whenever you pick up a powerup", "when you get a power-up"],
}
RACT = {
    ("spawn", "1"): ["a new enemy appears", "another enemy spawns", "one more enemy shows up"],
    ("spawn", "3"): ["three more enemies appear", "a wave of enemies spawns", "3 enemies spawn"],
    ("faster", "none"): ["the enemies get faster", "enemies speed up", "the monsters get quicker"],
    ("slower", "none"): ["the enemies slow down", "enemies get slower"],
    ("boost", "none"): ["you get a speed boost", "you move faster for a bit", "you get faster"],
    ("heal", "none"): ["you heal", "you get a heart back", "you regain health"],
    ("teleport", "none"): ["you teleport somewhere random", "you get teleported", "you warp to a random spot"],
    ("open_exit", "none"): ["the exit opens", "a door unlocks", "the exit unlocks", "the portal opens"],
    ("freeze", "none"): ["the enemies freeze", "everything freezes for a moment", "enemies stop moving for a bit"],
}


def make_rule(rng, goal, enemy="auto", powerup="auto", want=None):
    """A random rule that cannot contradict the rest of the prompt (e.g. no 'when you get hit' in a peaceful game).
    With want={trig, tn, act, an}: new words for exactly that rule."""
    trigs = [t for t in RTRIG if not (t == "hit" and enemy == "none") and not (t == "powerup" and powerup == "none")]
    acts = [k for k in RACT if not (enemy == "none" and k[0] in ("spawn", "faster", "slower", "freeze"))]
    if want:
        trig, tn, act, an = want["trig"], want["tn"], want["act"], want["an"]
    else:
        trig = rng.choice(trigs)
        tn = rng.choice(TN_BY_TRIG[trig])
        act, an = rng.choice(acts)
    item = {"coins": ("coin", "coins"), "gems": ("gem", "gems")}.get(goal) or rng.choice([("key", "keys"), ("item", "items")])
    t = rng.choice(RTRIG[trig]).format(i1=item[0], items=item[1], n=tn)
    a = rng.choice(RACT[(act, an)])
    text = rng.choice(["{t}, {a}", "{a} {t}", "{t} {a}", "{t} then {a}"]).format(t=t, a=a)
    return {"trig": trig, "tn": tn, "act": act, "an": an}, text


def typo(s, rng):
    w = s.split(" ")
    idx = [i for i, x in enumerate(w) if len(x) > 3]
    if not idx:
        return s
    i = rng.choice(idx); a = list(w[i]); j = rng.randint(1, len(a) - 2); op = rng.randrange(4)
    if op == 0: a[j], a[j + 1] = a[j + 1], a[j]
    elif op == 1: del a[j]
    elif op == 2: a.insert(j, a[j])
    else: a[j] = chr(97 + rng.randrange(26))
    w[i] = "".join(a)
    return " ".join(w)


def make_example(rng, n_typos=0, want=None):
    """A random practice request and its answer. With want=<answer>: a new way of asking for exactly that game
    (the self-improving loop uses this to drill the AI on the kinds of requests it got wrong)."""
    L, parts = {}, []
    def pick(f):
        if want is not None:
            c = want[f]; L[f] = c
            return None if c == AUTO else rng.choice(SYN[f][c])
        if rng.random() < PROB[f]:
            c = rng.choice(FIELDS[f]); L[f] = c
            return rng.choice(SYN[f][c])
        L[f] = AUTO
        return None
    def put(f, w):
        parts.append(rng.choice(TPL[f]).replace("W", w))
    for f in ("theme", "player"):
        w = pick(f)
        if w: put(f, w)
    en = pick("enemy")
    if en and L["enemy"] != "none":
        ct = pick("count")
        put("enemy", f"{ct} {en}" if ct else en)
    elif en:
        L["count"] = AUTO; parts.append(en)
    elif (want is not None and want["count"] != AUTO) or (want is None and rng.random() < .25):
        c = want["count"] if want is not None else rng.choice(FIELDS["count"]); L["count"] = c
        parts.append(rng.choice(["W", "with W", "add W"]).replace("W", f"{rng.choice(SYN['count'][c])} {rng.choice(GENERIC)}"))
    else:
        L["count"] = AUTO
    for f in ("speed", "goal", "difficulty", "size", "powerup"):
        w = pick(f)
        if w: put(f, w)
    for f in ("behavior", "doors", "levels", "boss", "timer"):  # never contradict the rest of the request
        if want is None and ((f == "behavior" and L["enemy"] == "none") or (f in ("doors", "timer") and L["goal"] == "survive")):
            L[f] = AUTO; continue
        w = pick(f)
        if w: put(f, w)
    rng.shuffle(parts)
    if rng.random() < .4:
        parts.insert(rng.randrange(len(parts) + 1), rng.choice(FILLER))
    s = rng.choice(["", "make a ", "i want a ", "create a ", "give me a ", "top down game ", "a game ", "build me a game ", "can you make "])
    for i, p in enumerate(parts):
        s += (rng.choice([" ", " and ", ", ", " with "]) if i else "") + p
    if want is not None:
        from .recipe import rules_of
        rules = [make_rule(rng, L["goal"], L["enemy"], L["powerup"], want=r) for r in rules_of(want)]
    else:
        k = 0 if rng.random() < .55 else 1 if rng.random() < .73 else 2
        rules = [make_rule(rng, L["goal"], L["enemy"], L["powerup"]) for _ in range(k)]
    for i in range(1, 3):
        r = rules[i - 1][0] if i <= len(rules) else {"trig": "none", "tn": "none", "act": "none", "an": "none"}
        for key, v in r.items():
            L[f"r{i}_{key}"] = v
    if rules:
        texts = [t for _, t in rules]
        if rng.random() < .2:
            s = texts[0] + ". " + s + ("" if len(texts) < 2 else ". " + texts[1])
        else:
            s = s.rstrip() + rng.choice([". ", ", ", " and ", ". also "]) + ". ".join(texts)
    for _ in range(n_typos):
        s = typo(s, rng)
    return {"prompt": s.strip(), "recipe": {f: L[f] for f in OUT_FIELDS}}


def generate(n, seed, typo_mode="mixed"):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if typo_mode == "mixed": k = (1 + rng.randrange(2)) if rng.random() < .3 else 0
        elif typo_mode == "one": k = 1
        else: k = 0
        ex = make_example(rng, k)
        while ex["recipe"]["enemy"] == "none" and ex["recipe"]["goal"] == "survive":  # contradiction: nothing to survive
            ex = make_example(rng, k)
        out.append(ex)
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=200_000)
    ap.add_argument("--val", type=int, default=4_000)
    ap.add_argument("--test", type=int, default=4_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = {"train": generate(a.train, a.seed), "val": generate(a.val, a.seed + 1000, "none"),
            "test_frozen": generate(a.test, 424242, "none"), "test_typo": generate(a.test, 737373, "one")}
    h = hashlib.sha256()
    for name, rows in sets.items():
        write(os.path.join(a.out, f"{name}.jsonl"), rows)
        h.update(json.dumps(rows).encode())
    from .recipe import GRAMMAR_VERSION
    manifest = {k: len(v) for k, v in sets.items()} | {"seed": a.seed, "sha256": h.hexdigest()[:16], "grammar_version": GRAMMAR_VERSION}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print("wrote", manifest)


if __name__ == "__main__":
    main()
