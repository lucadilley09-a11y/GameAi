"""The recipe language: the formal grammar the AI writes in.

Grammar (EBNF). The AI may only emit sentences of this grammar; constrained
decoding enforces it, so every output is valid by construction.

    recipe     := theme player enemy count speed goal difficulty size powerup
    theme      := "theme=" ( forest | dungeon | space | desert | snow | ocean | volcano | auto )
    player     := "player=" ( knight | cat | robot | wizard | dog | ninja | frog | auto )
    enemy      := "enemy=" ( bats | slimes | skeletons | aliens | ghosts | spiders | dragons | none | auto )
    count      := "count=" ( few | some | many | auto )
    speed      := "speed=" ( slow | fast | auto )
    goal       := "goal=" ( coins | gems | exit | survive | auto )
    difficulty := "difficulty=" ( easy | hard | auto )
    size       := "size=" ( small | big | auto )
    powerup    := "powerup=" ( hearts | boots | shield | none | auto )
    behavior   := "behavior=" ( patrol | shoot | wander | auto )      (auto: enemies chase you)
    doors      := "doors=" ( keys | switches | auto )                 (auto: no doors)
    levels     := "levels=" ( 3 | 5 | auto )                          (auto: 1 level)
    boss       := "boss=" ( yes | auto )                              (auto: no boss)
    timer      := "timer=" ( short | long | auto )                    (auto: no time limit)

"auto" means the prompt did not say; the engine fills it in (resolve()).

Phase 2 adds up to two RULES after the recipe:  when <trigger> [n] then <action> [n]
    rule     := trig tn act an
    trig     := none | collect | hit | every | score | powerup
    tn       := none | 3 | 5 | 10 | 20      (every: 5/10/20 seconds; score: 3/5/10 collected)
    act      := none | spawn | faster | slower | boost | heal | teleport | open_exit | freeze
    an       := none | 1 | 3                (spawn: how many)
CONSTRAINTS below make the choices depend on each other, so a rule can never be malformed.
"""
import random

FIELDS = {
    "theme": ["forest", "dungeon", "space", "desert", "snow", "ocean", "volcano"],
    "player": ["knight", "cat", "robot", "wizard", "dog", "ninja", "frog"],
    "enemy": ["bats", "slimes", "skeletons", "aliens", "ghosts", "spiders", "dragons", "none"],
    "count": ["few", "some", "many"],
    "speed": ["slow", "fast"],
    "goal": ["coins", "gems", "exit", "survive"],
    "difficulty": ["easy", "hard"],
    "size": ["small", "big"],
    "powerup": ["hearts", "boots", "shield", "none"],
    # Phase 3 (step 4): a bigger game language
    "behavior": ["patrol", "shoot", "wander"],
    "doors": ["keys", "switches"],
    "levels": ["3", "5"],
    "boss": ["yes"],
    "timer": ["short", "long"],
}
AUTO = "auto"
FIELD_NAMES = list(FIELDS)

THEME_ENEMY = {"forest": "slimes", "dungeon": "skeletons", "space": "aliens", "desert": "bats",
               "snow": "ghosts", "ocean": "slimes", "volcano": "dragons"}


def options(field):
    return FIELDS[field] + [AUTO]


def is_valid(recipe):
    return set(recipe) == set(FIELD_NAMES) and all(recipe[f] in options(f) for f in FIELD_NAMES)


def to_text(recipe):
    return " ".join(f"{f}={recipe[f]}" for f in FIELD_NAMES)


def resolve(recipe, rng=None, base=None):
    """Turn an AI recipe (may contain 'auto') into a concrete game spec.
    base: the previous game's resolved recipe, for edit requests ('kept' fields)."""
    rng = rng or random.Random()
    r = {}
    for f in FIELD_NAMES:
        v = recipe[f]
        r[f] = v if v != AUTO else (base[f] if base else AUTO)
    def fill(f, v):
        if r[f] == AUTO:
            r[f] = v
    fill("theme", rng.choice(FIELDS["theme"]))
    fill("player", rng.choice(FIELDS["player"]))
    fill("enemy", THEME_ENEMY[r["theme"]])
    fill("count", "some"); fill("speed", "normal"); fill("difficulty", "normal")
    fill("size", "medium"); fill("powerup", "none")
    fill("goal", rng.choice(["coins", "gems", "exit", "survive"]))
    fill("behavior", "chase"); fill("doors", "none"); fill("levels", "1"); fill("boss", "no"); fill("timer", "none")
    if r["enemy"] == "none" and recipe["count"] != AUTO and recipe["enemy"] == AUTO:
        r["enemy"] = THEME_ENEMY[r["theme"]]
    if r["enemy"] == "none" and r["goal"] == "survive":
        r["goal"] = "coins"
    return r


# ---------- Phase 2: rules ----------
TRIGS = ["none", "collect", "hit", "every", "score", "powerup"]
ACTS = ["none", "spawn", "faster", "slower", "boost", "heal", "teleport", "open_exit", "freeze"]
RULE_VALUES = {"trig": TRIGS, "tn": ["none", "3", "5", "10", "20"], "act": ACTS, "an": ["none", "1", "3"]}
MAX_RULES = 2
RULE_FIELDS = [f"r{i}_{k}" for i in range(1, MAX_RULES + 1) for k in ("trig", "tn", "act", "an")]
OUT_FIELDS = FIELD_NAMES + RULE_FIELDS
GRAMMAR_VERSION = 3  # 3: behavior, doors, levels, boss, timer (an AI trained on version 2 needs one /train)


def fill_defaults(out):
    """Older saved examples (before a field existed) count as not mentioning it."""
    for f in OUT_FIELDS:
        if f not in out: out[f] = AUTO if f in FIELDS else "none"
    return out


def out_options(field):
    return options(field) if field in FIELDS else RULE_VALUES[field.split("_", 1)[1]]


# slot -> (slot it depends on, {that slot's value: allowed values}); missing value key = all options
TN_BY_TRIG = {"none": ["none"], "collect": ["none"], "hit": ["none"], "powerup": ["none"], "every": ["5", "10", "20"], "score": ["3", "5", "10"]}
CONSTRAINTS = {}
for i in range(1, MAX_RULES + 1):
    CONSTRAINTS[f"r{i}_tn"] = (f"r{i}_trig", TN_BY_TRIG)
    CONSTRAINTS[f"r{i}_act"] = (f"r{i}_trig", {"none": ["none"], **{t: ACTS[1:] for t in TRIGS[1:]}})
    CONSTRAINTS[f"r{i}_an"] = (f"r{i}_act", {"none": ["none"], "spawn": ["1", "3"], **{a: ["none"] for a in ACTS[2:]}})
    if i > 1:
        CONSTRAINTS[f"r{i}_trig"] = (f"r{i-1}_trig", {"none": ["none"]})


def allowed(field, chosen):
    """Values the grammar allows for `field`, given the values already chosen."""
    opts = out_options(field)
    if field in CONSTRAINTS:
        dep, table = CONSTRAINTS[field]
        if dep in chosen and chosen[dep] in table:
            return table[chosen[dep]]
    return opts


def rules_of(out):
    """Pull the rule list out of a full output dict."""
    rules = []
    for i in range(1, MAX_RULES + 1):
        t = out.get(f"r{i}_trig", "none")
        if t != "none":
            rules.append({"trig": t, "tn": out[f"r{i}_tn"], "act": out[f"r{i}_act"], "an": out[f"r{i}_an"]})
    return rules


def is_valid_out(out):
    if not is_valid({f: out[f] for f in FIELD_NAMES}):
        return False
    chosen = {}
    for f in OUT_FIELDS:
        if out.get(f) not in allowed(f, chosen):
            return False
        chosen[f] = out[f]
    return True
