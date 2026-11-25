"""What every bake-off candidate is tested on, before any fine-tuning. Only VALIDATION material is used here
(validation decides between configurations; the sealed sets are never used to choose a model).

Each test takes ask(messages, **options) -> (text, info) and returns totals. Every check is automatic.
"""
import io, json, math, os, random, re, struct, wave

from ..recipe import FIELDS, FIELD_NAMES

# ---------- 1. reading game requests (the core job) ----------

SETTINGS_HELP = "\n".join(f"{f}: {', '.join(v)}, auto" for f, v in FIELDS.items() if f in FIELD_NAMES)
RECIPE_SYSTEM = ("You turn a game request into settings. For each setting write one line: name=value. Use exactly these names "
                 "and only these values; use auto when the request doesn't say.\n" + SETTINGS_HELP)


def recipe_grammar():
    """llama.cpp grammar: the answer can only be the settings, in order, with allowed values (grammar-locked output)."""
    lines = [f'root ::= {" ".join(f"l{i}" for i in range(len(FIELD_NAMES)))}']
    for i, f in enumerate(FIELD_NAMES):
        alts = " | ".join(f'"{v}"' for v in list(FIELDS[f]) + ["auto"])
        lines.append(f'l{i} ::= "{f}=" ({alts}) "\\n"')
    return "\n".join(lines)


def parse_recipe(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*(\S+)", line)
        if m and m.group(1) in FIELD_NAMES: out[m.group(1)] = m.group(2).strip().lower()
    return out


def recipe_items(n_frozen=100, n_typo=40):
    def rows(p, n=None):
        if not os.path.exists(p): return []
        r = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return r[:n] if n else r
    here = os.path.dirname(os.path.dirname(__file__))
    return rows(os.path.join(here, "gold.jsonl")) + rows("data/test_frozen.jsonl", n_frozen) + rows("data/test_typo.jsonl", n_typo)


def score_recipes(items, preds):
    fields = exact = 0
    for it, p in zip(items, preds):
        want = {f: it["recipe"].get(f, "auto") for f in FIELD_NAMES}
        hits = [p.get(f, "auto") == v for f, v in want.items()]
        fields += sum(hits); exact += all(hits)
    n = max(1, len(items))
    return {"field_accuracy": round(fields / (n * len(FIELD_NAMES)), 4), "fully_right": round(exact / n, 4), "n": len(items)}


def test_recipes(ask, items=None, thinking=None):
    items = items or recipe_items()
    preds, speed = [], []
    for it in items:
        text, info = ask([{"role": "system", "content": RECIPE_SYSTEM}, {"role": "user", "content": it["prompt"]}],
                         max_tokens=200, grammar=recipe_grammar(), thinking=thinking)
        preds.append(parse_recipe(text)); speed.append(info)
    return {**score_recipes(items, preds), **speed_summary(speed)}


def speed_summary(infos):
    def med(k):
        xs = sorted(x[k] for x in infos if x.get(k) is not None)
        return xs[len(xs) // 2] if xs else None
    walls = sorted(x["wall_s"] for x in infos)
    return {"answer_s_median": walls[len(walls) // 2] if walls else None, "answer_s_p95": walls[min(len(walls) - 1, int(.95 * len(walls)))] if walls else None,
            "gen_tok_per_s": med("gen_tok_per_s"), "prompt_tok_per_s": med("prompt_tok_per_s"), "first_token_s": med("first_token_s")}


# ---------- 2. JavaScript that must pass the sandbox ----------

CODE_SYSTEM = ("Write the body of a JavaScript function brain(e, p, t) for an enemy in a 2D game. e is the enemy {x, y, mem}, "
               "p is the player {x, y}, t is seconds since the start. Return the direction to move as [dx, dy]. "
               "Helpers: dist(a, b) is the distance between two points; dir(a, b) is the unit direction from a to b as [x, y]. "
               "Rules: only const, let, if, else and return; Math.sin, cos, atan2, hypot, abs, min, max, sign, sqrt, PI; "
               "no loops, no functions, no strings, at most 400 characters. Reply with only the code, no explanation.")


def test_code(ask, n=40, seed=11):
    from ..code import sandbox, specs
    rng, ok, why = random.Random(seed), 0, {}
    for i in range(n):
        s = specs.random_spec(rng)
        text, _ = ask([{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": "Make the enemy " + specs.say(s, rng)}], max_tokens=300)
        code = re.sub(r"^```\w*\n|```\s*$", "", text.strip(), flags=re.M).strip()
        code = re.sub(r"^\s*function\s+\w*\s*\([^)]*\)\s*\{([\s\S]*)\}\s*$", r"\1", code).strip()  # a whole function: keep its body
        good, reason = sandbox.verify(specs.spec_text(s), code, seed=i)
        ok += good
        if not good: why[reason.split(":")[0][:40]] = why.get(reason.split(":")[0][:40], 0) + 1
    return {"passes_sandbox": round(ok / n, 4), "n": n, "top_failures": sorted(why.items(), key=lambda x: -x[1])[:4]}


# ---------- 3. following instructions (each one checkable) ----------

INSTR = [
    ("List three fruits as a numbered list and nothing else.", lambda t: len(re.findall(r"^\s*\d[.)]", t, re.M)) == 3),
    ("Answer with exactly one word: what color is the sky on a clear day?", lambda t: len(t.strip().split()) == 1 and "blue" in t.lower()),
    ("Write a sentence about cats that does not use the letter e.", lambda t: len(t.strip()) > 10 and "e" not in t.lower()),
    ("Reply in all capital letters: say hello to the team.", lambda t: t.strip() and t.upper() == t and "HELLO" in t),
    ("Give me a JSON object with keys name and age for a 30 year old called Sam. Only the JSON.", lambda t: _json_ok(t, {"name": "Sam", "age": 30})),
    ("Write exactly two sentences about the moon.", lambda t: len(re.findall(r"[.!?](\s|$)", t.strip())) == 2),
    ("Name a planet. Put your answer inside square brackets, like [answer].", lambda t: bool(re.search(r"\[\s*(mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)\s*\]", t.lower()))),
    ("Count from 1 to 5, separated by commas, nothing else.", lambda t: re.sub(r"\s", "", t.strip().rstrip(".")) == "1,2,3,4,5"),
    ("Describe a game idea in under 15 words.", lambda t: 0 < len(t.split()) < 15),
    ("End your answer with the word DONE. What is 2 plus 2?", lambda t: t.strip().rstrip(".!").endswith("DONE") and "4" in t),
    ("Write the word 'banana' backwards and nothing else.", lambda t: t.strip().strip("'\".").lower() == "ananab"),
    ("Give three words that rhyme with cat, separated by spaces only.", lambda t: len(t.split()) == 3 and all(w.lower().strip(",.").endswith("at") for w in t.split())),
]


def _json_ok(t, want):
    m = re.search(r"\{[\s\S]*\}", t)
    try: return m and json.loads(m.group(0)) == want
    except Exception: return False


def test_instructions(ask, thinking=None):
    ok = sum(bool(check(ask([{"role": "user", "content": q}], max_tokens=150, thinking=thinking)[0])) for q, check in INSTR)
    return {"follows_instructions": round(ok / len(INSTR), 4), "n": len(INSTR)}


# ---------- 4. other languages ----------

LANGS = {"Spanish": {"el", "la", "que", "es", "de", "un", "una", "y", "en", "los"}, "French": {"le", "la", "est", "et", "un", "une", "les", "des", "de", "que"},
         "German": {"der", "die", "das", "ist", "und", "ein", "eine", "nicht", "mit", "zu"}, "Portuguese": {"o", "a", "que", "é", "de", "um", "uma", "e", "os", "não"},
         "Italian": {"il", "la", "che", "è", "di", "un", "una", "e", "per", "non"}}
TOPICS = ["Explain in two sentences why the sun rises.", "Describe your favourite kind of video game in two sentences."]


def test_languages(ask):
    res = {}
    for lang, words in LANGS.items():
        ok = 0
        for q in TOPICS:
            t = ask([{"role": "user", "content": f"Answer only in {lang}. {q}"}], max_tokens=150)[0].lower()
            w = re.findall(r"\w+", t); ok += bool(w) and sum(x in words for x in w) / len(w) > 0.12
        res[lang] = ok / len(TOPICS)
    return {"languages": round(sum(res.values()) / len(res), 4), "by_language": res}


# ---------- 5. knowledge and reasoning (short answers that can be checked) ----------

QA = [("What is the capital of France?", ["paris"]), ("How many legs does a spider have?", ["8", "eight"]),
      ("What gas do plants take in from the air?", ["carbon dioxide", "co2"]), ("Who wrote Romeo and Juliet?", ["shakespeare"]),
      ("What is the largest planet in our solar system?", ["jupiter"]), ("What is H2O commonly called?", ["water"]),
      ("A game has 3 levels with 12 coins each. How many coins in total?", ["36"]), ("If a train goes 60 km in 1 hour, how far does it go in 2.5 hours?", ["150"]),
      ("What is 17 times 6?", ["102"]), ("I have 5 apples, eat 2 and buy 7. How many now?", ["10"]),
      ("Which is heavier: a kilogram of feathers or a kilogram of iron?", ["same", "equal", "neither", "both"]),
      ("What is the next number: 2, 4, 8, 16, ...?", ["32"]), ("What language is mostly used to make web pages interactive?", ["javascript"]),
      ("In what year did humans first land on the Moon?", ["1969"]), ("What is the boiling point of water at sea level in Celsius?", ["100"]),
      ("A rectangle is 4 by 7. What is its area?", ["28"])]


def test_knowledge(ask, thinking=None):
    ok = 0
    for q, answers in QA:
        t = ask([{"role": "user", "content": q + " Answer briefly."}], max_tokens=120, thinking=thinking)[0].lower()
        ok += any(a in t for a in answers)
    return {"knowledge_and_reasoning": round(ok / len(QA), 4), "n": len(QA)}


# ---------- 6. interfaces (HTML) ----------

UI = [("a login form with an email box, a password box and a sign-in button", ["form", "input[type=email]", "input[type=password]", "button"]),
      ("a game menu with three buttons: Play, Settings, Quit", ["button:Play", "button:Settings", "button:Quit"]),
      ("a score panel showing Score and Lives with a pause button", ["text:score", "text:lives", "button"]),
      ("a settings page with a volume slider and a checkbox for music", ["input[type=range]", "input[type=checkbox]"]),
      ("a shop list with three items, each with a price and a Buy button", ["button:Buy", "button:Buy", "button:Buy"]),
      ("a level select screen with buttons for levels 1 to 4", ["button:1", "button:2", "button:3", "button:4"])]


def _html_has(html, need):
    from html.parser import HTMLParser
    class P(HTMLParser):
        def __init__(s): super().__init__(); s.tags, s.cur, s.text = [], None, ""
        def handle_starttag(s, tag, attrs): s.tags.append([tag, dict(attrs), ""]); s.cur = s.tags[-1]
        def handle_data(s, d):
            s.text += d
            if s.cur is not None: s.cur[2] += d
    p = P()
    try: p.feed(html)
    except Exception: return False
    used = set()
    for n in need:
        if n.startswith("text:"):
            if n[5:] not in p.text.lower(): return False
            continue
        m = re.match(r"(\w+)(?:\[type=(\w+)\])?(?::(.+))?", n); tag, typ, txt = m.groups()
        hit = next((i for i, (t, a, x) in enumerate(p.tags) if i not in used and t == tag and (not typ or a.get("type") == typ)
                    and (not txt or txt.lower() in (x or a.get("value", "")).lower())), None)
        if hit is None: return False
        used.add(hit)
    return True


def test_ui(ask):
    ok = 0
    for what, need in UI:
        t = ask([{"role": "user", "content": f"Write a complete HTML page for {what}. Reply with only the HTML."}], max_tokens=900)[0]
        ok += _html_has(t, need)
    return {"ui_html": round(ok / len(UI), 4), "n": len(UI)}


# ---------- 7. pictures (self-labelled: we draw them, so the right answer is known) ----------

def _image(seed):
    from PIL import Image, ImageDraw
    rng = random.Random(seed); W, H, T = 320, 240, 32
    im = Image.new("RGB", (W, H), (30, 34, 40)); d = ImageDraw.Draw(im)
    n_red, n_yellow = rng.randint(1, 6), rng.randint(0, 5)
    cells = rng.sample([(x, y) for x in range(W // T) for y in range(H // T)], n_red + n_yellow + 1)
    for i, (x, y) in enumerate(cells):
        col = (220, 40, 40) if i < n_red else (240, 210, 40) if i < n_red + n_yellow else (60, 120, 240)
        d.rectangle([x * T + 4, y * T + 4, x * T + T - 4, y * T + T - 4], fill=col)
    px, py = cells[-1]
    buf = io.BytesIO(); im.save(buf, "PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode(), {"red": n_red, "yellow": n_yellow, "blue_side": "left" if px < (W // T) / 2 else "right"}


def test_vision(ask, n=12):
    ok = tot = 0
    for i in range(n):
        b64, truth = _image(100 + i)
        for q, key in (("How many red squares are in this picture? Answer with a number.", "red"),
                       ("How many yellow squares are in this picture? Answer with a number.", "yellow"),
                       ("Is the blue square in the left half or the right half of the picture? Answer left or right.", "blue_side")):
            t = ask([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}, {"type": "text", "text": q}]}], max_tokens=20)[0].lower()
            want = str(truth[key]); words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six"}
            ok += bool(re.search(rf"\b({want}|{words.get(want, want)})\b", t)); tot += 1
    return {"vision": round(ok / tot, 4), "n": tot}


# ---------- 8. sound (self-labelled beeps) ----------

def _beeps(k, rate=16000):
    buf = io.BytesIO(); w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    frames = b""
    for i in range(k):
        frames += b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * 880 * j / rate))) for j in range(int(rate * .18)))
        frames += b"\x00\x00" * int(rate * .32)
    w.writeframes(frames); w.close()
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def test_audio(ask):
    ok = 0
    for k in (1, 2, 3, 4, 5, 3):
        t = ask([{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": _beeps(k), "format": "wav"}},
                                               {"type": "text", "text": "How many beeps are in this sound? Answer with a number."}]}], max_tokens=20)[0].lower()
        ok += str(k) in t
    return {"audio": round(ok / 6, 4), "n": 6}
