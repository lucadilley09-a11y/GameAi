"""UI tasks (plan: Phase 5 session 2; "generators: unlimited verified UI tasks"; "create a polished interface in a
requested style"). A task asks for one game screen in a style; it is checked by what must be on the screen and by the
layout checker at desktop and phone size, so there is no single right answer to copy.

    t = tasks.generate(rng, split="train")   # endless; validation uses its own names, words and phrasings
    tasks.val_set()                          # the fixed validation tasks (never trained on)

A task: {id, kind, theme, prompt, texts: [must appear], need: {css selector: at least n}, reference: body HTML}
"""
import random

THEME_WORDS = {"clean": ["clean modern", "simple, modern", "sleek dark"], "pixel": ["retro pixel", "8-bit style", "old-school pixel"],
               "neon": ["neon", "glowing neon", "synthwave neon"], "cozy": ["cozy", "warm, cozy", "storybook cozy"]}
# separate words for training and validation, so a validation task is never a reworded training task
WORDS = {
    "train": {"games": ["Gem Cave", "Sky Raiders", "Moss Knight", "Robo Rally", "Frost Fang", "Lava Run", "Star Hopper", "Pond Quest"],
              "buttons": ["Play", "Continue", "Settings", "Credits", "Quit", "Load Game", "Options", "Controls"],
              "pause": ["Resume", "Restart", "Settings", "Main Menu", "Quit"],
              "settings": ["Sound", "Music", "Vibration", "Screen shake", "Subtitles", "Fullscreen", "Show FPS"],
              "items": ["Key", "Gem", "Potion", "Shield", "Bomb", "Map", "Coin", "Rope"],
              "names": ["Ava", "Leo", "Mia", "Kai", "Zoe", "Max", "Ivy", "Sam"],
              "speakers": ["Old Miner", "Captain Rook", "The Owl", "Shopkeeper"],
              "lines": ["The bridge is out. Find another way across.", "Bring me three gems and I'll open the gate.", "Watch out, the bats wake up at night."],
              "stats": ["Score", "Lives", "Time", "Coins", "Level"]},
    "val": {"games": ["Dune Drifter", "Coral Keep", "Ember Wing", "Glacier Gate", "Owl Hollow", "Neon Alley"],
            "buttons": ["Start", "New Game", "Achievements", "Exit", "Leaderboard", "Help"],
            "pause": ["Back to game", "Try again", "Options", "Exit to menu"],
            "settings": ["Sound effects", "Background music", "Colour-blind mode", "Auto-save", "Hints"],
            "items": ["Torch", "Feather", "Lantern", "Scroll", "Crystal", "Apple"],
            "names": ["Noa", "Eli", "Rin", "Tom", "Uma", "Jay"],
            "speakers": ["Ranger Fen", "Queen Isla", "Grumpy Troll"],
            "lines": ["Nobody has come back from the north cave.", "Pay the toll, or answer my riddle.", "The tide rises fast here. Hurry!"],
            "stats": ["Points", "Hearts", "Gold", "Stage", "Clock"]},
}
PHRASES = {
    "train": ["Make a {style} {thing} for my game {game}{extra}.", "I need a {style} {thing}{extra}. The game is called {game}.",
              "Design the {thing} for {game} in a {style} style{extra}."],
    "val": ["Could you build {game}'s {thing}? {Style} look{extra}.", "{Style} {thing} please, for a game named {game}{extra}."],
}


def _btns(labels, primary=0):
    return "".join(f'<button class="gui-btn{" gui-primary" if i == primary else ""}">{b}</button>' for i, b in enumerate(labels))


def _title(rng, W):
    game = rng.choice(W["games"]); btns = rng.sample(W["buttons"], rng.randint(3, 4))
    body = f'<div class="gui-screen"><h1 class="gui-title">{game}</h1><p class="gui-subtitle">Press a button to begin</p><div class="gui-menu">{_btns(btns)}</div></div>'
    return game, "title screen", f" with {', '.join(btns[:-1])} and {btns[-1]} buttons", [game] + btns, {"button": len(btns)}, body


def _pause(rng, W):
    game = rng.choice(W["games"]); btns = rng.sample(W["pause"], 3)
    body = f'<div class="gui-screen"><h1 class="gui-title">Paused</h1><div class="gui-menu">{_btns(btns)}</div></div>'
    return game, "pause menu", f" with {', '.join(btns[:-1])} and {btns[-1]}", ["Paused"] + btns, {"button": 3}, body


def _game_over(rng, W):
    game = rng.choice(W["games"]); score = rng.randint(100, 9999); names = rng.sample(W["names"], 3)
    rows = "".join(f"<tr><td>{n}</td><td>{s}</td></tr>" for n, s in zip(names, sorted(rng.sample(range(100, 9999), 3), reverse=True)))
    body = (f'<div class="gui-screen"><h1 class="gui-title">Game Over</h1><p class="gui-subtitle">Your score: {score}</p>'
            f'<div class="gui-panel"><h2>High scores</h2><table class="gui-table"><tr><th>Name</th><th>Score</th></tr>{rows}</table></div>'
            f'<div class="gui-row">{_btns(["Play again", "Menu"])}</div></div>')
    return game, "game over screen", f" that shows the score {score}, a high score table with {', '.join(names)}, and Play again and Menu buttons", \
        ["Game Over", str(score), "Play again", "Menu"] + names, {"table": 1, "button": 2}, body


def _settings(rng, W):
    game = rng.choice(W["games"]); opts = rng.sample(W["settings"], rng.randint(3, 4))
    items = "".join(f'<li class="gui-item"><label for="o{i}">{o}</label><input id="o{i}" type="checkbox" class="gui-toggle"{" checked" if i % 2 == 0 else ""}></li>' for i, o in enumerate(opts))
    body = f'<div class="gui-screen"><div class="gui-panel"><h2>Settings</h2><ul class="gui-list">{items}</ul><div class="gui-row">{_btns(["Back"])}</div></div></div>'
    return game, "settings screen", f" with switches for {', '.join(opts)} and a Back button", ["Settings", "Back"] + opts, {"input": len(opts), "button": 1}, body


def _hud(rng, W):
    game = rng.choice(W["games"]); stats = rng.sample(W["stats"], 3)
    vals = {"Score": "1,240", "Points": "880", "Lives": "3", "Hearts": "4", "Time": "1:32", "Clock": "0:45", "Coins": "27", "Gold": "150", "Level": "4", "Stage": "2"}
    cells = "".join(f'<div class="gui-stat"><span>{s}</span><b>{vals[s]}</b></div>' for s in stats)
    body = f'<div class="gui-hud">{cells}<div class="gui-bar" role="progressbar" aria-label="Health"><i style="width:70%"></i></div></div>'
    return game, "HUD (the bar shown while playing)", f" showing {', '.join(stats)} and a health bar", stats, {".gui-bar,[role=progressbar],progress": 1}, body


def _inventory(rng, W):
    game = rng.choice(W["games"]); items = rng.sample(W["items"], rng.randint(4, 6))
    slots = "".join(f'<div class="gui-slot{" gui-selected" if i == 0 else ""}"><b>■</b>{it}</div>' for i, it in enumerate(items))
    body = f'<div class="gui-screen"><div class="gui-panel"><h2>Inventory</h2><div class="gui-grid">{slots}</div></div></div>'
    return game, "inventory screen", f" with a grid of these items: {', '.join(items)}", ["Inventory"] + items, {}, body


def _dialogue(rng, W):
    game = rng.choice(W["games"]); who = rng.choice(W["speakers"]); line = rng.choice(W["lines"]); choices = rng.sample(["Okay", "Why?", "No thanks", "Tell me more"], 2)
    body = f'<div class="gui-screen"><div class="gui-dialog"><span class="gui-speaker">{who}</span><p>{line}</p><div class="gui-row">{_btns(choices)}</div></div></div>'
    return game, "dialogue box", f" where {who} says \"{line}\" and the player can answer {choices[0]} or {choices[1]}", [who, line] + choices, {"button": 2}, body


def _levels(rng, W):
    game = rng.choice(W["games"]); n = rng.randint(4, 6); unlocked = rng.randint(1, n - 1)
    btns = "".join(f'<button class="gui-btn{" gui-primary" if i == 0 else ""}"{" disabled" if i >= unlocked else ""}>Level {i + 1}{" 🔒" if i >= unlocked else ""}</button>' for i in range(n))
    body = f'<div class="gui-screen"><h1 class="gui-title">Choose a level</h1><div class="gui-row">{btns}</div></div>'
    return game, "level select screen", f" with {n} levels, where only the first {unlocked} are unlocked", [f"Level {i + 1}" for i in range(n)], {"button": n}, body


KINDS = {"title": _title, "pause": _pause, "game_over": _game_over, "settings": _settings, "hud": _hud, "inventory": _inventory,
         "dialogue": _dialogue, "levels": _levels}


def generate(rng, split="train", kind=None, theme=None):
    W = WORDS[split]
    kind = kind or rng.choice(sorted(KINDS)); theme = theme or rng.choice(sorted(THEME_WORDS))
    game, thing, extra, texts, need, body = KINDS[kind](rng, W)
    style = rng.choice(THEME_WORDS[theme])
    prompt = rng.choice(PHRASES[split]).format(style=style, Style=style[:1].upper() + style[1:], thing=thing, game=game, extra=extra)
    return {"id": f"ui-{split}-{kind}-{theme}-{rng.randrange(10**9)}", "kind": kind, "theme": theme, "prompt": prompt, "texts": texts,
            "need": need, "reference": f'<div class="gui" data-theme="{theme}">{body}</div>', "split": split}


def val_set(n=32, seed=4242):
    """fixed validation tasks: every kind in every style (8 x 4 = 32)"""
    rng, out = random.Random(seed), []
    combos = [(k, t) for k in sorted(KINDS) for t in sorted(THEME_WORDS)]
    rng.shuffle(combos)
    for k, t in combos[:n]: out.append(generate(rng, "val", k, t))
    return out
