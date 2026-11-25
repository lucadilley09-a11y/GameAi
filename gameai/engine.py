"""Headless game engine: builds a top-down level from a resolved recipe and checks it.
No drawing, so it runs thousands of times faster than the browser version.
The browser engine is the renderer; this one is the referee (doc sections 7, 9)."""
import random
from collections import deque

TS = 32
MAP_SIZE = {"small": (16, 12), "medium": (20, 15), "big": (34, 24)}
DIFF = {"easy": 0.7, "normal": 1.0, "hard": 1.4}
HP = {"easy": 5, "normal": 3, "hard": 2}
PLAYER_SPEED = {"slow": 115, "normal": 150, "fast": 205}
COUNT_BASE = {"few": 3, "some": 6, "many": 10}
SURVIVE_SECONDS = {"easy": 20, "normal": 30, "hard": 45}
TIME_LIMIT = {"none": None, "short": 60, "long": 120}
DOORS_BY_SIZE = {"small": 1, "medium": 2, "big": 3}


def expected_enemies(r):
    if r["enemy"] == "none":
        return 0
    cols, rows = MAP_SIZE[r["size"]]
    return max(1, round(COUNT_BASE[r["count"]] * DIFF[r["difficulty"]] * cols * rows / 300))


def expected_items(r):
    cols, rows = MAP_SIZE[r["size"]]
    if r["goal"] not in ("coins", "gems"):
        return 0
    return round((10 if r["goal"] == "coins" else 6) * (cols * rows / 300) ** 0.5)


def bfs(grid, sx, sy):
    d = [[-1] * len(grid[0]) for _ in grid]
    d[sy][sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] == 0 and d[ny][nx] < 0:
                d[ny][nx] = d[y][x] + 1
                q.append((nx, ny))
    return d


def build(r, seed=0):
    rng = random.Random(seed)
    cols, rows = MAP_SIZE[r["size"]]
    area = cols * rows / 300
    for _ in range(40):
        grid = [[1 if x in (0, cols - 1) or y in (0, rows - 1) else 0 for x in range(cols)] for y in range(rows)]
        blocks = round((36 if r["goal"] == "exit" else 16) * area)
        for _ in range(blocks):
            x, y = 1 + rng.randrange(cols - 2), 1 + rng.randrange(rows - 2)
            ln, hz = 2 + rng.randrange(6 if r["goal"] == "exit" else 4), rng.random() < .5
            for i in range(ln):
                cx, cy = (x + i, y) if hz else (x, y + i)
                if 0 < cx < cols - 1 and 0 < cy < rows - 1:
                    grid[cy][cx] = 1
        for y in range(1, 4):
            for x in range(1, 4):
                grid[y][x] = 0
        dist = bfs(grid, 2, 2)
        cells = [(x, y, dist[y][x]) for y in range(rows) for x in range(cols) if dist[y][x] >= 0]
        if len(cells) > cols * rows * .5:
            break
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == 0 and dist[y][x] < 0:
                grid[y][x] = 1
    used = {(2, 2)}

    def take(min_d):
        opts = [c for c in cells if c[2] >= min_d and (c[0], c[1]) not in used] or cells
        c = rng.choice(opts); used.add((c[0], c[1]))
        return (c[0], c[1])

    rules = r.get("rules", [])
    needs_items = max([int(x["tn"]) for x in rules if x["trig"] == "score"] or [0])
    uses_collect = any(x["trig"] in ("collect", "score") for x in rules)
    lvl = {"rules": rules, "recipe": dict(r), "cols": cols, "rows": rows, "grid": grid, "start": (2, 2),
           "hp": HP[r["difficulty"]], "player_speed": PLAYER_SPEED[r["speed"]],
           "items": [], "exit": None, "powerups": [], "enemies": [],
           "survive_seconds": SURVIVE_SECONDS[r["difficulty"]] if r["goal"] == "survive" else None}
    n_items = max(expected_items(r), needs_items)
    for _ in range(n_items):
        lvl["items"].append(take(3))
    # rules that count collected things need something to collect: keys when the goal is not coins/gems
    lvl["keys"] = [take(3) for _ in range(max(3, needs_items))] if uses_collect and r["goal"] not in ("coins", "gems") else []
    lvl["exit_locked"] = any(x["act"] == "open_exit" for x in rules)
    if r["goal"] == "exit" or lvl["exit_locked"]:
        far = max(cells, key=lambda c: c[2]); lvl["exit"] = (far[0], far[1]); used.add(lvl["exit"])
    pu = r["powerup"]
    if pu == "none" and any(x["trig"] == "powerup" for x in rules):
        pu = "hearts"  # a 'when you grab a power-up' rule needs power-ups on the map
    if pu != "none":
        lvl["powerups"] = [(pu, *take(4)) for _ in range(5 if r["size"] == "big" else 3)]
    lvl["enemies"] = [(r["enemy"], *take(7)) for _ in range(expected_enemies(r))]
    # step 4: the bigger game language (the browser engine builds the details; this records what the level has)
    lvl["behavior"] = r.get("behavior", "chase") if lvl["enemies"] else None
    lvl["doors"] = r.get("doors", "none"); lvl["n_doors"] = DOORS_BY_SIZE[r["size"]] if lvl["doors"] != "none" else 0
    lvl["levels"] = int(r.get("levels", "1")); lvl["boss"] = r.get("boss", "no") == "yes"
    lvl["time_limit"] = TIME_LIMIT[r.get("timer", "none")] if r["goal"] != "survive" else None
    return lvl


def check(lvl):
    """Exact checks: everything the player must reach is reachable. Returns (ok, problems)."""
    g, problems = lvl["grid"], []
    d = bfs(g, *lvl["start"])
    for (x, y) in lvl["items"]:
        if d[y][x] < 0: problems.append(f"item at {x},{y} unreachable")
    if lvl["exit"] and d[lvl["exit"][1]][lvl["exit"][0]] < 0:
        problems.append("exit unreachable")
    for (_, x, y) in lvl["powerups"]:
        if d[y][x] < 0: problems.append(f"powerup at {x},{y} unreachable")
    for (_, x, y) in lvl["enemies"]:
        if g[y][x] != 0: problems.append(f"enemy inside wall at {x},{y}")
    for (x, y) in lvl.get("keys", []):
        if d[y][x] < 0: problems.append(f"key at {x},{y} unreachable")
    r = lvl["recipe"]
    # bug hunter, static part: every rule must be able to happen
    collectibles = len(lvl["items"]) + len(lvl.get("keys", []))
    for rule in lvl.get("rules", []):
        t = rule["trig"]
        if t in ("collect", "score") and collectibles == 0: problems.append(f"rule '{t}' can never happen: nothing to collect")
        if t == "score" and int(rule["tn"]) > collectibles: problems.append(f"rule needs {rule['tn']} collected but only {collectibles} exist")
        if t == "hit" and not lvl["enemies"]: problems.append("rule 'when you get hit' can never happen: no enemies")
        if t == "powerup" and not lvl["powerups"]: problems.append("rule 'when you grab a power-up' can never happen: no power-ups")
        if rule["act"] in ("faster", "slower", "freeze") and not lvl["enemies"] and not any(x["act"] == "spawn" for x in lvl["rules"]):
            problems.append(f"rule action '{rule['act']}' does nothing: no enemies")
    if r["goal"] == "exit" and lvl.get("exit_locked") and not any(x["act"] == "open_exit" for x in lvl["rules"]):
        problems.append("exit is locked forever")
    if r["goal"] in ("coins", "gems") and not lvl["items"]: problems.append("collect goal with no items")
    if r["goal"] == "exit" and not lvl["exit"]: problems.append("exit goal with no exit")
    if r["goal"] == "survive" and r["enemy"] == "none": problems.append("survive goal with no enemies")
    return (not problems), problems


def instruction_match(requested, lvl):
    """For every field the prompt actually asked for, does the BUILT level have it?
    Reads the level itself, not the recipe, so engine bugs are caught too."""
    r = lvl["recipe"]
    ok = total = 0
    from .recipe import rules_of, FIELD_NAMES
    want_rules = rules_of(requested)
    for rule in want_rules:
        total += 1; ok += rule in lvl.get("rules", [])
    for f, want in requested.items():
        if want == "auto" or f not in FIELD_NAMES:
            continue
        total += 1
        if f == "theme": got = r["theme"] == want
        elif f == "player": got = r["player"] == want
        elif f == "enemy": got = (len(lvl["enemies"]) == 0) if want == "none" else bool(lvl["enemies"]) and all(e[0] == want for e in lvl["enemies"])
        elif f == "count": got = len(lvl["enemies"]) == expected_enemies(dict(r, count=want))
        elif f == "speed": got = lvl["player_speed"] == PLAYER_SPEED[want]
        elif f == "goal": got = r["goal"] == want and (bool(lvl["items"]) if want in ("coins", "gems") else lvl["exit"] is not None if want == "exit" else lvl["survive_seconds"] is not None)
        elif f == "difficulty": got = lvl["hp"] == HP[want]
        elif f == "size": got = (lvl["cols"], lvl["rows"]) == MAP_SIZE[want]
        elif f == "behavior": got = lvl.get("behavior") == want
        elif f == "doors": got = lvl.get("doors") == want and lvl.get("n_doors", 0) > 0
        elif f == "levels": got = lvl.get("levels") == int(want)
        elif f == "boss": got = bool(lvl.get("boss"))
        elif f == "timer": got = lvl.get("time_limit") == TIME_LIMIT[want] or r["goal"] == "survive"
        elif f == "powerup": got = (not lvl["powerups"] or any(x["trig"] == "powerup" for x in lvl.get("rules", []))) if want == "none" else bool(lvl["powerups"]) and all(p[0] == want for p in lvl["powerups"])
        ok += got
    return ok, total
