"""Hand-written coding tasks: the permanent validation anchor for the code skill (plan: Phase 4). They are never trained
on. Each has visible tests (shown to the AI) and hidden tests (never shown) that check the same behaviour on other
inputs, so code that only fits the visible tests fails. `reference` is a correct solution, used only to check the task
itself (it must pass everything, and the starter must fail).

Game-flavoured on purpose: these are the kinds of small pieces the AI writes inside games.
"""

def js(id, prompt, file, reference, tests, hidden, starter=None):
    return {"id": id, "lang": "js", "prompt": prompt, "file": file, "reference": reference, "starter": starter or {},
            "tests": {file.replace(".js", ".test.js"): tests}, "hidden": {file.replace(".js", ".hidden.test.js"): hidden}}


def py(id, prompt, file, reference, tests, hidden, starter=None):
    mod = file[:-3]
    return {"id": id, "lang": "py", "prompt": prompt, "file": file, "reference": reference, "starter": starter or {},
            "tests": {f"test_{mod}.py": tests}, "hidden": {f"test_{mod}_hidden.py": hidden}}


def lua(id, prompt, file, reference, tests, hidden, starter=None):
    mod = file[:-4]
    return {"id": id, "lang": "lua", "prompt": prompt, "file": file, "reference": reference, "starter": starter or {},
            "tests": {f"{mod}_test.lua": tests}, "hidden": {f"{mod}_hidden_test.lua": hidden}}


SEED = [
    # ---------------- JavaScript ----------------
    js("js-clamp-health", "Write `export function heal(hp, amount, maxHp)` in health.js: add amount to hp but never go above maxHp, "
       "and never below 0 (amount can be negative for damage). Return the new hp.", "health.js",
       "export function heal(hp, amount, maxHp) {\n  return Math.max(0, Math.min(maxHp, hp + amount));\n}\n",
       "import { heal } from './health.js';\ntest('heals', () => expect(heal(5, 3, 10)).toBe(8));\ntest('caps at max', () => expect(heal(9, 5, 10)).toBe(10));\n",
       "import { heal } from './health.js';\ntest('damage', () => expect(heal(4, -3, 10)).toBe(1));\ntest('not below zero', () => expect(heal(2, -9, 7)).toBe(0));\n"
       "test('other max', () => expect(heal(1, 100, 3)).toBe(3));\n"),
    js("js-grid-neighbours", "Write `export function neighbours(grid, x, y)` in grid.js. grid is an array of strings, '#' is a wall. "
       "Return the walkable cells next to (x, y) as [x, y] pairs in the order up, right, down, left, skipping walls and cells "
       "outside the grid.", "grid.js",
       "export function neighbours(grid, x, y) {\n  const out = [];\n  for (const [dx, dy] of [[0, -1], [1, 0], [0, 1], [-1, 0]]) {\n"
       "    const nx = x + dx, ny = y + dy;\n    if (ny < 0 || ny >= grid.length || nx < 0 || nx >= grid[ny].length) continue;\n"
       "    if (grid[ny][nx] !== '#') out.push([nx, ny]);\n  }\n  return out;\n}\n",
       "import { neighbours } from './grid.js';\nconst g = ['...', '.#.', '...'];\ntest('corner', () => expect(neighbours(g, 0, 0)).toEqual([[1, 0], [0, 1]]));\n"
       "test('next to wall', () => expect(neighbours(g, 1, 0)).toEqual([[2, 0], [0, 0]]));\n",
       "import { neighbours } from './grid.js';\nconst g = ['#.#', '...', '#.#'];\ntest('centre', () => expect(neighbours(g, 1, 1)).toEqual([[1, 0], [2, 1], [1, 2], [0, 1]]));\n"
       "test('edge', () => expect(neighbours(g, 1, 2)).toEqual([[1, 1]]));\ntest('one row', () => expect(neighbours(['....'], 3, 0)).toEqual([[2, 0]]));\n"),
    js("js-shortest-path", "Write `export function pathLength(grid, start, goal)` in path.js: the number of steps of the shortest path "
       "from start to goal ([x, y] pairs) moving up/down/left/right through cells that aren't '#', or -1 if there is none.",
       "path.js",
       "export function pathLength(grid, start, goal) {\n  const key = (x, y) => x + ',' + y;\n  const seen = new Set([key(...start)]);\n"
       "  let frontier = [start], steps = 0;\n  while (frontier.length) {\n    const next = [];\n    for (const [x, y] of frontier) {\n"
       "      if (x === goal[0] && y === goal[1]) return steps;\n      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {\n"
       "        const nx = x + dx, ny = y + dy;\n        if (ny < 0 || ny >= grid.length || nx < 0 || nx >= grid[ny].length || grid[ny][nx] === '#') continue;\n"
       "        if (!seen.has(key(nx, ny))) { seen.add(key(nx, ny)); next.push([nx, ny]); }\n      }\n    }\n    frontier = next; steps++;\n  }\n  return -1;\n}\n",
       "import { pathLength } from './path.js';\ntest('straight', () => expect(pathLength(['....'], [0, 0], [3, 0])).toBe(3));\n"
       "test('around a wall', () => expect(pathLength(['.#.', '...'], [0, 0], [2, 0])).toBe(4));\n",
       "import { pathLength } from './path.js';\ntest('blocked', () => expect(pathLength(['.#.', '.#.'], [0, 0], [2, 1])).toBe(-1));\n"
       "test('same cell', () => expect(pathLength(['..'], [1, 0], [1, 0])).toBe(0));\n"
       "test('maze', () => expect(pathLength(['...#', '.#..', '...#', '#...'], [0, 0], [3, 3])).toBe(6));\n"
       "test('going left and up', () => expect(pathLength(['....', '.##.', '...#'], [2, 2], [3, 0])).toBe(7));\n"
       "test('never outside the grid', () => expect(pathLength(['.#', '.#', '..'], [0, 0], [1, 2])).toBe(3));\n"
       "test('walls at the edge stay walls', () => expect(pathLength(['..#.', '..#.', '..#.'], [0, 0], [3, 0])).toBe(-1));\n"),
    js("js-score-combo", "Write `export function comboScore(hits)` in combo.js. hits is a list of seconds when the player hit an enemy. "
       "Each hit is worth 10 points times the current combo; the combo starts at 1 and goes up by 1 for each hit less than "
       "1.5 seconds after the previous one, otherwise it goes back to 1. Return the total.", "combo.js",
       "export function comboScore(hits) {\n  let total = 0, combo = 0, last = -Infinity;\n  for (const t of hits) {\n"
       "    combo = t - last < 1.5 ? combo + 1 : 1;\n    total += 10 * combo; last = t;\n  }\n  return total;\n}\n",
       "import { comboScore } from './combo.js';\ntest('no hits', () => expect(comboScore([])).toBe(0));\ntest('combo', () => expect(comboScore([0, 1, 2])).toBe(60));\n",
       "import { comboScore } from './combo.js';\ntest('broken combo', () => expect(comboScore([0, 1, 5, 5.5])).toBe(60));\n"
       "test('just too slow', () => expect(comboScore([0, 1.5, 3])).toBe(30));\ntest('one hit', () => expect(comboScore([7])).toBe(10));\n"),
    js("js-kit-spawn-wave", "Write `function wave(g, n)` in waves.js (a plain script, GameKit is a global). It spawns n enemies that chase "
       "the player plus one heart if n is 5 or more, and returns the enemies it spawned.", "waves.js",
       "function wave(g, n) {\n  const foes = g.spawn('enemy', n, { behaviour: 'chase' });\n  if (n >= 5) g.spawn('heart', 1);\n  return foes;\n}\n",
       "test('three chasers', () => { const g = GameKit.game({ seed: 1 }); g.level('open', { w: 12, h: 10 }); g.player();\n"
       "  const f = wave(g, 3); expect(f.length).toBe(3); expect(g.ents.filter(e => e.kind === 'heart').length).toBe(0); });\n",
       "test('big wave has a heart', () => { const g = GameKit.game({ seed: 2 }); g.level('open', { w: 14, h: 10 }); g.player();\n"
       "  const f = wave(g, 6); expect(f.length).toBe(6); expect(g.ents.filter(e => e.kind === 'heart').length).toBe(1);\n"
       "  expect(g.ents.filter(e => e.kind === 'enemy').length).toBe(6); });\n"
       "test('exactly five', () => { const g = GameKit.game({ seed: 3 }); g.level('open', { w: 14, h: 10 }); g.player(); wave(g, 5);\n"
       "  expect(g.ents.filter(e => e.kind === 'heart').length).toBe(1); });\n"),
    js("js-kit-coin-game", "Write `function build(canvas, seed = 1)` in game.js (a plain script, GameKit is a global): a top-down game "
       "on a 'rooms' level 24x16 with a player, 8 coins, 3 patrolling enemies, and the goal of collecting every coin. Return the game.",
       "game.js",
       "function build(canvas, seed = 1) {\n  const g = GameKit.game({ canvas, seed });\n  g.level('rooms', { w: 24, h: 16 });\n  g.player();\n"
       "  g.spawn('coin', 8);\n  g.spawn('enemy', 3, { behaviour: 'patrol' });\n  g.goal('collect', 'coin');\n  return g;\n}\n",
       "test('has the parts', () => { const g = build(null, 1); expect(g.ents.filter(e => e.kind === 'coin').length).toBe(8);\n"
       "  expect(g.ents.filter(e => e.kind === 'enemy').length).toBe(3); });\n",
       "test('goal and size', () => { const g = build(null, 4); expect(g.goals[0].type).toBe('collect'); expect(g.goals[0].what).toBe('coin');\n"
       "  expect(g.grid[0].length).toBe(24); expect(g.grid.length).toBe(16); expect(g.ents.filter(e => e.kind === 'player').length).toBe(1); });\n"
       "test('enemies patrol', () => { const g = build(null, 7); expect(g.ents.filter(e => e.kind === 'enemy').every(e => e.behaviour === 'patrol' || e.brain === 'patrol')).toBe(true); });\n"),
    js("js-inventory", "Write `export class Inventory` in inventory.js with add(item, count = 1), remove(item, count = 1) -> true if "
       "there were enough (and removes them) else false (and changes nothing), and count(item). Items with a count of 0 disappear "
       "from `items()`, which returns the item names sorted.", "inventory.js",
       "export class Inventory {\n  constructor() { this.c = new Map(); }\n  add(item, count = 1) { this.c.set(item, (this.c.get(item) || 0) + count); }\n"
       "  remove(item, count = 1) { const n = this.c.get(item) || 0; if (n < count) return false; n === count ? this.c.delete(item) : this.c.set(item, n - count); return true; }\n"
       "  count(item) { return this.c.get(item) || 0; }\n  items() { return [...this.c.keys()].sort(); }\n}\n",
       "import { Inventory } from './inventory.js';\ntest('add and count', () => { const i = new Inventory(); i.add('key'); i.add('gem', 3); expect(i.count('gem')).toBe(3); });\n"
       "test('remove', () => { const i = new Inventory(); i.add('gem', 2); expect(i.remove('gem')).toBe(true); expect(i.count('gem')).toBe(1); });\n",
       "import { Inventory } from './inventory.js';\ntest('not enough', () => { const i = new Inventory(); i.add('gem', 1); expect(i.remove('gem', 2)).toBe(false); expect(i.count('gem')).toBe(1); });\n"
       "test('items sorted, empty gone', () => { const i = new Inventory(); i.add('sword'); i.add('apple', 2); i.add('key'); i.remove('key');\n"
       "  expect(i.items()).toEqual(['apple', 'sword']); });\ntest('missing item', () => expect(new Inventory().remove('x')).toBe(false));\n"),
    js("js-fix-collision", "collide.js has a bug: boxes that only touch at an edge count as overlapping, but they shouldn't. Fix "
       "`export function overlaps(a, b)` (boxes are {x, y, w, h}).", "collide.js",
       "export function overlaps(a, b) {\n  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;\n}\n",
       "import { overlaps } from './collide.js';\ntest('overlap', () => expect(overlaps({x:0,y:0,w:10,h:10}, {x:5,y:5,w:10,h:10})).toBe(true));\n"
       "test('touching edge', () => expect(overlaps({x:0,y:0,w:10,h:10}, {x:10,y:0,w:5,h:5})).toBe(false));\n",
       "import { overlaps } from './collide.js';\ntest('touching bottom', () => expect(overlaps({x:0,y:0,w:4,h:4}, {x:0,y:4,w:4,h:4})).toBe(false));\n"
       "test('far', () => expect(overlaps({x:0,y:0,w:1,h:1}, {x:9,y:9,w:1,h:1})).toBe(false));\ntest('inside', () => expect(overlaps({x:0,y:0,w:9,h:9}, {x:2,y:2,w:1,h:1})).toBe(true));\n"
       "test('touching left', () => expect(overlaps({x:5,y:0,w:4,h:4}, {x:1,y:0,w:4,h:4})).toBe(false));\n"
       "test('touching top', () => expect(overlaps({x:0,y:6,w:4,h:4}, {x:0,y:2,w:4,h:4})).toBe(false));\n",
       starter={"collide.js": "export function overlaps(a, b) {\n  return a.x <= b.x + b.w && b.x <= a.x + a.w && a.y <= b.y + b.h && b.y <= a.y + a.h;\n}\n"}),
    # ---------------- Python ----------------
    py("py-level-stats", "Write `level_stats(rows)` in stats.py. rows is a list of strings ('#' wall, 'C' coin, 'E' enemy, '.' floor). "
       "Return a dict with the counts: {'walls': n, 'coins': n, 'enemies': n, 'floor': n} (coins and enemies don't count as floor).",
       "stats.py",
       "def level_stats(rows):\n    text = ''.join(rows)\n    return {'walls': text.count('#'), 'coins': text.count('C'), 'enemies': text.count('E'), 'floor': text.count('.')}\n",
       "from stats import level_stats\n\ndef test_small():\n    assert level_stats(['#C.', '.E#']) == {'walls': 2, 'coins': 1, 'enemies': 1, 'floor': 2}\n",
       "from stats import level_stats\n\ndef test_empty():\n    assert level_stats([]) == {'walls': 0, 'coins': 0, 'enemies': 0, 'floor': 0}\n\n"
       "def test_bigger():\n    assert level_stats(['####', '#CC#', '#.E#', '####']) == {'walls': 12, 'coins': 2, 'enemies': 1, 'floor': 1}\n"),
    py("py-save-format", "Write `encode(save)` and `decode(text)` in save.py. A save is a dict with 'level' (int), 'score' (int) and "
       "'items' (list of str). encode returns one line 'level;score;item1,item2' and decode turns it back into the same dict "
       "(an empty item list encodes as nothing after the last ';').", "save.py",
       "def encode(save):\n    return f\"{save['level']};{save['score']};{','.join(save['items'])}\"\n\n\n"
       "def decode(text):\n    level, score, items = text.split(';')\n    return {'level': int(level), 'score': int(score), 'items': items.split(',') if items else []}\n",
       "from save import encode, decode\n\ndef test_encode():\n    assert encode({'level': 2, 'score': 50, 'items': ['key', 'gem']}) == '2;50;key,gem'\n\n"
       "def test_round_trip():\n    s = {'level': 1, 'score': 0, 'items': ['sword']}\n    assert decode(encode(s)) == s\n",
       "from save import encode, decode\n\ndef test_no_items():\n    assert encode({'level': 9, 'score': 7, 'items': []}) == '9;7;'\n    assert decode('9;7;') == {'level': 9, 'score': 7, 'items': []}\n\n"
       "def test_decode():\n    assert decode('3;120;a,b,c') == {'level': 3, 'score': 120, 'items': ['a', 'b', 'c']}\n"),
    py("py-high-scores", "Write `add_score(table, name, score, size=5)` in scores.py. table is a list of (name, score) tuples sorted by "
       "score, highest first. Return a new table with the score inserted in order (after existing equal scores), keeping at most "
       "`size` entries. Don't change the original list.", "scores.py",
       "def add_score(table, name, score, size=5):\n    out = list(table)\n    i = 0\n    while i < len(out) and out[i][1] >= score:\n        i += 1\n"
       "    out.insert(i, (name, score))\n    return out[:size]\n",
       "from scores import add_score\n\ndef test_insert():\n    assert add_score([('a', 50), ('b', 10)], 'c', 30) == [('a', 50), ('c', 30), ('b', 10)]\n",
       "from scores import add_score\n\ndef test_ties_go_after():\n    assert add_score([('a', 30)], 'b', 30) == [('a', 30), ('b', 30)]\n\n"
       "def test_size():\n    t = [('a', 5), ('b', 4)]\n    assert add_score(t, 'c', 1, size=2) == [('a', 5), ('b', 4)]\n    assert t == [('a', 5), ('b', 4)]\n\n"
       "def test_top():\n    assert add_score([('a', 1)], 'z', 99, size=1) == [('z', 99)]\n"),
    py("py-fix-wrap", "wrap.py has a bug: `wrap(x, width)` should wrap a position around the screen (for example -1 becomes width - 1, "
       "and width becomes 0) but it breaks for numbers more than one screen away. Fix it.", "wrap.py",
       "def wrap(x, width):\n    return x % width\n",
       "from wrap import wrap\n\ndef test_left():\n    assert wrap(-1, 10) == 9\n\ndef test_far_right():\n    assert wrap(25, 10) == 5\n",
       "from wrap import wrap\n\ndef test_far_left():\n    assert wrap(-21, 10) == 9\n\ndef test_inside():\n    assert wrap(3, 7) == 3\n\ndef test_edge():\n    assert wrap(7, 7) == 0\n",
       starter={"wrap.py": "def wrap(x, width):\n    if x < 0:\n        return x + width\n    if x >= width:\n        return x - width\n    return x\n"}),
    # ---------------- Lua ----------------
    lua("lua-cooldown", "Write a module cooldown.lua that returns a table M with `M.new(seconds)` returning a cooldown object with "
        "`:ready()` (true if it can be used), `:use()` (starts the cooldown, returns false if it wasn't ready) and `:update(dt)` "
        "(time passes). A new cooldown is ready.", "cooldown.lua",
        "local M = {}\nM.__index = M\nfunction M.new(seconds) return setmetatable({len = seconds, left = 0}, M) end\n"
        "function M:ready() return self.left <= 0 end\nfunction M:use() if not self:ready() then return false end self.left = self.len return true end\n"
        "function M:update(dt) self.left = self.left - dt end\nreturn M\n",
        "local C = require('cooldown')\ntest('new is ready', function() expect(C.new(2):ready()).toBe(true) end)\n"
        "test('use starts it', function() local c = C.new(2); expect(c:use()).toBe(true); expect(c:ready()).toBe(false) end)\n",
        "local C = require('cooldown')\ntest('ready again', function() local c = C.new(1); c:use(); c:update(0.6); expect(c:ready()).toBe(false); c:update(0.5); expect(c:ready()).toBe(true) end)\n"
        "test('use while cooling', function() local c = C.new(3); c:use(); expect(c:use()).toBe(false) end)\n"),
    lua("lua-lerp-colour", "Write colour.lua returning M with `M.lerp(a, b, t)`: a and b are colours {r, g, b} (0-255), t from 0 to 1. "
        "Return a new colour with each channel blended and rounded to the nearest whole number. t below 0 counts as 0, above 1 as 1.",
        "colour.lua",
        "local M = {}\nlocal function clamp(t) return math.max(0, math.min(1, t)) end\n"
        "function M.lerp(a, b, t)\n  t = clamp(t)\n  local out = {}\n  for i = 1, 3 do out[i] = math.floor(a[i] + (b[i] - a[i]) * t + 0.5) end\n  return out\nend\nreturn M\n",
        "local C = require('colour')\ntest('middle', function() expect(C.lerp({0, 0, 0}, {255, 100, 10}, 0.5)).toEqual({128, 50, 5}) end)\n"
        "test('start', function() expect(C.lerp({1, 2, 3}, {9, 9, 9}, 0)).toEqual({1, 2, 3}) end)\n",
        "local C = require('colour')\ntest('clamped', function() expect(C.lerp({0, 0, 0}, {10, 20, 30}, 2)).toEqual({10, 20, 30}); expect(C.lerp({5, 5, 5}, {9, 9, 9}, -1)).toEqual({5, 5, 5}) end)\n"
        "test('quarter', function() expect(C.lerp({0, 100, 200}, {100, 0, 0}, 0.25)).toEqual({25, 75, 150}) end)\n"),
    lua("lua-spawn-table", "Write loot.lua returning M with `M.pick(table, roll)`: table is a list of {name, weight}, roll a number from "
        "0 (inclusive) to the total weight (exclusive). Return the name whose share of the total contains roll, going through the "
        "list in order.", "loot.lua",
        "local M = {}\nfunction M.pick(t, roll)\n  for _, e in ipairs(t) do\n    if roll < e[2] then return e[1] end\n    roll = roll - e[2]\n  end\n  return t[#t][1]\nend\nreturn M\n",
        "local L = require('loot')\nlocal t = {{'coin', 5}, {'gem', 2}, {'heart', 1}}\ntest('first', function() expect(L.pick(t, 0)).toBe('coin') end)\n"
        "test('second', function() expect(L.pick(t, 5)).toBe('gem') end)\n",
        "local L = require('loot')\nlocal t = {{'a', 1}, {'b', 3}, {'c', 2}}\ntest('edges', function() expect(L.pick(t, 0.99)).toBe('a'); expect(L.pick(t, 3.5)).toBe('b'); expect(L.pick(t, 4)).toBe('c') end)\n"
        "test('last', function() expect(L.pick(t, 5.9)).toBe('c') end)\n"),
    lua("lua-fix-count", "count.lua has a bug: `M.count(list, value)` should count how many times value appears in list, but it "
        "misses the last element. Fix it.", "count.lua",
        "local M = {}\nfunction M.count(list, value)\n  local n = 0\n  for i = 1, #list do if list[i] == value then n = n + 1 end end\n  return n\nend\nreturn M\n",
        "local C = require('count')\ntest('middle', function() expect(C.count({1, 2, 1, 3}, 1)).toBe(2) end)\ntest('last', function() expect(C.count({'a', 'b'}, 'b')).toBe(1) end)\n",
        "local C = require('count')\ntest('only last', function() expect(C.count({0, 0, 7}, 7)).toBe(1) end)\ntest('empty', function() expect(C.count({}, 1)).toBe(0) end)\n"
        "test('all', function() expect(C.count({2, 2, 2}, 2)).toBe(3) end)\n",
        starter={"count.lua": "local M = {}\nfunction M.count(list, value)\n  local n = 0\n  for i = 1, #list - 1 do if list[i] == value then n = n + 1 end end\n  return n\nend\nreturn M\n"}),
]
