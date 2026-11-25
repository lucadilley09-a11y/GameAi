"""Play a generated game right inside the terminal (tile-based version of the browser engine).
Controls: arrows/WASD move · R restart · O open in browser · Q back to the terminal."""
import os, random, sys, time
from rich.live import Live
from rich.text import Text
from rich.console import Group
from .engine import build, check

FPS = 15
PAL = {"forest": ("#3d7a37", "#24411d"), "dungeon": ("#37323f", "#17141c"), "space": ("#1a1e38", "#3d4270"),
       "desert": ("#d4ad66", "#97703a"), "snow": ("#e6eef5", "#7fa4bf"), "ocean": ("#1d6b86", "#0f4a5e"), "volcano": ("#4a1f17", "#1f0b08")}
PL = {"knight": "🤺", "cat": "🐱", "robot": "🤖", "wizard": "🧙", "dog": "🐶", "ninja": "🥷", "frog": "🐸"}
EN = {"bats": ("🦇", 6, .4), "slimes": ("🟢", 9, 0), "skeletons": ("💀", 7, 0), "aliens": ("👾", 6, 0),
      "ghosts": ("👻", 8, 0), "spiders": ("🦂", 6, .1), "dragons": ("🐲", 8, 0)}  # emoji, ticks per move, randomness
PU = {"hearts": "💖", "boots": "👟", "shield": "💠"}
MOVE = {"slow": 4, "normal": 3, "fast": 2}
DIFF = {"easy": 1.3, "normal": 1.0, "hard": .75}
VIEW_W, VIEW_H = 20, 15


class Keys:
    """Non-blocking key reader for Windows and Linux/Mac terminals."""
    def __enter__(self):
        if os.name != "nt":
            import termios, tty
            self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        if os.name != "nt":
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self):
        out = []
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                c = msvcrt.getwch()
                if c in ("\x00", "\xe0"):
                    out.append({"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch(), ""))
                else:
                    out.append(c.lower())
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1)
                if c == "\x1b" and select.select([sys.stdin], [], [], 0.01)[0]:
                    sys.stdin.read(1); out.append({"A": "up", "B": "down", "D": "left", "C": "right"}.get(sys.stdin.read(1), ""))
                else:
                    out.append(c.lower())
        return out


class Game:
    def __init__(self, r, seed=None):
        self.r, self.seed = r, seed if seed is not None else random.randrange(10**6)
        self.reset()

    def reset(self):
        r = self.r
        L = self.lvl = build(r, self.seed)
        self.g, self.cols, self.rows = L["grid"], L["cols"], L["rows"]
        self.px, self.py = L["start"]
        self.items, self.exit = set(L["items"]), L["exit"]
        self.pups = {(x, y): t for (t, x, y) in L["powerups"]}
        self.enemies = [{"t": t, "x": x, "y": y, "cd": random.randrange(5)} for (t, x, y) in L["enemies"]]
        self.hp, self.inv, self.boost, self.shield, self.cool, self.tick = L["hp"], 0, 0, 0, 0, 0
        self.need, self.state, self.msg = len(self.items), "play", ""

    def free(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows and self.g[y][x] == 0

    def step(self, keys):
        if self.state != "play":
            return
        self.tick += 1
        for k in ("inv", "boost", "shield", "cool"):
            setattr(self, k, max(0, getattr(self, k) - 1))
        d = {"up": (0, -1), "w": (0, -1), "down": (0, 1), "s": (0, 1), "left": (-1, 0), "a": (-1, 0), "right": (1, 0), "d": (1, 0)}
        mv = next((d[k] for k in reversed(keys) if k in d), None)
        if mv and self.cool == 0 and self.free(self.px + mv[0], self.py + mv[1]):
            self.px += mv[0]; self.py += mv[1]
            self.cool = max(1, MOVE[self.r["speed"]] // (2 if self.boost else 1))
        p = (self.px, self.py)
        if p in self.items: self.items.discard(p)
        if p in self.pups:
            t = self.pups.pop(p)
            if t == "hearts": self.hp += 1
            elif t == "boots": self.boost = 6 * FPS
            else: self.shield = 6 * FPS
        for e in self.enemies:
            em, every, rnd = EN[e["t"]]
            e["cd"] -= 1
            if e["cd"] > 0: continue
            e["cd"] = max(2, round(every * DIFF[self.r["difficulty"]]))
            dx, dy = self.px - e["x"], self.py - e["y"]
            chase = abs(dx) + abs(dy) < 9 and not self.shield and random.random() >= rnd
            opts = ([(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)] if abs(dx) >= abs(dy) else [(0, 1 if dy > 0 else -1), (1 if dx > 0 else -1, 0)]) if chase else []
            opts += random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4)
            for ox, oy in opts:
                nx, ny = e["x"] + ox, e["y"] + oy
                if (ox, oy) != (0, 0) and (self.free(nx, ny) or (e["t"] == "ghosts" and 0 < nx < self.cols - 1 and 0 < ny < self.rows - 1)):
                    e["x"], e["y"] = nx, ny; break
            if (e["x"], e["y"]) == (self.px, self.py) and not self.inv and not self.shield:
                self.hp -= 1; self.inv = FPS
        if self.hp <= 0: self.state = "lose"
        goal = self.r["goal"]
        if goal in ("coins", "gems") and not self.items: self.state = "win"
        if goal == "exit" and p == self.exit: self.state = "win"
        if goal == "survive" and self.tick >= self.lvl["survive_seconds"] * FPS: self.state = "win"

    def render(self):
        floor, wall = PAL[self.r["theme"]]
        x0 = min(max(self.px - VIEW_W // 2, 0), max(0, self.cols - VIEW_W))
        y0 = min(max(self.py - VIEW_H // 2, 0), max(0, self.rows - VIEW_H))
        en = {(e["x"], e["y"]): EN[e["t"]][0] for e in self.enemies}
        item = "🪙" if self.r["goal"] == "coins" else "💎"
        t = Text()
        for y in range(y0, min(self.rows, y0 + VIEW_H)):
            for x in range(x0, min(self.cols, x0 + VIEW_W)):
                if (x, y) == (self.px, self.py) and not (self.inv and self.tick % 4 < 2):
                    t.append(PL[self.r["player"]], style=f"on {floor}")
                elif (x, y) in en: t.append(en[(x, y)], style=f"on {floor}")
                elif self.g[y][x]: t.append("  ", style=f"on {wall}")
                elif (x, y) in self.items: t.append(item, style=f"on {floor}")
                elif (x, y) in self.pups: t.append(PU[self.pups[(x, y)]], style=f"on {floor}")
                elif (x, y) == self.exit: t.append("🚪", style=f"on {floor}")
                else: t.append("  ", style=f"on {floor}")
            t.append("\n")
        g = self.r["goal"]
        goal = (f"{'Coins' if g == 'coins' else 'Gems'} {self.need - len(self.items)}/{self.need}" if g in ("coins", "gems")
                else "Find the exit 🚪" if g == "exit" else f"Survive {max(0, self.lvl['survive_seconds'] - self.tick / FPS):.0f}s")
        fx = ("  👟" if self.boost else "") + ("  💠" if self.shield else "")
        top = Text(f" {goal}   {'❤️ ' * max(0, self.hp)}{fx}", style="bold")
        if self.state != "play":
            bottom = Text(f" {'YOU WIN!' if self.state == 'win' else 'GAME OVER'}   R play again · O open in browser · Q back",
                          style="bold bright_green" if self.state == "win" else "bold red")
        else:
            bottom = Text(" arrows/WASD move · R restart · O open in browser · Q back", style="dim")
        return Group(top, t, bottom)


def play(r, open_browser):
    """Runs until Q. Returns nothing; calls open_browser() when O is pressed."""
    game = Game(r)
    with Keys() as kb, Live(game.render(), refresh_per_second=FPS, screen=True, transient=True) as live:
        while True:
            keys = kb.read()
            if "q" in keys or "\x1b" in keys: break
            if "r" in keys: game.reset()
            if "o" in keys: open_browser()
            game.step(keys)
            live.update(game.render())
            time.sleep(1 / FPS)
