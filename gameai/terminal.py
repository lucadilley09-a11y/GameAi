"""The Game AI terminal.  Start with:  python -m gameai   (or double-click start.bat)

Type a game idea to get a recipe. Commands start with '/'; type /help to see them."""
import json, os, random, subprocess, sys, time, webbrowser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .recipe import FIELD_NAMES, AUTO, resolve, to_text
from . import engine

C = Console()
ACC = "bright_green"
BANNER = r"""
  ____                         _    ___
 / ___| __ _ _ __ ___   ___   / \  |_ _|
| |  _ / _` | '_ ` _ \ / _ \ / _ \  | |
| |_| | (_| | | | | | |  __// ___ \ | |
 \____|\__,_|_| |_| |_|\___/_/   \_\___|
"""
HELP = [
    ("<your idea>", "make a game from a description, e.g.  a fast ninja in a spooky dungeon"),
    ("/edit <change>", "change the last game, e.g.  /edit make it harder"),
    ("/play", "play the last game right here in the terminal"),
    ("/open", "open the last game in your browser"),
    ("/save [name]", "save the last game as one HTML file in the games folder"),
    ("/rate <1-5> [note]", "rate the last game; ratings will teach it what is fun (Phase 3)"),
    ("/train [size]", "train a new AI from scratch: tiny, small (default), base, large"),
    ("/test", "run the benchmark on the latest trained AI"),
    ("/runs", "list every training and test run"),
    ("/export", "make the browser/Chromebook version"),
    ("/status", "show which AI is loaded and how good it is"),
    ("/clear", "clear the screen"),
    ("/quit", "leave"),
]


class App:
    def __init__(self):
        self.model = self.tok = None
        self.device = "cpu"
        self.last = None          # {"prompt", "pred", "recipe"(resolved)}
        self.load()

    # ---------- model ----------
    def load(self):
        if not os.path.exists("models/current.pt"):
            self.model = None
            return
        import torch
        from .generate import load
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, self.tok, _ = load("models/current.pt", self.device)
        except ValueError:
            self.model = None  # trained on an older game language: /train again

    def stats(self):
        p = "models/current.json"
        return json.load(open(p)) if os.path.exists(p) else None

    # ---------- screens ----------
    def banner(self):
        C.print(Text(BANNER, style=ACC))
        C.print("  your own game-making AI, trained from scratch on your PC\n", style="dim")
        self.status()
        C.print("  Type a game idea, or [bold]/help[/bold].\n", style="dim")

    def status(self):
        s = self.stats()
        if not self.model or not s:
            C.print(Panel("No trained AI yet. Type [bold]/train[/bold] to train one (about 10-30 minutes on your GPU).",
                          border_style="yellow", expand=False))
            return
        f = s["fresh"]
        gpu = "GPU" if self.device == "cuda" else "CPU"
        C.print(Panel(f"AI loaded on {gpu}  |  {self.model.num_params()/1e6:.1f}M parameters  |  "
                      f"{f['exact']:.1%} of new prompts fully right  |  {f['instruction_match']:.1%} of requests built",
                      border_style=ACC, expand=False))

    def show_help(self):
        t = Table(show_header=False, box=None, padding=(0, 2))
        for a, b in HELP:
            t.add_row(f"[bold {ACC}]{a}[/]", b)
        C.print(t)

    # ---------- making games ----------
    def make(self, prompt, edit=False):
        if not self.model:
            C.print("No trained AI yet. Type [bold]/train[/bold] first.", style="yellow"); return
        if edit and not self.last:
            C.print("No game to change yet. Describe one first.", style="yellow"); return
        from .generate import decode
        t0 = time.time()
        pred = decode(self.model, self.tok, [prompt], self.device)[0]
        ms = (time.time() - t0) * 1000
        rec = {f: pred[f][0] for f in FIELD_NAMES}
        base = self.last["recipe"] if edit else None
        r = resolve(rec, random.Random(), base)
        lvl = engine.build(r, seed=random.randrange(10**6))
        ok, problems = engine.check(lvl)

        t = Table(title=f"[bold]{'Changed game' if edit else 'New game'}[/]  [dim]({ms:.0f} ms)[/]", title_justify="left",
                  show_edge=False, header_style="dim")
        t.add_column("part"); t.add_column("value"); t.add_column("sure", justify="right")
        for f in FIELD_NAMES:
            v, c = pred[f]
            if v == AUTO:
                val = f"[dim]{'kept' if edit else 'auto'} -> {r[f]}[/]"
                sure = ""
            else:
                col = ACC if c >= .9 else "yellow" if c >= .6 else "red"
                val, sure = f"[bold]{r[f]}[/]", f"[{col}]{c:.0%}[/]"
            t.add_row(f, val, sure)
        C.print(t)
        C.print(f"  engine check: [{ACC}]playable, everything reachable[/]" if ok else f"  engine check: [red]{'; '.join(problems)}[/]")
        C.print("  [dim]/play here  |  /open in browser  |  /save  |  /edit <change>  |  /rate 1-5[/]\n")
        self.last = {"prompt": prompt, "recipe": r, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.makedirs("memory", exist_ok=True)
        with open("memory/episodes.jsonl", "a") as fh:  # episode log (doc section 11)
            fh.write(json.dumps({**self.last, "edit": edit, "ai_recipe": to_text(rec), "engine_ok": ok}) + "\n")

    def game_html(self):
        from .export_web import play_template
        tpl = play_template()
        return tpl.replace("/*GAMEAI_RECIPE*/", "window.GAMEAI_RECIPE=" + json.dumps(self.last["recipe"]) + ";")

    def open_browser(self):
        if not self.last:
            C.print("Make a game first.", style="yellow"); return
        os.makedirs("play", exist_ok=True)
        path = os.path.abspath("play/game.html")
        open(path, "w", encoding="utf-8").write(self.game_html())
        webbrowser.open("file:///" + path.replace("\\", "/"))

    def play(self):
        if not self.last:
            C.print("Make a game first.", style="yellow"); return
        from .tplay import play
        play(self.last["recipe"], self.open_browser)
        C.print("  back in the terminal  [dim]/open for the browser version  |  /save to keep it[/]\n")

    def save(self, name):
        if not self.last:
            C.print("Make a game first.", style="yellow"); return
        r = self.last["recipe"]
        name = "".join(ch for ch in (name or f"{r['player']}-{r['theme']}-{time.strftime('%H%M%S')}") if ch.isalnum() or ch in "-_") or "game"
        os.makedirs("games", exist_ok=True)
        path = os.path.abspath(f"games/{name}.html")
        open(path, "w", encoding="utf-8").write(self.game_html())
        C.print(f"  saved: {path}\n  [dim]a single file: double-click to play, or send it to a friend[/]\n")

    def rate(self, arg):
        if not self.last:
            C.print("Make and play a game first.", style="yellow"); return
        parts = arg.split(maxsplit=1)
        if not parts or parts[0] not in "12345" or len(parts[0]) != 1:
            C.print("Use: /rate 4   or   /rate 2 too hard", style="yellow"); return
        os.makedirs("memory", exist_ok=True)
        with open("memory/ratings.jsonl", "a") as fh:
            fh.write(json.dumps({**self.last, "rating": int(parts[0]), "note": parts[1] if len(parts) > 1 else ""}) + "\n")
        n = sum(1 for _ in open("memory/ratings.jsonl"))
        C.print(f"  saved. {n} rating{'s' if n != 1 else ''} so far; Phase 3 trains on them.\n", style=ACC)

    # ---------- long jobs ----------
    def job(self, args, title):
        C.print(Panel(title, border_style=ACC, expand=False))
        code = subprocess.run([sys.executable, "-m", *args]).returncode
        if code:
            C.print("That step failed. Copy the text above and send it to Claude.", style="red")
        return code == 0

    def train(self, size):
        if size not in ("tiny", "small", "base", "large"):
            C.print("Size must be tiny, small, base or large.", style="yellow"); return
        if self.job(["gameai.all", "--size", size], f"Training a {size} AI from scratch, then testing it"):
            self.load(); self.status()

    # ---------- loop ----------
    def run(self):
        C.clear(); self.banner()
        while True:
            try:
                line = C.input(f"[bold {ACC}]> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            cmd, _, arg = line.partition(" ")
            cmd = cmd.lower()
            if not line.startswith("/"): self.make(line)
            elif cmd in ("/quit", "/exit", "/q"): break
            elif cmd == "/help": self.show_help()
            elif cmd == "/edit": self.make(arg, edit=True) if arg else C.print("Use: /edit make it harder", style="yellow")
            elif cmd == "/play": self.play()
            elif cmd == "/open": self.open_browser(); C.print("  opened in your browser\n")
            elif cmd == "/save": self.save(arg.strip())
            elif cmd == "/rate": self.rate(arg)
            elif cmd == "/train": self.train(arg.strip() or "small")
            elif cmd == "/test":
                if self.job(["gameai.bench"], "Testing the latest trained AI"): self.load()
            elif cmd == "/runs": self.job(["gameai.track"], "All runs")
            elif cmd == "/export": self.job(["gameai.export_onnx"], "Making the browser version")
            elif cmd == "/status": self.status()
            elif cmd == "/clear": C.clear(); self.banner()
            else: C.print(f"Unknown command {cmd}. Type /help.", style="yellow")
        C.print("bye!", style="dim")


def main():
    App().run()


if __name__ == "__main__":
    main()
