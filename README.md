# Game AI — Phase 1

Your own game-designing AI, written and trained from scratch. Phase 1 of the plan in
"Research: A Local AI for Autonomous Game Development" (section 17).

## What's here

| File | What it does | Doc section |
|---|---|---|
| `gameai/recipe.py` | The recipe language as a formal grammar | 4, 17 |
| `gameai/datagen.py` | Makes training data, including typos and slang; frozen test sets | 2, 13 |
| `gameai/tokenizer.py` | Our own tokenizer (letters in, recipe tokens out) | 1 |
| `gameai/model.py` | Transformer from scratch: RoPE, RMSNorm, SwiGLU, grouped-query attention | 1 |
| `gameai/muon.py` | Muon optimizer (about 2x less training compute than AdamW) | 2 |
| `gameai/train.py` | Training: BF16, warmup-stable-decay schedule, early stopping, best checkpoint | 2, 10 |
| `gameai/generate.py` | Grammar-constrained generation: every output is valid | 3 |
| `gameai/engine.py` | Headless game engine + exact checks (reachability, instruction match) | 7, 9 |
| `gameai/bench.py` | Benchmark v1, Phase 1 gate, promotion to `models/current.pt` | 13 |
| `gameai/track.py` | Experiment tracking: every run saved to `runs/` | 16 |
| `gameai/export_onnx.py` | Export for the browser/Chromebook (ONNX, 8-bit) | 10 |

## Quick start (Windows)

1. Double-click **setup.bat** (only once). It installs everything it needs into this folder.
2. Double-click **start.bat**. Only the Game AI window opens (no black window).
   Close the window to quit; if it's training, it finishes first (or type `/quit`).
   If something goes wrong, **start-debug.bat** shows every message, and errors are saved in `logs/app.log`.
3. Type `/train` and watch the progress bar while it trains from scratch.
4. Type a game idea, like `a fast ninja escaping a spooky dungeon full of spiders`.
   The game appears next to the terminal. `/game pop` gives it its own window, `/game` hides or shows it,
   and `/save` keeps it as a file.

**terminal.bat** opens the classic text-only terminal instead (games play inside it with `/play`).

## Chromebook

Your Chromebook can start training, learning and the teacher; they run on your PC.

1. On the PC, type `/chromebook`. It turns sharing on (remembered, same 4-digit code every time) and makes
   `for-chromebook/GameAI.html` with your AI built in. Put that file on your Chromebook (Google Drive or USB).
2. Open GameAI.html on the Chromebook. If your PC is on and running Game AI, it switches to your PC by itself,
   so `/train`, `/learn on`, `/teacher` and `/stop` run there and it's always the newest version.
   If the PC is off, it works offline (making and playing games still works).
3. Optional: `/autostart on` on the PC starts Game AI hidden whenever the PC turns on, so the Chromebook can always reach it.
   If the file can't find your PC (its address changed), type `/pc <address> <code>` with what `/share` shows.

## Game rules (Phase 2)

Describe rules in plain words and the AI writes them into the game, up to two per game, for example
`every 10 seconds 3 more enemies appear` or `after you collect 5 keys the exit opens`.
When: you collect something, you get hit, every 5/10/20 seconds, after 3/5/10 collected, you grab a power-up.
Then: enemies spawn, enemies speed up or slow down, speed boost, heal, teleport, the exit unlocks, enemies freeze.
After updating to Phase 2, train once (`/train`): the AI has a bigger language to learn.

## The teacher (Phase 2)

`/teacher` (or `/teacher 5000`) has a free open AI, Qwen3.5-4B (Apache 2.0; add `--model Qwen/Qwen3.5-9B` for the bigger, slower one),
write natural, varied game requests for your AI to learn from. Every example is self-checked: the teacher
reads its own request back, and it's kept only if it gets exactly the same game. The first run downloads the
teacher (about 3 GB, no account). Examples are saved as they're written, so you can stop it any time.

**Learning while it writes:** `/learn on` runs the teacher at the same time. Every learning round learns from all teacher examples so far, so the AI improves as the
teacher goes. The graph shows how many it has written. The test gains a "real talk" score: how well your AI understands natural wording. Your AI never needs the teacher to run.

## Playtest bot

Every new game is played by a bot in fast-forward before you see it (a few milliseconds per try).
It tries several versions of the level and keeps one it can beat, with hardness closest to what you asked for.
`/quality fast|normal|best` = 1, 6 or 16 versions. If it can't beat any version you get a heads-up.
Results are saved in `memory/playtests.jsonl` for future training. The bot isn't a great player yet (it beats
about half of random games on the first try, and about 4 in 5 when it gets 6 versions to choose from).

## Always-on learning

`/learn on` keeps your AI getting smarter in rounds while you use it. Each round it practises on new
made-up prompts, then takes the same fixed test. A new version replaces the old one only if it scores
higher, so it never gets worse. `/graph` shows how smart it is over time. `/learn off` stops after the current round.

## Terminal commands

| Command | What it does |
|---|---|
| *your idea* | make a game from a description |
| `/edit make it harder` | change the last game, keeping everything else |
| `/play` | play the last game right here in the terminal (Q to come back) |
| `/open` | open the last game in your browser |
| `/save [name]` | save the last game as one HTML file in `games/` |
| `/rate 4 fun but short` | rate the last game (saved for Phase 3 learning) |
| `/train` or `/train base` | train a new AI from scratch, test it, export it |
| `/learn on` / `/learn off` | always-on learning (app) |
| `/teacher [n]` | the teacher writes n natural examples (app, needs the GPU) |
| `/graph` | how smart it is over time (app) |
| `/details` | each part of the last game and how sure it was (app) |
| `/chromebook` | make the one-file Chromebook app (app) |
| `/test` | run the benchmark again |
| `/runs` | every training and test run |
| `/export` | make the browser/Chromebook version |
| `/status`, `/help`, `/clear`, `/quit` | |

Every game you make is logged in `memory/episodes.jsonl` and ratings in
`memory/ratings.jsonl`; these become training data in Phase 3.

## Automatic updates

The app updates itself from the project's GitHub repo (set in `update.json`): when `start.bat` starts,
and every 15 seconds while it runs (if training is running, it waits until it finishes). It downloads only the code,
restarts in a second, and the window reloads, including on a Chromebook connected with `/share`.
Your trained AI, data, games and ratings are never touched. `/update` checks right away.

## Manual commands (optional)

```
python -m gameai                         # the terminal
python -m gameai.all --size small        # data -> train -> test -> export in one go
python -m gameai.datagen / train / bench / generate / export_onnx / track
```

Sizes: `tiny` (0.8M parameters), `small` (4.5M), `base` (12.7M), `large` (35.5M).
Out-of-memory error: `python -m gameai.train --size small --batch 64`.

## The Phase 1 gate

`bench` must show, on the **fresh** row: validity 100% and instruction match ≥ 99%.
A model only replaces `models/current.pt` if it passes the gate **and** beats the
current model on the frozen test. Wrong answers are saved to `bench/last_failures.json`;
reading them shows what to add to the data generator next.

## Optional live dashboard

`pip install trackio`, then set `GAMEAI_TRACKIO=1` before training
(`set GAMEAI_TRACKIO=1` on Windows) and run `trackio show --project gameai`.
Everything stays on your computer.

