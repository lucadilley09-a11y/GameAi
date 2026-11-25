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

## The game language (step 4)

Besides the place, player, enemies, goal, difficulty, map size, power-ups and up to 2 rules, a game can have:
- **Enemy behaviour**: patrol back and forth, shoot at you, or wander (default: chase you)
- **Doors**: locked doors you open with keys, or gates you open with switches (every level stays solvable)
- **Levels**: 3 or 5 levels, each a bit harder; **a boss** on the last level (big, shoots in 3 directions)
- **A timer**: about one or two minutes to win
Every game also has a score (and your best score) and little sound effects (`/sound off` to mute).
This is game language version 3: an AI trained before it needs one `/train` to learn the new parts
(a good moment for `/train base`).

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

## The code track: your AI writes code

A second AI of your own, the **coder**, writes real JavaScript: the "brain" that decides how enemies move.
`/train coder` trains it (about 10-20 minutes on your GPU). Then, with a game open:

    /code circle around me but run away when I get close

It writes what it understood (for example `spec: near 100: flee else circle`), then the code. Before anything
runs, each try is safety-checked (only allowed words: no loops, no strings, no functions, nothing outside the
maths) and run in a sandbox (a separate JavaScript engine with time and memory limits) against a reference for
what it said it would do, in 40 random situations. Up to 8 tries; the first that works goes into your game, and
the game is playtested again. If none works it says so and shows its best try.

What it can do so far: chase, flee, circle, keep a distance (100/150/200), zigzag, spiral in, pace left-right or
up-down, stand still, and "X when you're within 100/150/200, otherwise Y". Five of those combinations are never
practised, to test whether it can put pieces together in new ways. `python -m gameai.code.coder bench` shows
how often its code works.

## Bigger AI

`/train base` trains a 12.7 million parameter AI (about 3x the default 4.5 million), `/train large` 35 million.
Each size has its own training settings (more steps, bigger batches, gentler learning rates for bigger ones).
Always-on learning keeps whatever size is in use. The Chromebook file stores the AI compactly
(1 byte per weight plus a scale per row, about 4x smaller, same answers on 397 of 400 test requests);
the PC keeps full precision.

## Playtesters and game quality

Every new game is tested before you see it, by three testers (as in the research plan): an exact check that
every goal can be reached, bots at three skill levels (beginner, average, expert) that play in fast-forward,
and Go-Explore search for levels the bots lost. They run in background workers on all your processor cores.

Quality over speed: by default (`/quality best`) it tries up to 16 versions (`max` = 32) and keeps the one that is
proven beatable, as hard as you asked, and best on the research's quality measures: not over in a few seconds
and not a slog (time to beat it), you actually explore the map (share of the map used), some tension (close
calls or hits), and later levels not easier than the first. Coins, gems and keys are spread across the map.
In a test of 30 random games: 90% proven beatable, average quality 78%, 63% rated great.

## Always-on learning and the self-improving loop

`/learn on` keeps making your AI smarter in rounds, by itself, while you use it. Each round:

1. **Practice test.** It answers about 25,000 requests it has never seen (new practice ones plus the teacher's
   real-person ones). The right answers are known, so every mistake is found.
2. **Drill.** Each mistake becomes a lesson, plus 3 new wordings of exactly the same game, so it practises
   what it gets wrong instead of what it already knows. Your corrections count 5x, games you rated 4-5 count 2x.
3. **Learn.** It trains on fresh practice data plus the teacher's examples plus those lessons.
4. **Safety net.** The new version and the one in use take the same tests: frozen, typos, a hand-checked "gold"
   set of tricky requests (`gameai/gold.jsonl`), the teacher's real-talk set, and requests you corrected
   (every 5th correction is kept aside for testing only). The new version is used only if it scores better
   overall **and** isn't clearly worse on any test. So it can't quietly get worse.
5. **Versions.** Every version that was ever in use is kept (the last 20). `/versions` lists them, `/rollback`
   goes back one, `/rollback 3` goes to version 3.

The graph shows its test score (orange) and its practice-test score on new requests (dashed), and under it
what it drilled last round. Teach it yourself: when it asks "I'm not sure about…", your answer is saved as a
correction; `/fix enemy spiders` corrects the last game; `/rate 4` or `/rate 5` confirms it understood you.

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

## Updates never delete your AI

Updates replace program files only; your AI, its versions, the practice data, learning history, corrections and
saved games are kept. If an update makes the game language bigger, type `/upgrade`: your AI keeps everything it
learned (all its layers and every word it knew; in a test it still got 93.8% of the old settings right before any
catch-up), gets room for the new words, and a short catch-up training (5-10 minutes) teaches it the new parts.
No training from scratch.

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

