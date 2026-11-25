# GameKit: the game library

A whole 2D game in a few lines. Typed (`gameai/kit/gamekit.d.ts`), tested (`python -m tests.test_gamekit`), no
dependencies, the same result for the same seed. It runs in the browser with a canvas, or headless (no canvas) so
tests and playtest bots can play it.

```js
const g = GameKit.game({ canvas, seed: 1, title: 'Coin Dungeon' });
g.level('rooms', { w: 30, h: 20 });          // or text rows, see below
g.player();
g.spawn('coin', 12);
g.spawn('enemy', 4, { behaviour: 'patrol' });
g.goal('collect', 'coin');                    // win when every coin is collected
g.rule({ every: 20, then: g => g.spawn('enemy', 1, { behaviour: 'chase' }) });
g.start();
```

## Levels

`g.level([...rows])` with one character per tile: `#` wall, `.` floor, `P` player, `E` enemy, `B` boss, `C` coin,
`G` gem, `K` key, `D` locked door (a key opens it), `X` exit, `H` heart (heals). Or a generated level:
`g.level('rooms' | 'maze' | 'cave' | 'platforms' | 'open', { w, h })`. Generated levels are always fully connected.

## Things

| Call | What it does |
| --- | --- |
| `g.player({ speed, hp, canShoot })` | the player (arrows or WASD; space/J shoots when `canShoot`, jumps in platformers) |
| `g.enemy({ behaviour, speed, hp, damage, shootEvery, boss })` | an enemy at a given `x, y`, or a random place away from the player |
| `g.spawn(kind, n, options)` | n things at random open places |
| `g.add(kind, options)` | anything, exactly where you say |

Behaviours: `still`, `chase`, `flee`, `circle`, `wander`, `patrol` (turns round at walls; `axis: 'y'` for up and
down), `zigzag`, or your own function `(self, player, seconds, game) => [dx, dy]`.

## Winning and rules

- `g.goal('collect', 'coin')`, `g.goal('collect', 'gem', 5)`, `g.goal('reach', 'exit')`, `g.goal('survive', null, 60)`,
  `g.goal('defeat')`: the game is won when every goal is met.
- `g.timer(90)`: lose when time runs out (unless a survive goal).
- `g.rule({ every: 10, then: g => ... })`, `g.rule({ when: 'collect', n: 5, then: (g, thing) => ... })`.
- Events for `g.on(...)`: `start`, `collect`, `hit`, `defeat`, `open`, `reach`, `jump`, `respawn`, `win`, `lose`, `tick`.

## Screen and sound

`g.hud(['score', 'lives', 'time', 'goal', g => 'custom text'])`. Title, pause (P), win and game-over screens are
built in; Enter starts and restarts. Sounds play by themselves (`coin`, `jump`, `hit`, `shoot`, `key`, `door`, `win`,
`lose`...); `g.sfx('powerup')` plays one; `sound: false` turns them off.

## Platformers

`mode: 'platformer'`: gravity, jumping from the ground (up or space), jumping on enemies defeats them.

## Testing a game without a screen

```js
const g = build(null, 1).start();
g.run(30, (g, t) => g.press('right'));   // play 30 seconds holding right
g.state;        // 'win' | 'lose' | 'play'
g.log;          // every event with its time, e.g. [2.4, 'collect', 'coin']
g.progress();   // how far along each goal is (0 to 1)
```

Examples: `gameai/kit/examples/` (each under 40 lines, each tested to be winnable by a simple bot). Play them with
`/kit` in the app.
