// GameKit tests: every mechanic, the level generators, determinism, and that each example game can be played and won.
// Run by tests/test_gamekit.py (in V8, like the game page). Returns JSON: [[name, ok, detail], ...]
(function () {
const K = GameKit, out = [];
const t = (name, fn) => { try { const r = fn(); out.push([name, r === true || r === undefined, r === true || r === undefined ? '' : String(r)]); } catch (e) { out.push([name, false, String(e && e.stack || e).slice(0, 300)]); } };
const room = (rows, o = {}) => K.game(Object.assign({ seed: 3, sound: false }, o)).level(rows);
const hold = (g, keys, secs) => { for (const k of keys) g.press(k, true); g.run(secs); for (const k of keys) g.press(k, false); };

t('the player moves when an arrow is held, and stops at a wall', () => {
  const g = room(['#######', '#P....#', '#######']).start(); const x0 = g.hero.x;
  hold(g, ['right'], 0.3); if (!(g.hero.x > x0 + 20)) return 'did not move';
  hold(g, ['right'], 3); return g.hero.x + g.hero.w / 2 <= 6 * 32 + 0.5 || 'went through the wall';
});
t('diagonal movement is not faster than straight', () => {
  const a = room(['#########', '#P......#', '#.......#', '#.......#', '#########']).start(), b = room(['#########', '#P......#', '#.......#', '#.......#', '#########']).start();
  const ax = a.hero.x; hold(a, ['right'], 0.2); const bx = b.hero.x, by = b.hero.y; hold(b, ['right', 'down'], 0.2);
  const da = a.hero.x - ax, db = Math.hypot(b.hero.x - bx, b.hero.y - by); return Math.abs(da - db) < 1 || `${da} vs ${db}`;
});
t('platformer: gravity lands the player on the floor, jumping only works from the ground', () => {
  const g = room(['#######', '#.....#', '#..P..#', '#.....#', '#######'], { mode: 'platformer' }).start();
  g.run(1); const floorY = g.hero.y; if (!g.hero.onGround && g.hero.vy !== 0) return 'not on the ground';
  g.press('up'); g.step(1 / 60); g.press('up', false); g.run(0.15); if (!(g.hero.y < floorY - 20)) return 'did not jump';
  const yMid = g.hero.vy; g.press('up'); g.step(1 / 60); g.press('up', false); return g.hero.vy >= yMid || 'jumped again in the air';
});
t('collecting a coin scores, counts toward the goal, and the last one wins', () => {
  const g = room(['######', '#PCC.#', '######']).goal('collect', 'coin').start(); let got = 0; g.on('collect', () => got++);
  hold(g, ['right'], 1.5); return (got === 2 && g.score === 2 && g.state === 'win') || `collected ${got}, score ${g.score}, state ${g.state}`;
});
t('a locked door blocks without a key and opens with one', () => {
  const a = room(['#######', '#P.D.X#', '#######']).goal('reach', 'exit').start(); hold(a, ['right'], 2);
  if (a.state === 'win') return 'walked through a locked door';
  const b = room(['########', '#PK.D.X#', '########']).goal('reach', 'exit').start(); hold(b, ['right'], 3);
  return b.state === 'win' || `state ${b.state}, keys ${b.inventory.key}`;
});
t('chasing enemies get closer; patrolling ones turn round at walls', () => {
  const g = room(['##########', '#P.......#', '#........#', '#.......E#', '##########']).start(); const e = g.ents.find(x => x.team === 'enemy');
  const d0 = K.dist(e, g.hero); g.run(0.5); if (!(K.dist(e, g.hero) < d0)) return 'chaser did not get closer';
  const h = room(['#######', '#E....#', '#.....#', '#....P#', '#######']).start(); const p = h.ents.find(x => x.team === 'enemy'); p.behaviour = 'patrol';
  let turned = false, dir0 = null; for (let i = 0; i < 300; i++) { h.step(1 / 60); const d = Math.sign(p.vx); if (dir0 === null && d) dir0 = d; if (dir0 && d === -dir0) turned = true; }
  return turned || 'patrol never turned around';
});
t('touching an enemy hurts; running out of lives loses the game', () => {
  const g = room(['#####', '#PE.#', '#####'], { lives: 1 }).start(); g.hero.hp = 1; g.run(1);
  return (g.state === 'lose' && g.why === 'no lives left') || `state ${g.state}`;
});
t('after a hit the player is safe for a moment (no instant multi-hits)', () => {
  const g = room(['#####', '#PE.#', '#####'], { lives: 3 }).start(); g.run(0.5); return g.hero.hp === 2 || `hp ${g.hero.hp}`;
});
t('player bullets defeat enemies; enemies that shoot can hurt the player', () => {
  const g = room(['##########', '#P......E#', '##########']).goal('defeat').start(); g.player({ x: 48, y: 48, canShoot: true });
  g.ents.filter(e => e.team === 'enemy').forEach(e => { e.behaviour = 'still'; }); g.hero.face = [1, 0];
  for (let i = 0; i < 120; i++) { if (i % 20 === 0) g.press('action'); g.step(1 / 60); g.press('action', false); }
  if (g.state !== 'win') return `enemy not defeated (state ${g.state}, defeated ${g.defeated})`;
  const h = room(['##########', '#P......E#', '##########']).start(); const e = h.ents.find(x => x.team === 'enemy'); e.behaviour = 'still'; e.shootEvery = 0.5;
  h.run(3); return h.hero.hp < 3 || 'the shooting enemy never hit';
});
t('platformer: jumping on an enemy defeats it', () => {
  const g = room(['#######', '#.....#', '#.....#', '#..P..#', '#..E..#', '#######'], { mode: 'platformer' }).start();
  const e = g.ents.find(x => x.team === 'enemy'); e.behaviour = 'still'; g.hero.y -= 40; g.run(1);
  return (g.defeated === 1 && g.hero.hp === 3) || `defeated ${g.defeated}, hp ${g.hero.hp}`;
});
t('the timer loses the game; a survive goal wins it', () => {
  const a = room(['#####', '#P..#', '#####']).timer(2).start(); a.run(3); if (a.state !== 'lose') return 'timer did not end the game';
  const b = room(['#####', '#P..#', '#####']).goal('survive', null, 2).start(); b.run(3); return b.state === 'win' || `survive: ${b.state}`;
});
t('rules: every N seconds, and after every Nth event', () => {
  const g = room(['########', '#PCCCC.#', '########']).start(); let ticks = 0, bonus = 0;
  g.rule({ every: 1, then: () => ticks++ }); g.rule({ when: 'collect', n: 2, then: () => bonus++ });
  hold(g, ['right'], 3.05); return (ticks === 3 && bonus === 2) || `every: ${ticks}, every 2nd collect: ${bonus}`;
});
t('custom behaviour functions work (the AI can write its own enemy brain)', () => {
  const g = room(['#######', '#P...E#', '#######']).start(); const e = g.ents.find(x => x.team === 'enemy'); e.behaviour = () => [0, 0]; const x0 = e.x; g.run(1); return e.x === x0 || 'it moved';
});
t('generated levels: rooms, maze and cave are fully connected', () => {
  for (const kind of ['rooms', 'maze', 'cave']) for (let seed = 1; seed <= 5; seed++) {
    const rows = K.generate(kind, 31, 21, K.rng(seed)), open = [];
    rows.forEach((r, y) => [...r].forEach((c, x) => { if (c === '.') open.push([x, y]); }));
    if (open.length < 30) return `${kind} seed ${seed}: only ${open.length} open tiles`;
    const seen = new Set([open[0].join()]), q = [open[0]];
    while (q.length) { const [x, y] = q.pop(); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const k = [x + dx, y + dy].join(); if (rows[y + dy] && rows[y + dy][x + dx] === '.' && !seen.has(k)) { seen.add(k); q.push([x + dx, y + dy]); } } }
    if (seen.size !== open.length) return `${kind} seed ${seed}: ${open.length - seen.size} tiles can't be reached`;
  }
});
t('the same seed gives the same game, a different seed a different one', () => {
  const play = s => { const g = K.game({ seed: s, sound: false }).level('rooms', { w: 30, h: 20 }); g.player(); g.spawn('coin', 8); g.spawn('enemy', 3, { behaviour: 'wander' }); g.start();
    g.run(5, (g, t) => g.press(['left', 'right', 'up', 'down'][Math.floor(t) % 4], true)); return JSON.stringify([g.levelRows, g.ents.map(e => [e.kind, Math.round(e.x), Math.round(e.y)])]); };
  return (play(4) === play(4) && play(4) !== play(5)) || 'not deterministic';
});
t('things never spawn inside walls', () => {
  const g = K.game({ seed: 9, sound: false }).level('cave', { w: 32, h: 22 }); g.player(); g.spawn('coin', 20); g.spawn('enemy', 6);
  const bad = g.ents.filter(e => g.solidAt(e.x, e.y)); return bad.length === 0 || `${bad.length} inside walls`;
});

// ---- the example games: each builds, runs, and can be won by a simple bot ----
function pathStep(g, target) {  // breadth-first search on tiles from the player to target: which way to go
  const T = g.T, sx = Math.floor(g.hero.x / T), sy = Math.floor(g.hero.y / T), tx = Math.floor(target.x / T), ty = Math.floor(target.y / T);
  const prev = {}, q = [[sx, sy]]; prev[sx + ',' + sy] = null;
  while (q.length) { const [x, y] = q.shift(); if (x === tx && y === ty) break;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const nx = x + dx, ny = y + dy, k = nx + ',' + ny; if (!(k in prev) && g.grid[ny] && g.grid[ny][nx] === 0) { prev[k] = [x, y]; q.push([nx, ny]); } } }
  let cur = [tx, ty], k = tx + ',' + ty; if (!(k in prev)) return null;
  while (prev[k] && !(prev[k][0] === sx && prev[k][1] === sy)) { cur = prev[k]; k = cur.join(); }
  return { x: (cur[0] + 0.5) * T, y: (cur[1] + 0.5) * T };
}
function steer(g, to) {
  for (const k of ['left', 'right', 'up', 'down']) g.press(k, false);
  if (!to) return; const dx = to.x - g.hero.x, dy = to.y - g.hero.y;
  if (Math.abs(dx) > 3) g.press(dx > 0 ? 'right' : 'left'); if (Math.abs(dy) > 3) g.press(dy > 0 ? 'down' : 'up');
}
for (const [name, src] of Object.entries(globalThis.EXAMPLES || {})) {
  t(`example ${name}: under 40 lines`, () => src.split('\n').length < 40 || `${src.split('\n').length} lines`);
  t(`example ${name}: builds and runs 20 s without errors`, () => { const g = (0, eval)('(' + src.replace(/^\/\/.*\n/, '').replace(/^function build/, 'function') + ')')(null, 1); g.o.sound = false; g.start(); g.run(20, (g, t) => g.press(['left', 'right', 'up', 'down', 'action'][Math.floor(t * 2) % 5], Math.floor(t * 3) % 2 === 0)); return true; });
  t(`example ${name}: a simple bot can win it`, () => {
    const g = (0, eval)('(' + src.replace(/^\/\/.*\n/, '').replace(/^function build/, 'function') + ')')(null, 1); g.o.sound = false; g.start();
    if (g.o.mode === 'platformer') {  // walk right; jump only when an enemy or a wall is just ahead
      g.run(40, g => { const h = g.hero, T = g.T; g.press('right');
        const foe = g.ents.some(e => e.team === 'enemy' && !e.dead && e.x > h.x && e.x - h.x < T * 2.2 && Math.abs(e.y - h.y) < T);
        g.press('up', foe || g.solidAt(h.x + T * 0.8, h.y)); });
    }
    else {
      g.hero.hp = g.hero.maxHp = 99; g.hero.canShoot = true;
      g.run(240, g => { const want = g.goals.some(q => q.type === 'defeat') ? 'enemy' : g.goals[0].what;
        const targets = g.ents.filter(e => !e.dead && (e.kind === want || (want === 'enemy' && e.team === 'enemy')));
        const near = targets.sort((a, b) => K.dist(a, g.hero) - K.dist(b, g.hero))[0]; if (!near) return;
        if (want === 'enemy' && K.dist(near, g.hero) < g.T * 6) { g.hero.face = K.dirTo(g.hero, near); g.press('action'); }
        steer(g, pathStep(g, near)); });
    }
    return g.state === 'win' || `ended as ${g.state} (${g.why || 'time'}), goals ${g.progress().map(v => v.toFixed(2))}`;
  });
}
return JSON.stringify(out);
})()
