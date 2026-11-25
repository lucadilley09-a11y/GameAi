// GameKit: a small, typed 2D game library so a whole game is a few lines (see docs/gamekit.md, types in gamekit.d.ts).
// Works in the browser (pass a canvas) and headless (no canvas: tests and playtest bots drive it with step()/press()).
// Everything is deterministic for a given seed. No dependencies.
(function (root) {
'use strict';

// ---------- small helpers ----------
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function rng(seed) {  // mulberry32: same seed, same game
  let a = seed >>> 0;
  const r = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  r.int = n => Math.floor(r() * n);
  r.pick = arr => arr[Math.floor(r() * arr.length)];
  return r;
}
const KEYMAP = { ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right', ArrowUp: 'up', KeyW: 'up', ArrowDown: 'down', KeyS: 'down',
  Space: 'action', KeyJ: 'action', Enter: 'start', KeyP: 'pause', Escape: 'pause' };
const LEGEND = { '#': 'wall', '.': 'floor', ' ': 'floor', P: 'player', E: 'enemy', C: 'coin', G: 'gem', X: 'exit', K: 'key', D: 'door', H: 'heart', B: 'boss', '=': 'wall' };
const COLORS = { player: '#4fc1ff', enemy: '#f14c4c', boss: '#c586c0', coin: '#e5c07b', gem: '#4ec9b0', exit: '#89d185', key: '#dcdcaa', door: '#b5835a',
  heart: '#f48771', bullet: '#ffffff', wall: '#3c3c3c', floor: '#1e1e1e', platform: '#5a5a5a' };

// ---------- level building ----------
function parseLevel(rows) {
  const grid = [], spawns = [];
  rows.forEach((line, y) => {
    const row = [];
    for (let x = 0; x < line.length; x++) {
      const k = LEGEND[line[x]] || 'floor';
      row.push(k === 'wall' ? 1 : 0);
      if (k !== 'wall' && k !== 'floor') spawns.push({ kind: k, x, y });
    }
    grid.push(row);
  });
  const w = Math.max(...grid.map(r => r.length));
  grid.forEach(r => { while (r.length < w) r.push(1); });
  return { grid, spawns };
}

function generate(kind, w, h, r) {
  // -> rows of text, so generated levels go through the same path as typed ones
  const g = Array.from({ length: h }, () => Array(w).fill('#'));
  const open = (x, y) => { if (x > 0 && y > 0 && x < w - 1 && y < h - 1) g[y][x] = '.'; };
  if (kind === 'rooms') {
    const rooms = [];
    for (let i = 0; i < 40 && rooms.length < 7; i++) {
      const rw = 4 + r.int(6), rh = 3 + r.int(4), rx = 1 + r.int(w - rw - 2), ry = 1 + r.int(h - rh - 2);
      if (rooms.some(o => rx < o.x + o.w + 1 && rx + rw + 1 > o.x && ry < o.y + o.h + 1 && ry + rh + 1 > o.y)) continue;
      rooms.push({ x: rx, y: ry, w: rw, h: rh });
      for (let y = ry; y < ry + rh; y++) for (let x = rx; x < rx + rw; x++) open(x, y);
    }
    for (let i = 1; i < rooms.length; i++) {  // corridors join every room to the one before: always connected
      const a = rooms[i - 1], b = rooms[i];
      let x = a.x + (a.w >> 1), y = a.y + (a.h >> 1); const tx = b.x + (b.w >> 1), ty = b.y + (b.h >> 1);
      while (x !== tx) { open(x, y); x += Math.sign(tx - x); }
      while (y !== ty) { open(x, y); y += Math.sign(ty - y); }
    }
  } else if (kind === 'maze') {
    const st = [[1, 1]]; open(1, 1);
    while (st.length) {
      const [x, y] = st[st.length - 1];
      const n = [[2, 0], [-2, 0], [0, 2], [0, -2]].map(([dx, dy]) => [x + dx, y + dy, dx, dy]).filter(([nx, ny]) => nx > 0 && ny > 0 && nx < w - 1 && ny < h - 1 && g[ny][nx] === '#');
      if (!n.length) { st.pop(); continue; }
      const [nx, ny, dx, dy] = r.pick(n); open(x + dx / 2, y + dy / 2); open(nx, ny); st.push([nx, ny]);
    }
  } else if (kind === 'cave') {
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) if (r() > 0.42) g[y][x] = '.';
    for (let k = 0; k < 4; k++) {  // smooth into caves
      const c = g.map(row => row.slice());
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        let walls = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) walls += g[y + dy][x + dx] === '#';
        c[y][x] = walls > 4 ? '#' : '.';
      }
      for (let y = 0; y < h; y++) g[y] = c[y];
    }
    keepLargest(g, w, h);
  } else if (kind === 'platforms') {
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) g[y][x] = '.';
    for (let y = h - 4; y > 2; y -= 3) {  // ledges every 3 rows, each reachable from the one below
      let x = 1 + r.int(4);
      while (x < w - 4) { const len = 3 + r.int(5); for (let i = 0; i < len && x + i < w - 1; i++) g[y][x + i] = '#'; x += len + 2 + r.int(3); }
    }
  } else {
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) g[y][x] = '.';
  }
  return g.map(row => row.join(''));
}

function keepLargest(g, w, h) {
  const seen = g.map(r => r.map(() => 0)); let best = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (g[y][x] !== '.' || seen[y][x]) continue;
    const q = [[x, y]], area = []; seen[y][x] = 1;
    while (q.length) { const [cx, cy] = q.pop(); area.push([cx, cy]);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const nx = cx + dx, ny = cy + dy;
        if (g[ny] && g[ny][nx] === '.' && !seen[ny][nx]) { seen[ny][nx] = 1; q.push([nx, ny]); } } }
    if (area.length > best.length) best = area;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (g[y][x] === '.') g[y][x] = '#';
  for (const [x, y] of best) g[y][x] = '.';
}

// ---------- behaviours: how non-player things move. Each returns a direction [dx, dy] (top-down) ----------
const BEHAVIOURS = {
  still: () => [0, 0],
  chase: (e, p) => dirTo(e, p),
  flee: (e, p) => { const d = dirTo(e, p); return [-d[0], -d[1]]; },
  circle: (e, p) => { const d = dirTo(e, p); return [-d[1], d[0]]; },
  wander: (e, p, t, g) => { if (!e.mem.wt || t > e.mem.wt) { const a = g.rand() * Math.PI * 2; e.mem.wd = [Math.cos(a), Math.sin(a)]; e.mem.wt = t + 0.6 + g.rand(); } return e.mem.wd; },
  patrol: e => { if (!e.mem.pd) e.mem.pd = [e.axis === 'y' ? 0 : 1, e.axis === 'y' ? 1 : 0]; if (e.bumped) e.mem.pd = [-e.mem.pd[0], -e.mem.pd[1]]; return e.mem.pd; },
  zigzag: (e, p, t) => { const d = dirTo(e, p), s = Math.sin(t * 6) * 0.8; return [d[0] - d[1] * s, d[1] + d[0] * s]; },
};
function dirTo(a, b) { const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1; return [dx / d, dy / d]; }
function dist(a, b) { return Math.hypot(b.x - a.x, b.y - a.y); }

// ---------- sound: short synthesized effects (Web Audio), silent when headless ----------
const SFX = { coin: [880, 1320, 0.08, 'square'], gem: [660, 1760, 0.12, 'triangle'], jump: [300, 600, 0.12, 'square'], hit: [220, 80, 0.18, 'sawtooth'],
  shoot: [900, 300, 0.07, 'square'], key: [520, 1040, 0.15, 'triangle'], door: [180, 260, 0.2, 'sine'], heart: [600, 900, 0.15, 'sine'],
  win: [523, 1046, 0.5, 'triangle'], lose: [300, 100, 0.6, 'sawtooth'], powerup: [400, 1200, 0.25, 'square'] };
let actx = null;
function beep(name, on) {
  if (!on || typeof window === 'undefined' || !(window.AudioContext || window.webkitAudioContext)) return;
  const s = SFX[name]; if (!s) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator(), v = actx.createGain(), t = actx.currentTime;
    o.type = s[3]; o.frequency.setValueAtTime(s[0], t); o.frequency.exponentialRampToValueAtTime(Math.max(30, s[1]), t + s[2]);
    v.gain.setValueAtTime(0.12, t); v.gain.exponentialRampToValueAtTime(0.001, t + s[2]);
    o.connect(v); v.connect(actx.destination); o.start(t); o.stop(t + s[2] + 0.02);
  } catch (e) { /* sound is optional */ }
}

// ---------- pixel sprites: a small symmetric 8x8 picture per kind, the same every time for a seed ----------
function sprite(kind, color, seed) {
  const r = rng(seed * 31 + [...kind].reduce((a, c) => a * 33 + c.charCodeAt(0), 7));
  const px = [];
  for (let y = 0; y < 8; y++) { const row = []; for (let x = 0; x < 4; x++) row.push(r() > (x === 0 ? 0.7 : 0.45) ? 1 : 0); px.push([...row, ...row.slice().reverse()]); }
  px[3][3] = px[3][4] = px[4][3] = px[4][4] = 1;  // never empty in the middle
  return { px, color };
}

// ---------- the game ----------
class Game {
  constructor(o = {}) {
    this.o = Object.assign({ width: 640, height: 480, tile: 32, seed: 1, mode: 'topdown', gravity: 1500, jump: 560, sound: true,
      background: COLORS.floor, title: 'Game', lives: 3, canvas: null, pixel: true }, o);
    this.rand = rng(this.o.seed); this.t = 0; this.score = 0; this.lives = this.o.lives; this.state = 'title';
    this.ents = []; this.grid = [[0]]; this.keys = {}; this.pressed = {}; this.handlers = {}; this.rules = []; this.goals = [];
    this.counts = {}; this.inventory = {}; this.defeated = 0; this.cam = { x: 0, y: 0 }; this.nextId = 1; this.hudItems = ['score', 'lives', 'goal']; this.log = [];
    this.ctx = o.canvas && o.canvas.getContext ? o.canvas.getContext('2d') : null;
    if (this.ctx && typeof window !== 'undefined') this._listen();
  }
  get T() { return this.o.tile; }

  // world
  level(src, opts = {}) {
    const rows = Array.isArray(src) ? src : typeof src === 'string' && src.includes('\n') ? src.split('\n').filter(l => l.length)
      : generate(src, opts.w || 25, opts.h || 17, this.rand);
    const { grid, spawns } = parseLevel(rows);
    this.grid = grid; this.levelRows = rows;
    if (!this.o.keepSize) { this.o.width = Math.min(this.o.width, grid[0].length * this.T); this.o.height = Math.min(this.o.height, grid.length * this.T); }  // no empty space around small levels
    for (const s of spawns) {
      const at = { x: (s.x + 0.5) * this.T, y: (s.y + 0.5) * this.T };
      if (s.kind === 'player') this.player(at); else if (s.kind === 'enemy') this.enemy(at); else if (s.kind === 'boss') this.enemy({ ...at, boss: true });
      else this.add(s.kind, at);
    }
    return this;
  }
  free(n = 1, away = null, minDist = 0) {
    // -> n random open tile centres (not on other things; optionally at least minDist from `away`)
    const out = [], taken = new Set(this.ents.map(e => `${Math.floor(e.x / this.T)},${Math.floor(e.y / this.T)}`));
    for (let tries = 0; out.length < n && tries < 5000; tries++) {
      const y = this.rand.int(this.grid.length), x = this.rand.int(this.grid[0].length), k = `${x},${y}`;
      if (this.grid[y][x] || taken.has(k)) continue;
      const p = { x: (x + 0.5) * this.T, y: (y + 0.5) * this.T };
      if (away && dist(p, away) < minDist) continue;
      if (this.o.mode === 'platformer' && !(this.grid[y + 1] && this.grid[y + 1][x])) continue;  // standing on something
      taken.add(k); out.push(p);
    }
    return out;
  }
  add(kind, o = {}) {
    const T = this.T;
    const e = Object.assign({ id: this.nextId++, kind, x: T * 1.5, y: T * 1.5, w: T * 0.7, h: T * 0.7, vx: 0, vy: 0, speed: 0, hp: 1, solid: false,
      color: COLORS[kind] || '#cccccc', mem: {}, tags: [], dead: false, team: 'none' }, o);
    if (!e.sprite) e.sprite = sprite(kind, e.color, this.o.seed);
    this.ents.push(e); this.counts[kind] = (this.counts[kind] || 0) + 1;
    return e;
  }
  player(o = {}) {
    if (this.hero) this.hero.dead = true;
    const at = (o.x != null) ? {} : (this.free(1)[0] || {});
    this.hero = this.add('player', Object.assign({ speed: 170, hp: 3, maxHp: 3, team: 'player', solid: true }, at, o));
    return this.hero;
  }
  enemy(o = {}) {
    const at = (o.x != null) ? {} : (this.free(1, this.hero, this.T * 5)[0] || this.free(1)[0] || {});
    const boss = !!o.boss;
    return this.add(boss ? 'boss' : 'enemy', Object.assign({ speed: boss ? 80 : 90, hp: boss ? 5 : 1, damage: 1, behaviour: 'chase', team: 'enemy',
      solid: true, w: this.T * (boss ? 1.4 : 0.7), h: this.T * (boss ? 1.4 : 0.7), shootEvery: 0 }, at, o));
  }
  spawn(kind, n = 1, o = {}) {
    const out = [];
    for (const at of this.free(n, this.hero, kind === 'enemy' ? this.T * 5 : 0)) out.push(kind === 'enemy' || kind === 'boss' ? this.enemy({ ...o, ...at, boss: kind === 'boss' }) : this.add(kind, { ...o, ...at }));
    return out;
  }

  // logic
  on(ev, fn) { (this.handlers[ev] = this.handlers[ev] || []).push(fn); return this; }
  emit(ev, ...a) {
    if (ev !== 'tick') this.log.push([+this.t.toFixed(2), ev, a[0] && a[0].kind || a[0]]);  // what happened, for tests and playtests
    if (this.log.length > 5000) this.log.splice(0, 1000);
    for (const f of this.handlers[ev] || []) f(...a);
  }
  goal(type, what, n) {
    // 'collect' a kind (all of them, or n) | 'reach' the exit | 'survive' n seconds | 'defeat' all enemies (or n)
    this.goals.push({ type, what, n }); return this;
  }
  rule(o) {
    // { every: seconds, then: game => ... } or { when: event name, n: count, then: (game, thing) => ... }
    const r = Object.assign({ seen: 0, last: 0 }, o);
    this.rules.push(r);
    if (r.when) this.on(r.when, x => { r.seen++; if (!r.n || r.seen % r.n === 0) r.then(this, x); });
    return this;
  }
  hud(items) { this.hudItems = items; return this; }
  sfx(name) { beep(name, this.o.sound); }
  timer(seconds) { this.timeLimit = seconds; return this; }
  restart() { if (typeof location !== 'undefined') location.reload(); }

  // control
  start() { this.state = 'play'; this.emit('start'); if (this.ctx && !this._raf && typeof requestAnimationFrame !== 'undefined') this._loop(); return this; }
  press(name, down = true) { if (down && !this.keys[name]) this.pressed[name] = true; this.keys[name] = down; }
  run(seconds, input) {
    // headless: advance the game; input(game, t) may press keys. -> the game's state at the end
    const dt = 1 / 60, end = this.t + seconds;
    while (this.t < end && this.state === 'play') { if (input) input(this, this.t); this.step(dt); }
    return this.state;
  }
  _listen() {
    const set = (ev, down) => { const k = KEYMAP[ev.code]; if (!k) return; ev.preventDefault();
      if (down && k === 'start' && this.state !== 'play') { this.state === 'title' ? this.start() : this.restart && this.restart(); }
      if (down && k === 'pause') this.state = this.state === 'paused' ? 'play' : this.state === 'play' ? 'paused' : this.state;
      this.press(k, down); };
    addEventListener('keydown', e => set(e, true)); addEventListener('keyup', e => set(e, false));
  }
  _loop() {
    let last = performance.now(), acc = 0;
    const tick = now => { acc += Math.min(0.25, (now - last) / 1000); last = now;
      while (acc >= 1 / 60) { if (this.state === 'play') this.step(1 / 60); acc -= 1 / 60; }
      this.draw(); this._raf = requestAnimationFrame(tick); };
    this._raf = requestAnimationFrame(tick);
  }

  // one fixed step of 1/60 s
  step(dt) {
    if (this.state !== 'play') return;
    this.t += dt;
    const p = this.hero, plat = this.o.mode === 'platformer';
    if (p && !p.dead) {
      const ix = (this.keys.right ? 1 : 0) - (this.keys.left ? 1 : 0), iy = (this.keys.down ? 1 : 0) - (this.keys.up ? 1 : 0);
      if (plat) {
        p.vx = ix * p.speed;
        if ((this.pressed.up || this.pressed.action) && p.onGround) { p.vy = -this.o.jump; this.sfx('jump'); this.emit('jump', p); }
      } else {
        const l = Math.hypot(ix, iy) || 1; p.vx = ix / l * p.speed; p.vy = iy / l * p.speed;
        if (this.keys.action && p.canShoot) this.shoot(p, p.face || [1, 0]);  // hold to keep firing (at most every fireDelay)
        if (ix || iy) p.face = [ix / l, iy / l];
      }
    }
    for (const e of this.ents) {
      if (e.dead || e === p) continue;
      if (e.team === 'enemy' && e.kind !== 'bullet') {
        const fn = typeof e.behaviour === 'function' ? e.behaviour : BEHAVIOURS[e.behaviour] || BEHAVIOURS.chase;
        let d = p ? fn(e, p, this.t, this) : [0, 0];
        const l = Math.hypot(d[0], d[1]); if (l > 1) d = [d[0] / l, d[1] / l];
        e.vx = d[0] * e.speed; if (!plat) e.vy = d[1] * e.speed;
        if (e.shootEvery && p && this.t - (e.mem.lastShot || 0) > e.shootEvery && dist(e, p) < this.T * 9) { e.mem.lastShot = this.t; this.shoot(e, dirTo(e, p)); }
      }
      if (e.life != null && (e.life -= dt) <= 0) e.dead = true;
    }
    for (const e of this.ents) if (!e.dead && (e.vx || e.vy || plat)) this._move(e, dt, plat && e.kind !== 'bullet' && e.fly !== true);
    this._touches();
    for (const r of this.rules) if (r.every && this.t - r.last >= r.every) { r.last = this.t; r.then(this); }
    if (this.timeLimit && this.t >= this.timeLimit && !this.goals.some(g => g.type === 'survive')) this.lose('time ran out');
    this._checkGoals();
    this.ents = this.ents.filter(e => !e.dead || e === this.hero);
    this.pressed = {};
    this.emit('tick', dt);
  }
  _move(e, dt, gravity) {
    if (gravity) e.vy = Math.min(e.vy + this.o.gravity * dt, 900);
    e.bumped = false; e.onGround = false;
    for (const axis of ['x', 'y']) {
      const v = axis === 'x' ? e.vx : e.vy; if (!v) continue;
      e[axis] += v * dt;
      if (e.kind === 'bullet') { if (this.solidAt(e.x, e.y)) e.dead = true; continue; }
      const hit = this._overlapWall(e);
      if (hit) {
        const T = this.T, half = axis === 'x' ? e.w / 2 : e.h / 2;
        e[axis] = v > 0 ? hit[axis] * T - half - 0.01 : (hit[axis] + 1) * T + half + 0.01;
        if (axis === 'y') { if (v > 0) e.onGround = true; e.vy = 0; } else e.vx = 0;
        e.bumped = true;
      }
    }
  }
  solidAt(x, y) { const tx = Math.floor(x / this.T), ty = Math.floor(y / this.T); return !this.grid[ty] || this.grid[ty][tx] !== 0; }
  _overlapWall(e) {
    const T = this.T, x0 = Math.floor((e.x - e.w / 2) / T), x1 = Math.floor((e.x + e.w / 2) / T), y0 = Math.floor((e.y - e.h / 2) / T), y1 = Math.floor((e.y + e.h / 2) / T);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (!this.grid[y] || this.grid[y][x] !== 0) return { x, y };
    for (const d of this.ents) if (d.kind === 'door' && !d.dead && !d.open && e !== d && overlap(e, d)) {
      if (e === this.hero && (this.inventory.key || 0) > 0) {  // walking into a locked door with a key opens it
        this.inventory.key--; d.open = true; d.dead = true; this.sfx('door'); this.emit('open', d); continue;
      }
      return { x: Math.floor(d.x / T), y: Math.floor(d.y / T) };
    }
    return null;
  }
  _touches() {
    const p = this.hero; if (!p || p.dead) return;
    for (const e of this.ents) {
      if (e === p || e.dead) continue;
      if (e.kind === 'bullet') {
        const target = e.team === 'player' ? this.ents.find(x => x.team === 'enemy' && !x.dead && overlap(x, e)) : overlap(p, e) ? p : null;
        if (target) { e.dead = true; this.damage(target, e.damage || 1, e); }
        continue;
      }
      if (!overlap(p, e)) continue;
      if (e.team === 'enemy') {
        if (this.o.mode === 'platformer' && p.vy > 0 && p.y < e.y - e.h / 4) { p.vy = -this.o.jump * 0.6; this.damage(e, 1, p); }  // stomp
        else this.damage(p, e.damage || 1, e);
      } else if (['coin', 'gem', 'key', 'heart', 'item'].includes(e.kind) || e.pickup) {
        e.dead = true; this.inventory[e.kind] = (this.inventory[e.kind] || 0) + 1;
        if (e.kind === 'coin' || e.kind === 'gem') this.score += e.value || (e.kind === 'gem' ? 5 : 1);
        if (e.kind === 'heart') p.hp = Math.min(p.maxHp || 3, p.hp + 1);
        this.sfx(e.kind === 'gem' ? 'gem' : e.kind === 'key' ? 'key' : e.kind === 'heart' ? 'heart' : 'coin');
        this.emit('collect', e);
      } else if (e.kind === 'exit') {
        this.emit('reach', e);
      }
    }
  }
  shoot(from, dir, o = {}) {
    if (this.t - (from.mem.lastFire || -1) < (from.fireDelay || 0.25)) return null;
    from.mem.lastFire = this.t; this.sfx('shoot');
    return this.add('bullet', Object.assign({ x: from.x, y: from.y, w: 6, h: 6, vx: dir[0] * 360, vy: dir[1] * 360, team: from.team, life: 1.5, damage: 1, fly: true }, o));
  }
  damage(e, n, by) {
    if (e.mem.hurtUntil > this.t) return;
    e.hp -= n; e.mem.hurtUntil = this.t + (e === this.hero ? 1 : 0.1);
    this.emit('hit', e, by);
    if (e === this.hero) { this.sfx('hit'); if (e.hp <= 0) { this.lives--; if (this.lives <= 0) this.lose('no lives left'); else { e.hp = e.maxHp || 3; this.emit('respawn', e); } } }
    else if (e.hp <= 0) { e.dead = true; this.defeated = (this.defeated || 0) + 1; this.score += e.kind === 'boss' ? 10 : 1; this.emit('defeat', e); }
  }
  win(why = 'goal reached') { if (this.state !== 'play') return; this.state = 'win'; this.why = why; this.sfx('win'); this.emit('win', why); }
  lose(why = 'lost') { if (this.state !== 'play') return; this.state = 'lose'; this.why = why; this.sfx('lose'); this.emit('lose', why); }
  left(kind) { return this.ents.filter(e => !e.dead && (e.kind === kind || (kind === 'enemy' && e.team === 'enemy'))).length; }
  progress() {
    // -> for each goal, how far along it is (0..1)
    return this.goals.map(g => {
      if (g.type === 'collect') { const total = g.n || this.counts[g.what] || 1; return Math.min(1, (this.inventory[g.what] || 0) / total); }
      if (g.type === 'defeat') { const total = g.n || ((this.counts.enemy || 0) + (this.counts.boss || 0)) || 1; return Math.min(1, this.defeated / total); }
      if (g.type === 'survive') return Math.min(1, this.t / g.n);
      if (g.type === 'reach') return this.log.some(l => l[1] === 'reach') ? 1 : 0;
      return 0;
    });
  }
  _checkGoals() { if (this.goals.length && this.progress().every(v => v >= 1)) this.win(); }

  // drawing (browser only)
  draw() {
    const c = this.ctx; if (!c) return;
    const W = this.o.width, H = this.o.height, T = this.T, p = this.hero;
    if (p) { this.cam.x = clamp(p.x - W / 2, 0, Math.max(0, this.grid[0].length * T - W)); this.cam.y = clamp(p.y - H / 2, 0, Math.max(0, this.grid.length * T - H)); }
    c.imageSmoothingEnabled = false; c.fillStyle = this.o.background; c.fillRect(0, 0, W, H);
    c.save(); c.translate(-Math.round(this.cam.x), -Math.round(this.cam.y));
    const x0 = Math.floor(this.cam.x / T), y0 = Math.floor(this.cam.y / T);
    for (let y = y0; y < y0 + H / T + 2; y++) for (let x = x0; x < x0 + W / T + 2; x++) if (this.grid[y] && this.grid[y][x]) { c.fillStyle = COLORS.wall; c.fillRect(x * T, y * T, T, T); c.fillStyle = '#4a4a4a'; c.fillRect(x * T, y * T, T, 3); }
    for (const e of this.ents) {
      if (e.dead) continue;
      const blink = e.mem.hurtUntil > this.t && Math.floor(this.t * 20) % 2;
      if (blink) continue;
      if (this.o.pixel && e.sprite && e.kind !== 'bullet') {
        const s = e.w / 8; c.fillStyle = e.color;
        e.sprite.px.forEach((row, yy) => row.forEach((on, xx) => { if (on) c.fillRect(e.x - e.w / 2 + xx * s, e.y - e.h / 2 + yy * s, s + 0.5, s + 0.5); }));
      } else { c.fillStyle = e.color; c.fillRect(e.x - e.w / 2, e.y - e.h / 2, e.w, e.h); }
    }
    c.restore();
    this._hud(c, W, H);
  }
  _hud(c, W, H) {
    c.font = '16px monospace'; c.textBaseline = 'top'; c.fillStyle = '#ffffff';
    const parts = [];
    for (const k of this.hudItems) {
      if (k === 'score') parts.push(`Score ${this.score}`);
      else if (k === 'lives') parts.push(`Lives ${this.lives}` + (this.hero ? `  HP ${Math.max(0, this.hero.hp)}` : ''));
      else if (k === 'time') parts.push(`Time ${this.timeLimit ? Math.max(0, Math.ceil(this.timeLimit - this.t)) : Math.floor(this.t)}`);
      else if (k === 'goal' && this.goals.length) parts.push(`Goal ${Math.floor(this.progress().reduce((a, b) => a + b, 0) / this.goals.length * 100)}%`);
      else if (typeof k === 'function') parts.push(String(k(this)));
    }
    c.fillText(parts.join('   '), 10, 8);
    if (this.state !== 'play') {
      c.fillStyle = 'rgba(0,0,0,.6)'; c.fillRect(0, 0, W, H); c.fillStyle = '#ffffff'; c.textAlign = 'center'; c.font = '28px monospace';
      const big = { title: this.o.title, win: 'You win!', lose: 'Game over', paused: 'Paused' }[this.state];
      c.fillText(big, W / 2, H / 2 - 30); c.font = '16px monospace';
      c.fillText(this.state === 'title' ? 'Press Enter to start' : this.state === 'paused' ? 'Press P to carry on' : (this.why || '') + ' - press Enter to play again', W / 2, H / 2 + 10);
      c.textAlign = 'left';
    }
  }
}
function overlap(a, b) { return Math.abs(a.x - b.x) * 2 < a.w + b.w && Math.abs(a.y - b.y) * 2 < a.h + b.h; }

function game(o) { return new Game(o); }
const GameKit = { game, Game, rng, generate, parseLevel, BEHAVIOURS, SFX, dirTo, dist, overlap, version: '1.0.0' };
if (typeof module !== 'undefined' && module.exports) module.exports = GameKit; else root.GameKit = GameKit;
})(typeof globalThis !== 'undefined' ? globalThis : this);
