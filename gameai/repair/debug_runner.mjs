// The debugger tool's runner (plan: Phase 4): runs a project's JavaScript tests and reports WHY a test failed:
// the error and its stack trace (in the project's own files, with the source lines), structured console errors,
// failing assertions (expected vs actual), and the variable values where it went wrong. Full game state only when
// asked (inspect). The values come from V8's own debugger (node:inspector), which pauses at the exception.
//
// node debug_runner.mjs config.json  ->  one JSON object on stdout
// config: {root, scripts: [files loaded as plain scripts, in order], tests: [test files], inspect, kit: gamekit.js path}
import inspector from 'node:inspector';
import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const ROOT = path.resolve(cfg.root);
const SELF = fileURLToPath(import.meta.url);
const out = { load_errors: [], tests: [], console: [], passed: 0, failed: 0 };

// ---------- the debugger: pause on every exception, keep what the project's frames looked like ----------
const session = new inspector.Session();
session.connect();
function post(method, params = {}) {  // same-thread session: the reply comes back before post() returns
  let res, err, done = false;
  session.post(method, params, (e, r) => { err = e; res = r; done = true; });
  if (!done) throw new Error('inspector reply was not synchronous');
  if (err) throw err;
  return res;
}
const rel = u => {
  if (!u) return null;
  let p = u.startsWith('file:') ? fileURLToPath(u) : u;
  if (!path.isAbsolute(p) || p === SELF) return null;
  p = path.resolve(p);
  return p.startsWith(ROOT + path.sep) ? path.relative(ROOT, p).split(path.sep).join('/') : null;
};
const sourceLines = {};
function lineOf(file, n) {
  try { sourceLines[file] ??= fs.readFileSync(path.join(ROOT, file), 'utf8').split('\n'); return (sourceLines[file][n - 1] ?? '').trim(); }
  catch { return ''; }
}
function show(v) {  // a CDP RemoteObject -> short readable text
  if (!v) return 'undefined';
  if (v.type === 'string') return JSON.stringify(v.value.length > 80 ? v.value.slice(0, 80) + '…' : v.value);
  if (v.type === 'undefined') return 'undefined';
  if ('value' in v && v.type !== 'object') return String(v.value);
  if (v.subtype === 'null') return 'null';
  if (v.type === 'function') return 'function ' + (v.description || '').split('(')[0].replace(/^function\s*/, '').slice(0, 40);
  if (v.preview) {
    const ps = v.preview.properties.slice(0, 8).map(p => (v.subtype === 'array' ? '' : p.name + ': ') + (p.type === 'string' ? JSON.stringify(p.value) : p.value));
    const body = ps.join(', ') + (v.preview.overflow ? ', …' : '');
    return v.subtype === 'array' ? `[${body}] (${v.description})` : `${v.className === 'Object' ? '' : v.className + ' '}{${body}}`;
  }
  return v.description || v.type;
}
let lastPause = null;
const scriptUrl = {};  // callFrame.url is often empty: the script's name comes from scriptParsed
session.on('Debugger.scriptParsed', ({ params }) => { scriptUrl[params.scriptId] = params.url; });
session.on('Debugger.paused', ({ params }) => {
  try {
    if (params.reason !== 'exception' && params.reason !== 'promiseRejection') return;
    const frames = [];
    for (const f of params.callFrames) {
      const file = rel(f.url || scriptUrl[f.location.scriptId]);
      if (!file) continue;
      const fr = { function: f.functionName || (file === currentFile || /test/.test(file) ? `test "${currentTest}"` : '(top level)'), file, line: f.location.lineNumber + 1, col: f.location.columnNumber + 1 };
      fr.source = lineOf(file, fr.line);
      if (frames.length < 2) {  // the variable values of the two innermost project frames
        fr.variables = {};
        for (const sc of f.scopeChain) {
          if (!['local', 'block', 'closure'].includes(sc.type) || !sc.object.objectId) continue;
          const props = post('Runtime.getProperties', { objectId: sc.object.objectId, ownProperties: true, generatePreview: true }).result;
          for (const p of props) {
            if (Object.keys(fr.variables).length >= 15) break;
            if (p.name in fr.variables || p.name === 'this' || !p.value) continue;
            if (sc.type === 'closure' && p.value.type === 'function') continue;
            fr.variables[p.name] = show(p.value);
          }
          if (sc.type === 'closure') break;
        }
        if (f.this && f.this.type === 'object' && f.this.className !== 'global' && f.this.className !== 'Object') {
          const th = post('Runtime.getProperties', { objectId: f.this.objectId, ownProperties: true, generatePreview: true }).result;
          fr.this = Object.fromEntries(th.filter(p => p.value && p.value.type !== 'function').slice(0, 10).map(p => [p.name, show(p.value)]));
        }
      }
      frames.push(fr);
      if (frames.length >= 6) break;
    }
    lastPause = { error: params.data ? (params.data.description || '').split('\n')[0] : '', frames };
  } catch (e) { lastPause = { error: 'could not read the variables: ' + e.message, frames: [] }; }
  finally { session.post('Debugger.resume'); }
});
post('Debugger.enable');
post('Debugger.setPauseOnExceptions', { state: 'all' });

// ---------- what the tests can use ----------
class AssertionError extends Error {
  constructor(msg, actual, expected) { super(msg); this.name = 'AssertionError'; this.actual = actual; this.expected = expected; }
}
const fmt = v => { if (v === undefined) return 'undefined'; if (typeof v === 'number' && !Number.isFinite(v)) return String(v); try { return typeof v === 'string' ? JSON.stringify(v) : JSON.stringify(v, null, 0)?.slice(0, 200) ?? String(v); } catch { return String(v); } };
function expect(actual) {
  const fail = (what, expected) => { throw new AssertionError(`expected ${fmt(actual)} ${what}${expected === undefined ? '' : ' ' + fmt(expected)}`, actual, expected); };
  const deq = (a, b) => fmt(a) === fmt(b);
  const m = {
    toBe: e => Object.is(actual, e) || fail('to be', e),
    toEqual: e => deq(actual, e) || fail('to equal', e),
    toBeTruthy: () => actual || fail('to be truthy'),
    toBeFalsy: () => !actual || fail('to be falsy'),
    toBeGreaterThan: e => actual > e || fail('to be greater than', e),
    toBeLessThan: e => actual < e || fail('to be less than', e),
    toBeCloseTo: (e, d = 2) => Math.abs(actual - e) < 10 ** -d / 2 || fail('to be close to', e),
    toContain: e => (actual && actual.includes && actual.includes(e)) || fail('to contain', e),
    toThrow: () => { try { actual(); } catch { return true; } fail('to throw'); },
  };
  m.not = Object.fromEntries(Object.keys(m).map(k => [k, (...a) => { let threw = false; try { m[k](...a); } catch { threw = true; } if (!threw) throw new AssertionError(`expected ${fmt(actual)} not ${k.replace(/^to/, 'to ')}${a.length ? ' ' + fmt(a[0]) : ''}`, actual, a[0]); return true; }]));
  return m;
}
const registered = [];
globalThis.test = (name, fn) => registered.push({ name, fn, file: currentFile });
globalThis.expect = expect;
globalThis.assert = (cond, msg = 'assertion failed') => { if (!cond) throw new AssertionError(msg, cond, true); };
for (const level of ['error', 'warn']) {
  const orig = console[level];
  console[level] = (...a) => {
    const where = (new Error().stack || '').split('\n').slice(2).map(l => l.match(/\(?((?:file:\/\/)?[^()\s]+):(\d+):\d+\)?$/)).find(m => m && rel(m[1]));
    out.console.push({ level, message: a.map(x => typeof x === 'string' ? x : fmt(x)).join(' ').slice(0, 300),
                       file: where ? rel(where[1]) : null, line: where ? +where[2] : null, test: currentTest });
  };
}
globalThis.window ??= globalThis;

function summarizeGame(g) { try { return gameState(g); } catch (e) { return { error: 'could not read the game: ' + e.message }; } }
function gameState(g) {  // full game state, only when asked
  if (!g || !Array.isArray(g.ents)) return null;
  const kinds = {};
  for (const e of g.ents) if (!e.dead) kinds[e.kind] = (kinds[e.kind] || 0) + 1;
  const h = g.hero || g.ents.find(e => e.kind === 'player');
  return { state: g.state, why: g.why, time: +(g.t || 0).toFixed(2), score: g.score, lives: g.lives, things: kinds,
           player: h ? { x: Math.round(h.x), y: Math.round(h.y), hp: h.hp, holding: h.keys ?? h.items } : null,
           goals: (g.goals || []).map(x => [x.type, x.what, x.n].filter(v => v !== undefined).join(' ')),
           progress: typeof g.progress === 'function' ? g.progress() : undefined };
}

// ---------- load the project and run its tests ----------
let currentFile = null, currentTest = null;
function failure(e) {
  const r = { error: `${e && e.name || 'Error'}: ${e && e.message || e}` };
  if (e instanceof AssertionError) { r.assertion = { actual: fmt(e.actual), expected: fmt(e.expected) }; }
  const p = lastPause && (!e || !e.message || lastPause.error.includes(e.message.split('\n')[0].slice(0, 60))) ? lastPause : null;
  r.stack = p ? p.frames : [];
  if (!r.stack.length && e && e.stack) {  // no pause info: at least the project's lines of the stack trace
    r.stack = e.stack.split('\n').slice(1).map(l => l.match(/at (?:(.+?) \()?((?:file:\/\/)?[^()\s]+):(\d+):(\d+)\)?$/)).filter(m => m && rel(m[2]))
      .slice(0, 6).map(m => ({ function: m[1] || '(top level)', file: rel(m[2]), line: +m[3], col: +m[4], source: lineOf(rel(m[2]), +m[3]) }));
  }
  return r;
}
async function load(file, asModule) {
  currentFile = file; lastPause = null;
  const abs = path.join(ROOT, file);
  try {
    if (asModule) await import(pathToFileURL(abs).href);
    else vm.runInThisContext(fs.readFileSync(abs, 'utf8'), { filename: abs });
  } catch (e) { out.load_errors.push({ file, ...failure(e) }); }
}
const isModule = f => /\.mjs$/.test(f) || /^\s*(import|export)\s/m.test(fs.readFileSync(path.join(ROOT, f), 'utf8'));
if (cfg.kit) vm.runInThisContext(fs.readFileSync(cfg.kit, 'utf8'), { filename: cfg.kit });
for (const f of cfg.scripts || []) await load(f, isModule(f));
for (const f of cfg.tests || []) await load(f, isModule(f));
for (const t of registered) {
  currentTest = t.name; lastPause = null;
  const r = { name: t.name, file: t.file };
  const t0 = Date.now();
  try {
    const v = await t.fn();
    r.passed = true;
    if (cfg.inspect && v) r.game = summarizeGame(v);
  } catch (e) {
    Object.assign(r, { passed: false }, failure(e));
    if (cfg.inspect && globalThis.__game) r.game = summarizeGame(globalThis.__game);
  }
  r.ms = Date.now() - t0;
  out.tests.push(r); r.passed ? out.passed++ : out.failed++;
  process.stderr.write(`ran: ${t.name}\n`);  // so a timeout can say which test was running
}
session.post('Debugger.disable');
process.stdout.write(JSON.stringify(out));
process.exit(0);
