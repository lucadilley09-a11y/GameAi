// The JavaScript side of the code checkers (plan: Phase 4, formatters then static checkers, and a language server).
// One long-running Node process, so each check is fast (no start-up cost): Python sends one JSON request per line on
// stdin and gets one JSON answer per line on stdout.
//
//   {"id":1,"op":"format","files":{"game.js":"..."}}                  -> {"id":1,"files":{"game.js":"..."},"errors":{}}
//   {"id":2,"op":"check","files":{...},"kit":true}                     -> {"id":2,"diagnostics":[...]}
//   {"id":3,"op":"hover"|"definition"|"references"|"completions","files":{...},"file":"game.js","line":3,"col":5}
//
// Tools (installed by gameai/checks/tools.py into runtime/tools/node): TypeScript (type checker and language
// service, the engine behind the TypeScript language server), ESLint (+ typescript-eslint), Prettier, StyLua (Lua,
// as WebAssembly, so no extra program is needed).
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import readline from 'node:readline';

const NM = process.env.GAMEAI_TOOLS_NM || process.argv[2];
const req = createRequire(path.join(NM, 'x.js'));
const ts = req('typescript');
const prettierMod = await import(pathToFileURL(req.resolve('prettier')).href);
const prettier = prettierMod.format ? prettierMod : prettierMod.default;
const { Linter } = req('eslint');
const js = req('@eslint/js');
const tseslint = req('typescript-eslint');
const globals = req('globals');
let stylua = null;
try { stylua = await import(pathToFileURL(path.join(NM, '@johnnymorganz', 'stylua', 'stylua_lib_node.mjs')).href); } catch { }

const KIT_DTS = process.env.GAMEAI_KIT_DTS || '';
const KIT = KIT_DTS ? readFileSync(KIT_DTS, 'utf8') : '';
const TESTING = readFileSync(new URL('./testing.d.ts', import.meta.url), 'utf8');
const TESTING_NAME = '/__testing__.d.ts';
const KIT_NAME = '/__gamekit__.d.ts';
const SCRIPT = /\.(m?js|cjs|jsx|ts|tsx|mts|cts)$/i;
const PRETTIER = /\.(m?js|cjs|jsx|ts|tsx|mts|cts|json|css|html|md)$/i;

// ---------- formatting ----------
async function format(files) {
  const out = {}, errors = {};
  for (const [name, text] of Object.entries(files)) {
    try {
      if (PRETTIER.test(name)) out[name] = await prettier.format(text, { filepath: name, printWidth: 100, singleQuote: true });
      else if (/\.lua$/i.test(name) && stylua) out[name] = stylua.formatCode(text, stylua.Config.new(), null, stylua.OutputVerification.None);
      else out[name] = text;  // not a format these tools handle (Python goes to ruff)
    } catch (e) { errors[name] = String(e.message || e).split('\n').slice(0, 6).join('\n'); out[name] = text; }
  }
  return { files: out, errors };
}

// ---------- the language service (type checker, hover, go to definition, references, completions) ----------
const OPTIONS = {
  target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler,
  lib: ['lib.es2022.d.ts', 'lib.dom.d.ts'], allowJs: true, checkJs: true, strict: true, noEmit: true,
  noFallthroughCasesInSwitch: true, noImplicitReturns: true, skipLibCheck: true, types: [],
};
// plain JavaScript can't write types, so "implicitly any" is not its fault (the same errors still apply to TypeScript)
const JS_ONLY_IGNORED = new Set([7005, 7006, 7008, 7019, 7031, 7034, 7043, 7044, 7045, 7046, 7047, 7048, 7049, 7050]);

function service(files, kit) {
  const names = Object.keys(files).filter(n => SCRIPT.test(n)).map(n => '/' + n.replace(/^\/+/, ''));
  if (kit && KIT) names.push(KIT_NAME);
  if (Object.keys(files).some(n => /(\.test\.|\.spec\.|(^|\/)tests\/)/.test(n))) names.push(TESTING_NAME);
  const text = n => n === KIT_NAME ? KIT : n === TESTING_NAME ? TESTING : files[n.slice(1)];
  const host = {
    getScriptFileNames: () => names,
    getScriptVersion: () => '1',
    getScriptSnapshot: n => {
      const t = names.includes(n) ? text(n) : (ts.sys.fileExists(n) ? ts.sys.readFile(n) : undefined);
      return t === undefined ? undefined : ts.ScriptSnapshot.fromString(t);
    },
    getCurrentDirectory: () => '/',
    getCompilationSettings: () => OPTIONS,
    getDefaultLibFileName: o => ts.getDefaultLibFilePath(o),
    fileExists: n => names.includes(n) || ts.sys.fileExists(n),
    readFile: n => names.includes(n) ? text(n) : ts.sys.readFile(n),
    readDirectory: ts.sys.readDirectory, getDirectories: ts.sys.getDirectories,
    directoryExists: d => names.some(n => n.startsWith(d.replace(/\/?$/, '/'))) || ts.sys.directoryExists(d),  // the project's folders exist only in memory
  };
  return { ls: ts.createLanguageService(host, ts.createDocumentRegistry()), names };
}

function where(sf, start) {
  const p = sf.getLineAndCharacterOfPosition(start || 0);
  return { line: p.line + 1, col: p.character + 1, line_text: sf.text.split('\n')[p.line] ?? '' };
}

function typeDiagnostics(files, kit) {
  const { ls, names } = service(files, kit), out = [];
  for (const n of names) {
    if (n === KIT_NAME || n === TESTING_NAME) continue;
    const isJs = /\.(m?js|cjs|jsx)$/i.test(n);
    const sf = ls.getProgram().getSourceFile(n);
    for (const d of [...ls.getSyntacticDiagnostics(n), ...ls.getSemanticDiagnostics(n)]) {
      if (isJs && JS_ONLY_IGNORED.has(d.code)) continue;
      out.push({ file: n.slice(1), ...where(sf, d.start), severity: d.category === ts.DiagnosticCategory.Error ? 'error' : 'warning',
                 source: 'typescript', code: 'TS' + d.code, message: ts.flattenDiagnosticMessageText(d.messageText, '\n') });
    }
  }
  return out;
}

// ---------- linting ----------
const linter = new Linter({ configType: 'flat' });
const RULES = {
  ...js.configs.recommended.rules,
  'no-undef': 'off',            // the type checker already reports unknown names, with better messages
  'no-unused-vars': ['warn', { args: 'none', varsIgnorePattern: '^(build|_)' }],
  'no-empty': ['error', { allowEmptyCatch: false }],   // an empty catch hides errors instead of fixing them
  'eqeqeq': ['warn', 'smart'],
  'no-eval': 'error', 'no-implied-eval': 'error', 'no-new-func': 'error',
};
const LANG = { ecmaVersion: 2022, sourceType: 'script', globals: { ...globals.browser, GameKit: 'readonly', test: 'readonly', expect: 'readonly', assert: 'readonly' } };
function lint(files) {
  const out = [];
  for (const [name, text] of Object.entries(files)) {
    if (!SCRIPT.test(name)) continue;
    const isTs = /\.(ts|tsx|mts|cts)$/i.test(name);
    const config = isTs
      ? [{ files: ['**/*'], languageOptions: { ...LANG, sourceType: 'module', parser: tseslint.parser },
           plugins: { '@typescript-eslint': tseslint.plugin },
           rules: { ...RULES, 'no-unused-vars': 'off', '@typescript-eslint/no-unused-vars': RULES['no-unused-vars'] } }]
      : [{ files: ['**/*'], languageOptions: { ...LANG, sourceType: /\.mjs$/i.test(name) || /^\s*(import|export)\s/m.test(text) ? 'module' : 'script' }, rules: RULES }];
    config[0].linterOptions = { reportUnusedDisableDirectives: 'error' };  // a disable comment that hides nothing is an error too
    for (const m of linter.verify(text, config, name)) {
      const lines = text.split('\n');
      out.push({ file: name, line: m.line || 1, col: m.column || 1, line_text: lines[(m.line || 1) - 1] ?? '',
                 severity: m.severity === 2 ? 'error' : 'warning', source: 'eslint', code: m.ruleId || 'parse', message: m.message });
    }
  }
  return out;
}

// ---------- language-server style questions ----------
function position(ls, file, line, col) {
  const sf = ls.getProgram().getSourceFile('/' + file);
  if (!sf) throw new Error(`no such script: ${file}`);
  return [sf, sf.getPositionOfLineAndCharacter(Math.max(0, line - 1), Math.max(0, col - 1))];
}
function loc(ls, fileName, span) {
  if (fileName === KIT_NAME) return { file: 'gamekit.d.ts', ...where(ls.getProgram().getSourceFile(fileName), span.start) };
  return { file: fileName.slice(1), ...where(ls.getProgram().getSourceFile(fileName), span.start) };
}
function ask(r) {
  const { ls } = service(r.files, r.kit !== false);
  const [, pos] = position(ls, r.file, r.line, r.col), f = '/' + r.file;
  if (r.op === 'hover') {
    const q = ls.getQuickInfoAtPosition(f, pos);
    return { text: q ? ts.displayPartsToString(q.displayParts) : null, docs: q ? ts.displayPartsToString(q.documentation) : null };
  }
  if (r.op === 'definition') return { locations: (ls.getDefinitionAtPosition(f, pos) || []).map(d => loc(ls, d.fileName, d.textSpan)) };
  if (r.op === 'references') return { locations: (ls.getReferencesAtPosition(f, pos) || []).map(d => loc(ls, d.fileName, d.textSpan)) };
  if (r.op === 'completions') {
    const c = ls.getCompletionsAtPosition(f, pos, {});
    return { names: c ? c.entries.slice(0, r.limit || 50).map(e => e.name) : [] };
  }
  throw new Error('unknown op ' + r.op);
}

async function handle(r) {
  if (r.op === 'ping') return { ok: true, typescript: ts.version, lua: !!stylua };
  if (r.op === 'format') return await format(r.files);
  if (r.op === 'check') return { diagnostics: [...typeDiagnostics(r.files, r.kit !== false), ...lint(r.files)] };
  return ask(r);
}

const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  if (!line.trim()) continue;
  let r = {};
  try { r = JSON.parse(line); process.stdout.write(JSON.stringify({ id: r.id, ...(await handle(r)) }) + '\n'); }
  catch (e) { process.stdout.write(JSON.stringify({ id: r.id, error: String(e && e.message || e) }) + '\n'); }
}
