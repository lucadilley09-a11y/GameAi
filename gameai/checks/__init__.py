"""Code checkers (plan: Phase 4, session 2): formatters first (changes shown, not silent), then static checkers
(the TypeScript type checker, linters and a language server), then the suppression rules.

    python -m gameai.checks install                 # once: about 65 MB in runtime/tools (works offline afterwards)
    python -m gameai.checks validate game.js ...    # format, check, and pass/fail
    python -m gameai.checks hover game.js 3 12      # what is at line 3, column 12 (language server)

From Python:
    from gameai import checks
    r = checks.validate({"game.js": code})          # r["passed"], r["files"] (formatted), r["diffs"], r["errors"] ...
    checks.report(r)                                # the same as plain text, for the AI's repair prompt and for you

What runs, by language:
    JavaScript / TypeScript   Prettier, then TypeScript (types, also for plain JS, with GameKit's types) and ESLint
    Python                    ruff format, then ruff check
    Lua                       StyLua (format)
    C#                        CSharpier (format; added with the C# sandbox, Phase 4 session 4)
Every language: suppression rules (see suppressions.py).
"""
import difflib, json, os, subprocess, sys, threading

from . import tools
from . import suppressions as _sup

KIT_DTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kit", "gamekit.d.ts")
HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helper.mjs")
NODE_EXT = (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts", ".json", ".css", ".html", ".md", ".lua")
SCRIPT_EXT = (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts")
# ruff findings that are real errors (syntax, unknown names, hidden errors); the rest are warnings
RUFF_ERRORS = ("E9", "F63", "F7", "F82", "E722", "S110", "RUF100")
RUFF_SELECT = "E,F,W,B,S110,E722,RUF100"


class Helper:
    """The long-running Node process (helper.mjs); started on first use, restarted if it dies."""
    def __init__(self):
        self.p, self.n, self.lock = None, 0, threading.Lock()

    def _start(self):
        node, nm = tools.node(), tools.node_modules()
        if not node or not nm: raise RuntimeError("the code checkers aren't installed yet: run  python -m gameai.checks install")
        env = dict(os.environ, GAMEAI_TOOLS_NM=nm, GAMEAI_KIT_DTS=KIT_DTS)
        self.p = subprocess.Popen([node, HELPER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8", env=env, creationflags=tools.NO_WINDOW)

    def ask(self, op, **kw):
        with self.lock:
            for attempt in (1, 2):
                if not self.p or self.p.poll() is not None: self._start()
                self.n += 1
                try:
                    self.p.stdin.write(json.dumps({"id": self.n, "op": op, **kw}) + "\n"); self.p.stdin.flush()
                    line = self.p.stdout.readline()
                    if line: break
                except (BrokenPipeError, OSError):
                    pass
                self.p = None
                if attempt == 2: raise RuntimeError("the code checker stopped unexpectedly")
            r = json.loads(line)
            if r.get("error"): raise RuntimeError(r["error"])
            return r

    def close(self):
        if self.p and self.p.poll() is None:
            self.p.stdin.close(); self.p.wait(timeout=5)
        self.p = None


_helper = Helper()


def _run(cmd, text):
    return subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", creationflags=tools.NO_WINDOW)


# ---------- formatting ----------
def format_files(files):
    """-> (formatted files, {name: diff}, {name: error}). Formatting never changes what code does, only its layout;
    every change is returned as a diff so it is shown, never silent."""
    out, errors = dict(files), {}
    node_files = {k: v for k, v in files.items() if k.lower().endswith(NODE_EXT)}
    if node_files:
        r = _helper.ask("format", files=node_files)
        out.update(r["files"]); errors.update(r["errors"])
    for name, text in files.items():
        low = name.lower()
        if low.endswith(".py") and tools.ruff():
            p = _run([tools.ruff(), "format", "--stdin-filename", name, "-"], text)
            if p.returncode == 0: out[name] = p.stdout
            else: errors[name] = p.stderr.strip()[:500]
        elif low.endswith(".cs") and tools.csharpier():
            p = _run([tools.csharpier(), "format"], text)
            if p.returncode == 0 and p.stdout.strip(): out[name] = p.stdout
            else: errors[name] = (p.stderr or p.stdout).strip()[:500]
    diffs = {n: "".join(difflib.unified_diff(files[n].splitlines(True), out[n].splitlines(True), f"{n} (before)", f"{n} (formatted)", n=1))
             for n in files if out[n] != files[n]}
    return out, diffs, errors


# ---------- static checks ----------
def _ruff_check(name, text):
    p = _run([tools.ruff(), "check", "--output-format", "json", "--select", RUFF_SELECT, "--ignore", "E501",
              "--stdin-filename", name, "-"], text)
    lines, out = text.split("\n"), []
    try: found = json.loads(p.stdout or "[]")
    except ValueError: return [{"file": name, "line": 1, "col": 1, "line_text": "", "severity": "error", "source": "ruff",
                                "code": "ruff", "message": (p.stderr or p.stdout).strip()[:300]}]
    for d in found:
        row = (d.get("location") or {}).get("row") or 1
        code = d.get("code") or "syntax"
        out.append({"file": name, "line": row, "col": (d.get("location") or {}).get("column") or 1, "line_text": lines[row - 1] if row <= len(lines) else "",
                    "severity": "error" if code == "syntax" or code.startswith(RUFF_ERRORS) else "warning",
                    "source": "ruff", "code": code, "message": d.get("message", "")})
    return out


def check(files, kit=True):
    """-> diagnostics: [{file, line, col, line_text, severity ('error'|'warning'), source, code, message}]
    kit=True includes GameKit's types, so wrong GameKit names and arguments are caught too."""
    diags = []
    scripts = {k: v for k, v in files.items() if k.lower().endswith(SCRIPT_EXT)}
    if scripts: diags += _helper.ask("check", files=scripts, kit=kit)["diagnostics"]
    for name, text in files.items():
        if name.lower().endswith(".py") and tools.ruff(): diags += _ruff_check(name, text)
    return sorted(diags, key=lambda d: (d["file"], d["line"], d["col"]))


def validate(files, kit=True):
    """Format, then check the formatted code, then the suppression rules. Passes only with no errors and no
    unjustified suppressions (warnings are reported but don't fail)."""
    formatted, diffs, fmt_errors = format_files(files)
    diags = check(formatted, kit=kit)
    for name, err in fmt_errors.items():  # a formatter that can't read the file means it doesn't parse
        if not any(d["file"] == name and d["severity"] == "error" for d in diags):
            diags.append({"file": name, "line": 1, "col": 1, "line_text": "", "severity": "error", "source": "formatter",
                          "code": "parse", "message": err.splitlines()[0] if err else "couldn't be read"})
    sup = _sup.suppressions(formatted)
    bad = [s for s in sup if not s["justified"]]
    errors = [d for d in diags if d["severity"] == "error"]
    return {"passed": not errors and not bad, "files": formatted, "diffs": diffs,
            "errors": errors, "warnings": [d for d in diags if d["severity"] == "warning"],
            "suppressions": sup, "unjustified": bad}


def report(r, max_items=30):
    """Plain-text version of validate()'s result: each problem with its line, so it can go straight into a repair prompt."""
    out = ["PASSED" if r["passed"] else "FAILED"]
    if r["diffs"]: out.append(f"formatted {len(r['diffs'])} file(s): " + ", ".join(r["diffs"]))
    items = [("error", d) for d in r["errors"]] + [("not justified", s) for s in r["unjustified"]] + [("warning", d) for d in r["warnings"]]
    for kind, d in items[:max_items]:
        out.append(f"{d['file']}:{d['line']}:{d.get('col', 1)} {kind} [{d.get('code') or d.get('kind')}] {d['message']}")
        if d.get("line_text", "").strip(): out.append("    " + d["line_text"].strip()[:160])
    if len(items) > max_items: out.append(f"... and {len(items) - max_items} more")
    return "\n".join(out)


# ---------- language server ----------
def hover(files, file, line, col, kit=True):
    """the type and docs of what's at line/col (1-based)"""
    return _helper.ask("hover", files=files, file=file, line=line, col=col, kit=kit)


def definition(files, file, line, col, kit=True):
    return _helper.ask("definition", files=files, file=file, line=line, col=col, kit=kit)["locations"]


def references(files, file, line, col, kit=True):
    return _helper.ask("references", files=files, file=file, line=line, col=col, kit=kit)["locations"]


def completions(files, file, line, col, kit=True, limit=50):
    return _helper.ask("completions", files=files, file=file, line=line, col=col, kit=kit, limit=limit)["names"]


def status():
    """which checkers are ready"""
    s = {"node": tools.node(), "js_tools": bool(tools.node_modules()), "ruff": tools.ruff(), "csharpier": tools.csharpier()}
    if s["node"] and s["js_tools"]:
        try: s.update(_helper.ask("ping"))
        except Exception as e: s["error"] = str(e)
    return s
