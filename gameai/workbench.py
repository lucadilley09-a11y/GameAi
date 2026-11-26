"""Local developer tools for Game AI.

Safe, workspace-relative utilities: compact files, search source, map symbols,
filter command output, save/search project notes, run approved commands, and batch
multiple read-only or approved operations.

Run: python -m gameai.workbench --help
"""
from __future__ import annotations
import argparse, ast, fnmatch, json, os, re, shlex, subprocess, sys, time, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAULT_DIR = Path("memory") / "vault"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "models", "checkpoints", "runs", "bench", "data", "datasets", "training_data", "export", "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".gguf", ".npy", ".npz", ".parquet", ".arrow", ".sqlite", ".db", ".pyc"}

def set_root(root: str | Path) -> None:
    global ROOT
    ROOT = Path(root).expanduser().resolve(strict=True)
    if not ROOT.is_dir():
        raise ValueError("workspace root must be a directory")
MAX_FILE = 8 * 1024 * 1024
MAX_OUTPUT = 12000
ERROR_RE = re.compile(r"(?:^|\b)(?:error|exception|traceback|failed|failure|fatal|assertionerror|syntaxerror|typeerror|valueerror)(?:\b|:)", re.I)
ALLOWED = {
    "tests": None,  # Game AI's own sandboxed game-test runner; args: [project-folder]
    "compile": [sys.executable, "-m", "compileall", "-q", "gameai"],
    "git-status": ["git", "status", "--short"],
    "git-diff": ["git", "diff", "--"],
    "list": None,
}


def _path(value: str) -> Path:
    p = (ROOT / value).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError("path must stay inside the project folder")
    if p.exists() and p.is_dir():
        return p
    return p


def _read(p: Path) -> str:
    if not p.is_file(): raise FileNotFoundError(str(p))
    if p.stat().st_size > MAX_FILE: raise ValueError(f"file exceeds {MAX_FILE} byte safety limit")
    return p.read_text(encoding="utf-8", errors="replace")


def shrink_file(file: str, max_chars: int = 12000, mode: str = "auto") -> dict:
    """Return a compact, useful view without modifying the original file."""
    p = _path(file); raw = _read(p); lines = raw.splitlines()
    if mode not in ("auto", "errors", "tail"): raise ValueError("mode must be auto, errors, or tail")
    if mode == "errors" or (mode == "auto" and any(ERROR_RE.search(x) for x in lines)):
        hits = [i for i, line in enumerate(lines) if ERROR_RE.search(line)]
        chosen = set()
        for i in hits:
            chosen.update(range(max(0, i-2), min(len(lines), i+3)))
        indexes = sorted(chosen)
        compact = []
        prev = -2
        for i in indexes:
            if i > prev + 1: compact.append("... omitted ...")
            compact.append(f"{i+1}: {lines[i]}")
            prev = i
        result = "\n".join(compact) if compact else "(no error-like lines found)"
    elif mode == "tail":
        result = "\n".join(f"{i+1}: {x}" for i, x in enumerate(lines[-max(1, max_chars//100):], start=max(0, len(lines)-max(1, max_chars//100))))
    else:
        # Keep the beginning, end, and distinctive lines; collapse repeated adjacent lines.
        dedup = []
        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines) and lines[j] == lines[i]: j += 1
            dedup.append(f"{i+1}: {lines[i]}" + (f"  [repeated {j-i} times]" if j-i > 1 else ""))
            i = j
        text = "\n".join(dedup)
        if len(text) <= max_chars: result = text
        else:
            half = max(100, (max_chars-80)//2)
            result = text[:half] + f"\n... {len(lines)} source lines; middle omitted ...\n" + text[-half:]
    if len(result) > max_chars: result = result[:max_chars] + "\n... truncated ..."
    return {"file": str(p.relative_to(ROOT)), "source_lines": len(lines), "source_chars": len(raw), "result": result}


def search_files(query: str, pattern: str = "**/*", case_sensitive: bool = False, limit: int = 200) -> dict:
    """Search source/config/docs while excluding model weights, datasets, caches and build output."""
    if not query: raise ValueError("query cannot be empty")
    needle = query if case_sensitive else query.casefold()
    matches, skipped = [], 0
    for p in ROOT.glob(pattern):
        try:
            rel = p.relative_to(ROOT)
            if not p.is_file() or any(part in SKIP_DIRS or part.startswith(".") and part not in {".github"} for part in rel.parts[:-1]):
                skipped += 1; continue
            if p.suffix.lower() in SKIP_SUFFIXES or p.name.lower() in {"manifest.sha256.json"}:
                skipped += 1; continue
            if p.stat().st_size > MAX_FILE: skipped += 1; continue
            data = p.read_bytes()
            if b"\0" in data[:4096]: skipped += 1; continue
            for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                hay = line if case_sensitive else line.casefold()
                if needle in hay:
                    matches.append({"file": rel.as_posix(), "line": n, "text": line[:500]})
                    if len(matches) >= max(1, min(int(limit), 1000)):
                        return {"matches": matches, "truncated": True, "skipped": skipped}
        except (OSError, UnicodeError, ValueError):
            skipped += 1
    return {"matches": matches, "truncated": False, "skipped": skipped}



def file_map(file: str) -> dict:
    """Outline Python functions/classes and JS/TS declarations without dumping source."""
    p = _path(file); src = _read(p); symbols = []
    if p.suffix == ".py":
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append({"kind": "class" if isinstance(node, ast.ClassDef) else "function", "name": node.name, "line": node.lineno,
                                    "end_line": getattr(node, "end_lineno", node.lineno),
                                    "summary": ast.get_docstring(node) or ""})
        except SyntaxError as e:
            return {"file": str(p.relative_to(ROOT)), "error": f"syntax error at line {e.lineno}: {e.msg}", "symbols": []}
    elif p.suffix.lower() in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        pat = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+([\w$]+)|^\s*(?:export\s+)?(?:const|let)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^\n]*\)|[\w$]+)\s*=>", re.M)
        for m in pat.finditer(src):
            symbols.append({"kind": "declaration", "name": m.group(1) or m.group(2), "line": src.count("\n", 0, m.start())+1})
    else:
        for n, line in enumerate(src.splitlines(), 1):
            if re.match(r"^#{1,4}\s+", line): symbols.append({"kind": "heading", "name": line.lstrip("# ").strip(), "line": n})
    return {"file": str(p.relative_to(ROOT)), "lines": len(src.splitlines()), "symbols": sorted(symbols, key=lambda x: x["line"])}


def errors_only(command: list[str], timeout: int = 120) -> dict:
    """Run only known diagnostic commands; never invokes a shell or arbitrary executable."""
    if not isinstance(command, list) or not command or len(command) > 30: raise ValueError("command must be a non-empty argv list")
    exe = Path(command[0]).name.lower()
    if exe in ("python", "python.exe", "python3", "python3.exe") and command[1:3] == ["-m", "compileall"] and command[3:] == ["-q", "gameai"]:
        argv = [sys.executable, "-m", "compileall", "-q", "gameai"]
    elif exe == "git" and command[1:] == ["status", "--short"]:
        argv = ["git", "status", "--short"]
    elif exe == "git" and command[1:] == ["diff", "--"]:
        argv = ["git", "diff", "--"]
    else:
        raise ValueError("allowed diagnostics: python -m compileall -q gameai, git status --short, or git diff --")
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=min(max(int(timeout), 1), 300), shell=False)
    lines = (p.stdout + "\n" + p.stderr).splitlines()
    hits = [i for i, line in enumerate(lines) if ERROR_RE.search(line)]
    keep = set()
    for i in hits: keep.update(range(max(0, i-2), min(len(lines), i+3)))
    selected = [lines[i] for i in sorted(keep)]
    tail = lines[-12:]
    result = "\n".join(selected + (["... final output ..."] if selected and tail else []) + tail)
    return {"returncode": p.returncode, "ok": p.returncode == 0, "output": (result or "(no output)")[-MAX_OUTPUT:]}



def _vault_dir() -> Path:
    """memory/vault, with the starter notes from vault_seed/ added if they aren't there yet (never overwritten,
    so edits made in Obsidian or by Claude stay)"""
    p = ROOT / VAULT_DIR
    p.mkdir(parents=True, exist_ok=True)
    seed = ROOT / "vault_seed"
    if seed.is_dir():
        for f in seed.glob("*.md"):
            t = p / f.name
            if not t.exists(): t.write_text(f.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    return p


def _slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:70] or "note"


def vault_save(title: str, content: str, tags: list[str] | None = None, links: list[str] | None = None) -> dict:
    """Save a standalone Obsidian-compatible Markdown note; never overwrites an existing note."""
    if not title.strip() or not content.strip(): raise ValueError("title and content are required")
    folder = _vault_dir(); slug = _slug(title); target = folder / f"{slug}.md"; suffix = 2
    while target.exists():
        target = folder / f"{slug}-{suffix}.md"; suffix += 1
    clean_tags = [re.sub(r"[^a-zA-Z0-9_-]", "", str(t))[:40] for t in (tags or [])[:20]]
    clean_links = [_slug(str(x)) for x in (links or [])[:20] if str(x).strip()]
    body = ["---", f'title: {json.dumps(title[:200], ensure_ascii=False)}', "tags:"]
    body += [f"  - {json.dumps(t)}" for t in clean_tags]
    body += ["---", "", content[:12000].rstrip()]
    if clean_links: body += ["", "Related:", *[f"- [[{x}]]" for x in clean_links]]
    target.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    return {"saved": True, "title": title[:200], "file": target.relative_to(ROOT).as_posix(), "links": [f"[[{x}]]" for x in clean_links]}


def vault_search(query: str, limit: int = 20) -> dict:
    folder = _vault_dir(); terms = [t.casefold() for t in re.findall(r"\w+", query) if len(t) > 1]
    rows = []
    for p in folder.glob("*.md"):
        try:
            text = p.read_text(encoding="utf-8")
            title_match = re.search(r'^title:\s*(.+)$', text, re.M)
            title = json.loads(title_match.group(1)) if title_match else p.stem.replace("-", " ")
            score = sum(text.casefold().count(t) for t in terms)
            if score: rows.append({"score": score, "title": title, "file": p.relative_to(ROOT).as_posix(), "content": text[-12000:]})
        except (OSError, ValueError): continue
    rows.sort(key=lambda x: (-x["score"], x["title"].casefold()))
    return {"notes": rows[:max(1, min(int(limit), 100))]}



def run_command(name: str, args: list[str] | None = None, timeout: int = 120) -> dict:
    """Run fixed project diagnostics. 'tests' uses Game AI's own sandboxed game-test runner."""
    args = args or []
    if name not in ALLOWED: raise ValueError(f"command must be one of: {', '.join(ALLOWED)}")
    if name == "list":
        return {"ok": True, "files": sorted(x.name for x in ROOT.iterdir())}
    if name == "tests":
        if len(args) > 1: raise ValueError("tests accepts at most one project folder")
        target = _path(args[0] if args else ".")
        if not target.is_dir(): raise ValueError("tests target must be a project folder")
        from .repair import debugger
        result = debugger.run(str(target), timeout=min(max(int(timeout), 1), 120))
        return {"ok": debugger.ok(result), "result": result, "report": debugger.report(result)}
    if name in ("compile", "git-status") and args: raise ValueError(f"{name} takes no extra arguments")
    if name == "git-diff":
        if len(args) > 1 or (args and args[0].startswith("-")): raise ValueError("git-diff accepts at most one file path")
        if args: _path(args[0])
    argv = ALLOWED[name] + ([str(_path(args[0]))] if name == "git-diff" and args else [])
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, errors="replace",
                       timeout=min(max(int(timeout), 1), 300), shell=False)
    return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout[-MAX_OUTPUT:], "stderr": p.stderr[-MAX_OUTPUT:]}



def batch(operations: list[dict]) -> dict:
    """Execute up to 20 named tools in order; stop on first failure."""
    if not isinstance(operations, list) or not 1 <= len(operations) <= 20: raise ValueError("batch needs 1–20 operations")
    funcs = {"shrink_file": lambda a: shrink_file(**a), "search_files": lambda a: search_files(**a),
             "file_map": lambda a: file_map(**a), "errors_only": lambda a: errors_only(**a),
             "vault_save": lambda a: vault_save(**a), "vault_search": lambda a: vault_search(**a),
             "run_command": lambda a: run_command(**a)}
    results = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict) or op.get("tool") not in funcs: raise ValueError(f"invalid operation at index {i}")
        try: results.append({"tool": op["tool"], "result": funcs[op["tool"]](op.get("args", {}))})
        except Exception as e:
            results.append({"tool": op["tool"], "error": f"{type(e).__name__}: {e}"})
            break
    return {"results": results, "completed": len(results), "total": len(operations)}



# Machine-readable tool catalog used by the app's tool-calling bridge.
TOOL_SPECS = [
    {"name":"shrink_file","description":"Return a compact view of a project file or log without modifying it.","parameters":{"type":"object","properties":{"file":{"type":"string"},"max_chars":{"type":"integer","minimum":200,"maximum":50000},"mode":{"type":"string","enum":["auto","errors","tail"]}},"required":["file"]}},
    {"name":"search_files","description":"Search project files and return matching lines with line numbers.","parameters":{"type":"object","properties":{"query":{"type":"string"},"pattern":{"type":"string"},"case_sensitive":{"type":"boolean"},"limit":{"type":"integer","minimum":1,"maximum":1000}},"required":["query"]}},
    {"name":"file_map","description":"Outline classes, functions, and declarations in a project file.","parameters":{"type":"object","properties":{"file":{"type":"string"}},"required":["file"]}},
    {"name":"errors_only","description":"Run a strictly allow-listed test/build command and return errors plus final output.","parameters":{"type":"object","properties":{"command":{"type":"array","items":{"type":"string"}},"timeout":{"type":"integer","minimum":1,"maximum":300}},"required":["command"]}},
    {"name":"vault_save","description":"Save a durable project note to the local project vault.","parameters":{"type":"object","properties":{"title":{"type":"string"},"content":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"links":{"type":"array","items":{"type":"string"}}},"required":["title","content"]}},
    {"name":"vault_search","description":"Search saved project notes.","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":100}},"required":["query"]}},
    {"name":"run_command","description":"Run fixed approved diagnostics. tests uses Game AI sandbox runner and accepts an optional project-folder argument.","parameters":{"type":"object","properties":{"name":{"type":"string","enum":["tests","compile","git-status","git-diff","list"]},"args":{"type":"array","items":{"type":"string"}},"timeout":{"type":"integer","minimum":1,"maximum":300}},"required":["name"]}},
    {"name":"batch","description":"Run up to 20 workbench tools sequentially; stops after the first failed operation.","parameters":{"type":"object","properties":{"operations":{"type":"array","minItems":1,"maxItems":20,"items":{"type":"object","properties":{"tool":{"type":"string","enum":["shrink_file","search_files","file_map","errors_only","vault_save","vault_search","run_command"]},"args":{"type":"object"}},"required":["tool"]}}},"required":["operations"]}}
]

def call_tool(name: str, arguments: dict | None = None) -> dict:
    """Dispatch one named tool from the explicit catalog; unknown tools are rejected."""
    if not isinstance(arguments if arguments is not None else {}, dict):
        raise ValueError("tool arguments must be an object")
    args = arguments or {}
    funcs = {"shrink_file": shrink_file, "search_files": search_files, "file_map": file_map,
             "errors_only": errors_only, "vault_save": vault_save, "vault_search": vault_search,
             "run_command": run_command, "batch": batch}
    if name not in funcs:
        raise ValueError("unknown tool; use the MCP tools/list catalog for available tools")
    result = funcs[name](**args)
    return {"tool": name, "ok": True, "result": result}


def main():
    ap = argparse.ArgumentParser(description="Game AI local developer tools")
    sub = ap.add_subparsers(dest="tool", required=True)
    q = sub.add_parser("shrink_file"); q.add_argument("file"); q.add_argument("--max-chars", type=int, default=12000); q.add_argument("--mode", choices=("auto","errors","tail"), default="auto")
    q = sub.add_parser("search_files"); q.add_argument("query"); q.add_argument("--pattern", default="**/*")
    q = sub.add_parser("vault_search"); q.add_argument("query")
    q = sub.add_parser("run_command"); q.add_argument("name", choices=ALLOWED)
    a = ap.parse_args()
    if a.tool == "shrink_file": result = shrink_file(a.file, a.max_chars, a.mode)
    elif a.tool == "file_map": result = file_map(a.file)
    elif a.tool == "search_files": result = search_files(a.query, a.pattern)
    elif a.tool == "vault_search": result = vault_search(a.query)
    else: result = run_command(a.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
