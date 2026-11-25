"""Structured edits (plan: Phase 4): instead of rewriting whole files, the AI changes code with small, checked edits.

    {"file": "src/player.js", "search": "speed = 3", "replace": "speed = 4"}          # exact text, must match once
    {"file": "src/player.js", "search": "hp", "replace": "health", "count": 5}       # ...or exactly 5 times
    {"file": "src/player.js", "symbol": "Player.move", "replace": "move(dx, dy) {...}"}  # a whole function/method/class
    {"file": "src/player.js", "insert_after": "speed = 3;", "text": "\\n  jumps = 2;"}  # also insert_before
    {"file": "src/new.js", "create": "..."}      {"file": "src/old.js", "delete": true}
Any edit may carry "version": the file version the AI saw (version(text)); if the file changed since, it's refused.

Rules: an anchor must match exactly the expected number of times (default 1), so an edit never lands in the wrong
place; all edits apply together or none do; after each edit the file is re-parsed, and an edit that breaks the
syntax of a file that was fine is refused, naming that edit. The full checks (format, types, lint, tests) run on
the whole change afterwards (atomic.py).
"""
import difflib, hashlib

from . import repomap


class EditError(Exception):
    def __init__(self, problems):
        self.problems = problems
        super().__init__("\n".join(problems))


def version(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _lines_with(text, needle):
    out, start = [], 0
    while True:
        i = text.find(needle, start)
        if i < 0: return out
        out.append(text.count("\n", 0, i) + 1); start = i + max(1, len(needle))


def _closest(text, needle):
    """the part of the file most like a missed anchor, to help the AI fix it"""
    lines, n = text.split("\n"), max(1, needle.count("\n") + 1)
    windows = ["\n".join(lines[i:i + n]) for i in range(max(1, len(lines) - n + 1))]
    best = max(range(len(windows)), key=lambda i: difflib.SequenceMatcher(None, windows[i], needle).ratio(), default=None)
    if best is None: return ""
    ratio = difflib.SequenceMatcher(None, windows[best], needle).ratio()
    return f" The closest text is at line {best + 1} ({ratio:.0%} similar):\n{windows[best][:400]}" if ratio > 0.5 else ""


def _syntax_errors(path, text):
    lang = repomap.lang_of(path)
    if not lang: return 0
    try: return repomap.index_file(path, text)["parse_errors"]
    except Exception: return 0


def _symbol_span(path, text, name):
    """(first line, last line) of a definition, 1-based"""
    hits = [s for s in repomap.index_file(path, text)["symbols"] if s["name"] == name]
    if not hits:
        short = [s["name"] for s in repomap.index_file(path, text)["symbols"] if s["name"].split(".")[-1] == name.split(".")[-1]]
        raise ValueError(f"no definition named {name!r}" + (f" (did you mean {', '.join(short)}?)" if short else ""))
    if len(hits) > 1: raise ValueError(f"{name!r} is defined {len(hits)} times (lines {', '.join(str(h['line']) for h in hits)}); use a search edit")
    return hits[0]["line"], hits[0]["end_line"]


def apply(files, edits):
    """files: {path: text}; edits: a list of edits. -> (new files, [one line per edit saying what changed]).
    Raises EditError at the first problem; then nothing is changed."""
    out, done, problems = dict(files), [], []
    for i, e in enumerate(edits, 1):
        f = e.get("file")
        tag = f"edit {i} ({f})"
        try:
            if not f: raise ValueError("no file given")
            if e.get("version") and f in files and version(files[f]) != e["version"]:
                raise ValueError("the file changed since it was read: read it again and redo this edit")
            if "create" in e:
                if f in out and not e.get("overwrite"): raise ValueError("the file already exists (use edits, or overwrite: true)")
                out[f] = e["create"]; done.append(f"{tag}: created ({e['create'].count(chr(10)) + 1} lines)"); continue
            if e.get("delete"):
                if f not in out: raise ValueError("there is no such file")
                del out[f]; done.append(f"{tag}: deleted"); continue
            if f not in out: raise ValueError("there is no such file" + (f"; files: {', '.join(sorted(out)[:20])}" if out else ""))
            text, before_errors = out[f], _syntax_errors(f, out[f])
            if "symbol" in e:
                a, b = _symbol_span(f, text, e["symbol"])
                lines = text.split("\n")
                new = e["replace"].rstrip("\n").split("\n")
                out[f] = "\n".join(lines[:a - 1] + new + lines[b:])
                done.append(f"{tag}: replaced {e['symbol']} (lines {a}-{b})")
            else:
                anchor = e.get("search", e.get("insert_after", e.get("insert_before")))
                if not anchor: raise ValueError("an edit needs search, symbol, insert_after, insert_before, create or delete")
                want, where = int(e.get("count", 1)), _lines_with(text, anchor)
                if len(where) != want:
                    if not where: raise ValueError(f"the text to find wasn't found.{_closest(text, anchor)}")
                    raise ValueError(f"the text to find appears {len(where)} times (lines {', '.join(map(str, list(dict.fromkeys(where))[:12]))}), "
                                     f"expected {want}: include more surrounding text so it's unique, or set count")
                if "search" in e: new = text.replace(anchor, e["replace"])
                elif "insert_after" in e: new = text.replace(anchor, anchor + e["text"])
                else: new = text.replace(anchor, e["text"] + anchor)
                out[f] = new
                done.append(f"{tag}: changed {want} place(s), line {', '.join(map(str, where[:6]))}")
            if not before_errors and _syntax_errors(f, out[f]):
                raise ValueError("after this edit the file no longer parses (unbalanced brackets or quotes?)")
        except (ValueError, KeyError, TypeError) as ex:  # later edits may depend on this one, so stop here
            problems.append(f"{tag}: {ex}")
            if i < len(edits): problems.append(f"edits {i + 1}-{len(edits)} were not tried")
            raise EditError(problems)
    return out, done
