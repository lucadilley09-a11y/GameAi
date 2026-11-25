"""Suppression rules (plan: Phase 4): ways of hiding a problem instead of fixing it (any, @ts-ignore, eslint-disable
and similar) are allowed only with an explicit justification; unjustified ones fail validation.

A justification is at least 3 words, written either
  - on the same line, after " -- " (ESLint's own style) or after "because" / "reason:" / "justification:" / "why:"
        // eslint-disable-next-line no-console -- the debug overlay prints on purpose
        const data: any = JSON.parse(text);  // because the save file format is checked right below
  - or as a comment on the line directly above, starting with one of those words
        // justification: the browser API has no types for this yet
        // @ts-expect-error
"""
import re

# (languages, kind, pattern). Patterns run on the line with its string contents blanked out, so text inside quotes
# never counts.
JS, TS, PY, LUA = "js", "ts", "py", "lua"
RULES = [
    ({JS, TS}, "ts-ignore", r"@ts-ignore\b"),
    ({JS, TS}, "ts-expect-error", r"@ts-expect-error\b"),
    ({JS, TS}, "ts-nocheck", r"@ts-nocheck\b"),
    ({JS, TS}, "eslint-disable", r"eslint-disable(?:-next-line|-line)?\b"),
    ({TS}, "any", r"(?::\s*any\b|\bas\s+any\b|<any>|\bany\s*\[\]|<\s*any\s*[,>])"),
    ({TS}, "double cast", r"\bas\s+unknown\s+as\b"),
    ({JS, TS}, "any (JSDoc)", r"[@](?:type|param|returns?)\s*\{\s*(?:any|\*)\s*\}"),
    ({PY}, "type: ignore", r"#\s*type:\s*ignore\b"),
    ({PY}, "noqa", r"#\s*noqa\b"),
    ({PY}, "pyright/pylint off", r"#\s*(?:pyright:\s*ignore|pylint:\s*disable)\b"),
    ({PY}, "Any", r"(?::\s*Any\b|->\s*Any\b|\bcast\(\s*Any\b)"),
    ({LUA}, "diagnostic off", r"---?\s*@diagnostic\s+disable"),
    ({LUA}, "luacheck off", r"--\s*luacheck:\s*ignore\b"),
]
LANG_OF = {".js": JS, ".mjs": JS, ".cjs": JS, ".jsx": JS, ".ts": TS, ".tsx": TS, ".mts": TS, ".cts": TS,
           ".py": PY, ".lua": LUA}
WHY_SAME_LINE = re.compile(r"(?:(?<=\s)--\s+|\bbecause\b|\breason:|\bjustification:|\bwhy:)\s*(.*)", re.I)
WHY_ABOVE = re.compile(r"^\s*(?://|#|--|/\*|\*)\s*(?:because\b|reason:|justification:|why:)\s*(.*)", re.I)
MIN_WORDS = 3
STRING = re.compile(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)""")


def _blank_strings(line, lang):
    if lang == LUA and "--" in line:  # in Lua, "--" starts a comment, so only blank strings before it
        i = line.index("--"); return STRING.sub(lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1], line[:i]) + line[i:]
    return STRING.sub(lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1], line)


def _words(s):
    s = re.sub(r"\*/\s*$", "", s or "").strip()
    return len(re.findall(r"[A-Za-z]{2,}", s))


def suppressions(files):
    """-> [{file, line, col, line_text, kind, justified, message}] for every suppression in the files"""
    out = []
    for name, text in files.items():
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        lang = LANG_OF.get(ext)
        if not lang: continue
        lines = text.split("\n")
        rules = [(k, re.compile(p)) for langs, k, p in RULES if lang in langs]
        for i, raw in enumerate(lines):
            line = _blank_strings(raw, lang)
            for kind, rx in rules:
                m = rx.search(line)
                if not m: continue
                after = WHY_SAME_LINE.search(line[m.end():]) or WHY_SAME_LINE.search(line)
                above = WHY_ABOVE.match(lines[i - 1]) if i else None
                ok = bool(after and _words(after.group(1)) >= MIN_WORDS) or bool(above and _words(above.group(1)) >= MIN_WORDS)
                out.append({"file": name, "line": i + 1, "col": m.start() + 1, "line_text": raw, "kind": kind, "justified": ok,
                            "message": f"{kind} " + ("is justified" if ok else "hides a problem without a reason: fix it, or add a reason "
                                                     "of at least 3 words (after ' -- ' or 'because')")})
    return out
