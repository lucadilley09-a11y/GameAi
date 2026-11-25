"""Safety check for code the AI writes, before it ever runs.

Only an allowed list of words may appear: no loops (so it always finishes), no strings, no `new`, `this`,
`window`, `function` or `=>`, and only a few property names. So the code can only do maths on the enemy,
the player and the time, and can't reach anything else on your computer or in the web page.
The same rules are checked in the browser (gameai/play_template.html: safeBrain)."""
import re

MAX_LEN = 400
KEYWORDS = {"let", "const", "if", "else", "return", "true", "false"}
GLOBALS = {"e", "p", "t", "Math", "dist", "dir"}
MATH = {"sin", "cos", "atan2", "hypot", "abs", "min", "max", "sign", "PI", "sqrt", "floor", "round", "random"}
PROPS = {"x", "y", "mem"}
TOKEN = re.compile(r"\s+|(?P<num>\d+(\.\d+)?)|(?P<id>[A-Za-z_]\w*)|(?P<op>==?=?|!==?|<=|>=|&&|\|\||[-+*/%<>!?:;,.(){}\[\]=])")


def check(code):
    """-> (ok, reason)"""
    if len(code) > MAX_LEN: return False, f"too long ({len(code)} characters, max {MAX_LEN})"
    if "=>" in code: return False, "functions inside the code are not allowed"
    pos, prev, local = 0, None, set()
    toks = []
    while pos < len(code):
        m = TOKEN.match(code, pos)
        if not m: return False, f"character not allowed: {code[pos]!r}"
        pos = m.end()
        if m.group("num") or m.group("id") or m.group("op"): toks.append(m.group(0))
    for i, tk in enumerate(toks):
        if re.match(r"[A-Za-z_]", tk):
            before = toks[i - 1] if i else ""
            if before == ".":
                owner = toks[i - 2] if i > 1 else ""
                if owner == "Math":
                    if tk not in MATH: return False, f"Math.{tk} is not allowed"
                elif toks[i - 3:i - 1] == ["e", "."] or (i > 2 and toks[i - 2] == "mem"):
                    if tk.startswith("_") or tk in ("constructor", "prototype"): return False, f"name not allowed: {tk}"
                elif tk not in PROPS:
                    return False, f".{tk} is not allowed"
                continue
            if before in ("let", "const") or (before == "," and tk not in GLOBALS and tk not in KEYWORDS and _declaring(toks, i)):
                if tk in GLOBALS or tk in KEYWORDS or tk in ("window", "this", "globalThis"): return False, f"can't name a variable {tk}"
                local.add(tk); continue
            if tk not in KEYWORDS and tk not in GLOBALS and tk not in local:
                return False, f"unknown name: {tk}"
            if tk == "Math" and (i + 1 >= len(toks) or toks[i + 1] != "."): return False, "Math must be used as Math.something"
    if "return" not in toks: return False, "it never says which way to move (no return)"
    return True, "ok"


def _declaring(toks, i):
    """Is toks[i] (after a comma) another name in a `let a = 1, b = 2` list?"""
    depth = 0
    for j in range(i - 1, -1, -1):
        if toks[j] in (")", "]", "}"): depth += 1
        elif toks[j] in ("(", "[", "{"):
            if depth == 0: return False
            depth -= 1
        elif depth == 0 and toks[j] in (";",): return False
        elif depth == 0 and toks[j] in ("let", "const"): return True
    return False
