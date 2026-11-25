"""Repository map (plan: Phase 4): a Tree-sitter index of a project's file paths, imports, exports, classes, functions,
types and their signatures (JavaScript/TypeScript, Python, Lua), so the AI sees what already exists before it writes code that uses it. This means fewer
invented functions and fewer wrong imports.

The map is compact (signatures only, no bodies) but not cut to a fixed size: every file and every top-level name is
listed, and the files that matter for the current task (focus) come first with more detail.

    from gameai.repair import repomap
    m = repomap.build({"src/player.js": "...", ...})     # or repomap.build_dir("my_game")
    print(repomap.render(m, focus=["src/player.js"]))
    repomap.find(m, "spawnEnemy")                        # where it is defined, and its signature
"""
import os, re

LANGS = {".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
         ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
         ".py": "python", ".lua": "lua"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".history", ".staging", "dist", "build", "bin", "obj"}
_parsers = {}


def parser(lang):
    """a Tree-sitter parser for one language (the grammars are small pip packages)"""
    if lang not in _parsers:
        import tree_sitter as ts
        if lang == "javascript": import tree_sitter_javascript as m; L = m.language()
        elif lang == "typescript": import tree_sitter_typescript as m; L = m.language_typescript()
        elif lang == "tsx": import tree_sitter_typescript as m; L = m.language_tsx()
        elif lang == "python": import tree_sitter_python as m; L = m.language()
        elif lang == "lua": import tree_sitter_lua as m; L = m.language()
        else: raise ValueError(lang)
        _parsers[lang] = ts.Parser(ts.Language(L))
    return _parsers[lang]


def lang_of(path):
    return LANGS.get(os.path.splitext(path)[1].lower())


def _text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _sig(node, src):
    """the definition's first part, up to its body: 'function move(dx, dy)', 'class Enemy extends Thing'"""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    s = src[node.start_byte:end].decode("utf-8", "replace")
    s = re.sub(r"\s+", " ", s).strip().rstrip("{:= ").strip()
    s = re.sub(r"\s+end$", "", s)  # Lua functions with an empty body
    return s[:200]


def _name(node, src):
    n = node.child_by_field_name("name")
    return _text(n, src) if n is not None else None


# ---------- per language: what counts as a definition ----------
JS_FUNCS = {"function_declaration", "generator_function_declaration", "function_signature"}
JS_CLASS = {"class_declaration", "abstract_class_declaration"}
TS_TYPES = {"interface_declaration", "type_alias_declaration", "enum_declaration"}


def _js(root, src, out):
    def var_decls(decl, exported):
        for d in decl.children:
            if d.type != "variable_declarator": continue
            name = _name(d, src); val = d.child_by_field_name("value")
            if not name: continue
            if val is not None and val.type in ("arrow_function", "function_expression", "function"):
                params = val.child_by_field_name("parameters") or val.child_by_field_name("parameter")
                sig = f"{'const ' if decl.type == 'lexical_declaration' else ''}{name} = ({_text(params, src).strip('()') if params is not None else ''}) =>"
                add("function", name, sig, d, exported)
            elif val is not None and val.type == "class":
                add("class", name, f"class {name}", d, exported); members(val, name)
            else:
                add("variable", name, re.sub(r"\s+", " ", _text(d, src))[:120], d, exported)

    def members(cls, cname):
        body = cls.child_by_field_name("body")
        if body is None: return
        for m in body.children:
            if m.type in ("method_definition", "method_signature", "abstract_method_signature"):
                add("method", f"{cname}.{_name(m, src)}", _sig(m, src), m, False)
            elif m.type in ("public_field_definition", "field_definition"):
                nm = m.child_by_field_name("name") or m.child_by_field_name("property")
                if nm is not None: add("field", f"{cname}.{_text(nm, src)}", re.sub(r"\s+", " ", _text(m, src))[:120], m, False)

    def add(kind, name, sig, node, exported):
        out["symbols"].append({"kind": kind, "name": name, "signature": sig, "line": node.start_point[0] + 1,
                               "end_line": node.end_point[0] + 1, "exported": exported})
        if exported: out["exports"].append(name.split(".")[0])

    def visit(n, exported=False):
        t = n.type
        if t == "export_statement":
            decl = n.child_by_field_name("declaration")
            if decl is not None: visit(decl, True)
            else:
                for c in n.children:  # export { a, b as c }  /  export default x
                    if c.type == "export_clause":
                        for s in c.children:
                            if s.type == "export_specifier":
                                al = s.child_by_field_name("alias") or s.child_by_field_name("name")
                                out["exports"].append(_text(al, src))
                    elif c.type in ("identifier",): out["exports"].append("default")
            src_node = n.child_by_field_name("source")
            if src_node is not None: out["imports"].append(_text(src_node, src).strip("'\""))
        elif t in JS_FUNCS: add("function", _name(n, src), _sig(n, src), n, exported)
        elif t in JS_CLASS:
            cname = _name(n, src); add("class", cname, _sig(n, src), n, exported); members(n, cname)
        elif t in TS_TYPES:
            tname = _name(n, src)
            add("type", tname, _sig(n, src) if t != "type_alias_declaration" else re.sub(r"\s+", " ", _text(n, src))[:160], n, exported)
            body = n.child_by_field_name("body")
            for m in (body.children if body is not None and t == "interface_declaration" else []):
                if m.type in ("method_signature", "property_signature"):
                    nm = m.child_by_field_name("name")
                    if nm is not None: add("method" if m.type == "method_signature" else "field", f"{tname}.{_text(nm, src)}",
                                           re.sub(r"\s+", " ", _text(m, src)).rstrip(";,")[:160], m, False)
        elif t in ("lexical_declaration", "variable_declaration"): var_decls(n, exported)
        elif t == "import_statement":
            s = n.child_by_field_name("source")
            if s is not None: out["imports"].append(_text(s, src).strip("'\""))
        elif t == "ambient_declaration" or t == "internal_module" or t == "module":
            name = _name(n, src)
            if name: add("namespace", name, _sig(n, src), n, exported)
            for c in n.children:
                if c.type == "statement_block":
                    for cc in c.children: visit(cc, exported)
                elif c.type not in ("declare", "namespace", "module"): visit(c, exported)
        elif t == "expression_statement":  # require('x') and module.exports = ...
            txt = _text(n, src)
            for m in re.finditer(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", txt): out["imports"].append(m.group(1))
            m = re.match(r"(?:module\.)?exports\.(\w+)\s*=", txt)
            if m: out["exports"].append(m.group(1))

    for c in root.children: visit(c)


def _py(root, src, out):
    def add(kind, name, sig, node):
        out["symbols"].append({"kind": kind, "name": name, "signature": sig, "line": node.start_point[0] + 1,
                               "end_line": node.end_point[0] + 1, "exported": not name.split(".")[-1].startswith("_")})

    def visit(n, cls=None):
        d = n.child_by_field_name("definition") if n.type == "decorated_definition" else n
        if d is None: return
        if d.type == "function_definition":
            name = _name(d, src); add("method" if cls else "function", f"{cls}.{name}" if cls else name, _sig(d, src), n)
        elif d.type == "class_definition":
            name = _name(d, src); add("class", name, _sig(d, src), n)
            body = d.child_by_field_name("body")
            for c in (body.children if body is not None else []): visit(c, name)
        elif d.type in ("import_statement", "import_from_statement") and not cls:
            m = d.child_by_field_name("module_name")
            if m is not None: out["imports"].append(_text(m, src))
            else: out["imports"] += [_text(c, src) for c in d.children if c.type == "dotted_name"]
        elif d.type == "expression_statement" and not cls:
            a = d.children[0] if d.children else None
            if a is not None and a.type == "assignment":
                left = a.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    add("variable", _text(left, src), re.sub(r"\s+", " ", _text(a, src))[:120], n)

    for c in root.children: visit(c)
    out["exports"] = [s["name"] for s in out["symbols"] if s["exported"] and "." not in s["name"]]


def _lua(root, src, out):
    for n in root.children:
        if n.type == "function_declaration":
            name = _name(n, src) or ""
            local = _text(n, src).lstrip().startswith("local")
            out["symbols"].append({"kind": "function", "name": name.replace(":", "."), "signature": _sig(n, src), "line": n.start_point[0] + 1,
                                   "end_line": n.end_point[0] + 1, "exported": not local})
        elif n.type in ("variable_declaration", "assignment_statement"):
            txt = re.sub(r"\s+", " ", _text(n, src))
            m = re.match(r"(local\s+)?([\w.]+)\s*=", txt)
            if m: out["symbols"].append({"kind": "variable", "name": m.group(2), "signature": txt[:120], "line": n.start_point[0] + 1,
                                         "end_line": n.end_point[0] + 1, "exported": not m.group(1)})
        for m in re.finditer(r"require\s*\(?\s*['\"]([^'\"]+)['\"]", _text(n, src)): out["imports"].append(m.group(1))
    out["exports"] = [s["name"] for s in out["symbols"] if s["exported"]]


def index_file(path, text):
    lang = lang_of(path)
    out = {"path": path, "lang": lang, "lines": text.count("\n") + 1, "imports": [], "exports": [], "symbols": [], "parse_errors": 0}
    if not lang: return out
    src = text.encode("utf-8")
    tree = parser(lang).parse(src)
    out["parse_errors"] = sum(1 for _ in _errors(tree.root_node))
    {"javascript": _js, "typescript": _js, "tsx": _js, "python": _py, "lua": _lua}[lang](tree.root_node, src, out)
    out["symbols"] = [s for s in out["symbols"] if s["name"]]
    out["exports"] = sorted(set(e for e in out["exports"] if e))
    return out


def _errors(node):
    if node.type == "ERROR" or node.is_missing: yield node
    if node.has_error:
        for c in node.children: yield from _errors(c)


def build(files):
    """files: {path: text} -> the map {path: file index}"""
    return {p: index_file(p, t) for p, t in sorted(files.items())}


def read_dir(root):
    files = {}
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for n in sorted(names):
            if lang_of(n) or n.endswith((".json", ".html", ".css", ".md")):
                p = os.path.join(d, n)
                if os.path.getsize(p) > 2_000_000: continue
                try: files[os.path.relpath(p, root).replace("\\", "/")] = open(p, encoding="utf-8").read()
                except UnicodeDecodeError: pass
    return files


def build_dir(root):
    return build(read_dir(root))


def find(m, name):
    """every definition of a name ('spawn', 'Enemy.update' or just 'update')"""
    hits = []
    for f in m.values():
        for s in f["symbols"]:
            if s["name"] == name or s["name"].split(".")[-1] == name: hits.append({"file": f["path"], **s})
    return hits


def related(m, focus):
    """files connected to the focus files: what they import, and what imports them"""
    focus = [f for f in focus if f in m]
    out = set(focus)
    for f in focus:
        base = os.path.splitext(f)[0]
        for imp in m[f]["imports"]:
            for p in m:
                if os.path.splitext(p)[0].endswith(imp.lstrip("./").rsplit(".", 1)[0] if imp.startswith(".") else imp.replace(".", "/")): out.add(p)
        for p, info in m.items():
            if any(base.endswith(i.lstrip("./").rsplit(".", 1)[0]) for i in info["imports"] if i.startswith(".")): out.add(p)
    return out


def render(m, focus=()):
    """the map as compact text. Focus files (and files connected to them) come first with every member;
    the rest list their top-level names with signatures."""
    near = related(m, list(focus)) if focus else set()
    order = sorted(m, key=lambda p: (p not in focus, p not in near, p))
    lines = []
    for p in order:
        f = m[p]
        head = f"{p}  ({f['lines']} lines" + (f", {f['parse_errors']} syntax errors" if f["parse_errors"] else "") + ")"
        lines.append(head)
        if f["imports"]: lines.append("  imports: " + ", ".join(dict.fromkeys(f["imports"])))
        detail = p in near or not focus
        for s in f["symbols"]:
            if "." in s["name"] and not detail: continue
            mark = "export " if s["exported"] and f["lang"] in ("javascript", "typescript", "tsx") else ""
            lines.append(f"  {'  ' if '.' in s['name'] else ''}{s['line']}: {mark if not s['signature'].startswith('export') else ''}{s['signature']}")
        if not detail and any("." in s["name"] for s in f["symbols"]):
            n = sum(1 for s in f["symbols"] if "." in s["name"]); lines.append(f"    (+{n} members)")
    return "\n".join(lines)
