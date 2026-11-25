"""python -m gameai.checks install | status | validate FILES... [--write] [--no-kit] | hover|definition|references|completions FILE LINE COL [FILES...]"""
import argparse, json, os, sys
from . import tools, validate, report, hover, definition, references, completions, status


def _read(paths):
    return {os.path.basename(p) if len(paths) == 1 else os.path.relpath(p).replace("\\", "/"): open(p, encoding="utf-8").read() for p in paths}


def main():
    ap = argparse.ArgumentParser(prog="python -m gameai.checks", description="format and check code")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install"); i.add_argument("--csharp", action="store_true")
    sub.add_parser("status")
    v = sub.add_parser("validate"); v.add_argument("files", nargs="+"); v.add_argument("--write", action="store_true", help="save the formatted files")
    v.add_argument("--no-kit", action="store_true", help="don't include GameKit's types"); v.add_argument("--json", action="store_true")
    for name in ("hover", "definition", "references", "completions"):
        q = sub.add_parser(name); q.add_argument("file"); q.add_argument("line", type=int); q.add_argument("col", type=int); q.add_argument("more", nargs="*")
    a = ap.parse_args()
    if a.cmd == "install": print(json.dumps(tools.install(csharp=a.csharp), indent=1)); return
    if a.cmd == "status": print(json.dumps(status(), indent=1)); return
    if a.cmd == "validate":
        files = _read(a.files)
        r = validate(files, kit=not a.no_kit)
        if a.json: print(json.dumps({k: v for k, v in r.items() if k != "files"}, indent=1))
        else:
            for name, d in r["diffs"].items(): print(d)
            print(report(r))
        if a.write:
            for path, name in zip(a.files, files): open(path, "w", encoding="utf-8", newline="").write(r["files"][name])
        sys.exit(0 if r["passed"] else 1)
    files = _read([a.file] + a.more); name = next(iter(files))
    fn = {"hover": hover, "definition": definition, "references": references, "completions": completions}[a.cmd]
    print(json.dumps(fn(files, name, a.line, a.col), indent=1))


if __name__ == "__main__":
    main()
