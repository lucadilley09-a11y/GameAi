import argparse, json, sys
from . import repomap, debugger, atomic


def main():
    ap = argparse.ArgumentParser(prog="python -m gameai.repair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map"); m.add_argument("project"); m.add_argument("--focus", nargs="*", default=[])
    f = sub.add_parser("find"); f.add_argument("project"); f.add_argument("name")
    t = sub.add_parser("test"); t.add_argument("project"); t.add_argument("--inspect", action="store_true")
    a = sub.add_parser("apply"); a.add_argument("project"); a.add_argument("edits", help="a JSON file with a list of edits")
    h = sub.add_parser("history"); h.add_argument("project")
    r = sub.add_parser("rollback"); r.add_argument("project")
    x = ap.parse_args()
    if x.cmd == "map": print(repomap.render(repomap.build_dir(x.project), focus=x.focus))
    elif x.cmd == "find": print(json.dumps(repomap.find(repomap.build_dir(x.project), x.name), indent=1))
    elif x.cmd == "test":
        res = debugger.run(x.project, inspect=x.inspect); print(debugger.report(res)); sys.exit(0 if debugger.ok(res) else 1)
    elif x.cmd == "apply":
        res = atomic.change(x.project, json.load(open(x.edits, encoding="utf-8")))
        for d in res["diffs"].values(): print(d)
        print(res["report"]); sys.exit(0 if res["applied"] else 1)
    elif x.cmd == "history":
        for e in atomic.history(x.project): print(e["time"], "; ".join(e["edits"]))
    elif x.cmd == "rollback":
        man = atomic.rollback(x.project)
        print("nothing to undo" if not man else "undone: " + "; ".join(man["edits"]))


if __name__ == "__main__":
    main()
