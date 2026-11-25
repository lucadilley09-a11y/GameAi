import argparse, json
from . import project as P, components as C


def main():
    ap = argparse.ArgumentParser(prog="python -m gameai.planner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("project")
    c = sub.add_parser("components"); c.add_argument("words", nargs="*")
    u = sub.add_parser("unblock"); u.add_argument("project"); u.add_argument("feature"); u.add_argument("note", nargs="*")
    a = ap.parse_args()
    if a.cmd == "status":
        plan = P.load(a.project); print(P.summary(plan) if plan else "no plan in this folder")
    elif a.cmd == "components":
        found = C.search(" ".join(a.words), k=20) if a.words else C.load()
        for x in found: print(f"{x['name']} ({x['lang']}, used {x['uses']}x): {x['description']}")
    elif a.cmd == "unblock":
        from .builder import unblock
        f = unblock(a.project, a.feature, " ".join(a.note)); print("ready to try again" if f else "no such blocked feature")


if __name__ == "__main__":
    main()
