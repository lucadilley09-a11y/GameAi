"""Every AI version that was ever in use is kept in models/versions/ (the last 20).

    python -m gameai.versions            list them
    python -m gameai.versions 3          go back to version 3 (or /rollback 3 in the app)
"""
import json, os, shutil, sys

DIR = "models/versions"


def list_versions():
    if not os.path.isdir(DIR): return []
    out = []
    for f in sorted(os.listdir(DIR)):
        if f.endswith(".json"):
            d = json.load(open(os.path.join(DIR, f)))
            out.append({"version": d.get("version") or int(f[1:4]), "time": d.get("time"), "total": d.get("total"),
                        "frozen": (d.get("frozen") or {}).get("exact"), "gold": (d.get("gold") or {}).get("exact"),
                        "note": d.get("note") or d.get("why")})
    cur = json.load(open("models/current.json")).get("version") if os.path.exists("models/current.json") else None
    for v in out: v["in_use"] = v["version"] == cur
    return out


def rollback(n=None):
    vs = list_versions()
    if not vs: raise ValueError("No saved versions yet. They're kept from the next time the AI improves.")
    cur = next((v["version"] for v in vs if v["in_use"]), vs[-1]["version"])
    if n is None:
        older = [v["version"] for v in vs if v["version"] < cur]
        if not older: raise ValueError("There's no older version to go back to.")
        n = older[-1]
    pt, js = f"{DIR}/v{n:03d}.pt", f"{DIR}/v{n:03d}.json"
    if not os.path.exists(pt): raise ValueError(f"Version {n} isn't kept anymore.")
    shutil.copy(pt, "models/current.pt.new"); os.replace("models/current.pt.new", "models/current.pt")
    shutil.copy(js, "models/current.json")
    return n


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Now using version {rollback(int(sys.argv[1]))}.")
    else:
        for v in list_versions():
            print(f"v{v['version']:03d}  {'(in use) ' if v['in_use'] else ''}score {v['total'] or 0:.2%}  {v['note'] or ''}")
