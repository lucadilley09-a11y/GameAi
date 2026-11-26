"""python -m gameai.ui page.html [--out folder]   layout check at desktop and phone size, with screenshots"""
import argparse
from . import render

ap = argparse.ArgumentParser(prog="python -m gameai.ui"); ap.add_argument("page"); ap.add_argument("--out", default=None)
a = ap.parse_args()
r = render.check(a.page, out_dir=a.out)
print(render.report(r))
for name, s in r["sizes"].items():
    if s["screenshot"]: print(f"{name} screenshot: {s['screenshot']}")
