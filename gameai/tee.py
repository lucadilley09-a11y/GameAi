"""Copy what comes in to the screen and to a file (Windows has no `tee`).  python -m gameai.tee out.txt"""
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    for line in sys.stdin:
        sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
