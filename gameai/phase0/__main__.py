"""Phase 0: measure everything before choosing the base model.

    python -m gameai.phase0 hardware        hardware baseline (about 2 minutes)
    python -m gameai.phase0 fullstack       the whole app at once: memory and speed (about 5 minutes)
    python -m gameai.phase0 profile         where the time goes in training and making games (about 5 minutes)
    python -m gameai.phase0 licences        licence and size of every candidate (nothing downloaded)
    python -m gameai.phase0 bakeoff [model ...] [--minutes 30]   the bake-off, one model at a time (hours)
    python -m gameai.phase0 independence    everything must work with no internet
    python -m gameai.phase0 report          the bake-off table again
    python -m gameai.phase0 all             all of the above, in order
"""
import sys


def main():
    a = sys.argv[1:] or ["help"]
    what, rest = a[0], a[1:]
    minutes = 30
    if "--minutes" in rest:
        i = rest.index("--minutes"); minutes = float(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
    from . import hardware, fullstack, profile, candidates, bakeoff, independence
    steps = {"hardware": hardware.run, "fullstack": fullstack.run, "profile": profile.run, "licences": candidates.record_all,
             "bakeoff": lambda: bakeoff.run(rest or None, minutes), "independence": independence.run, "report": lambda: print(bakeoff.summary())}
    if what == "all":
        for k in ("hardware", "fullstack", "profile", "independence", "bakeoff"):
            try: steps[k]()
            except Exception as e: print(f"{k} failed: {e}", flush=True)
        print(bakeoff.summary())
    elif what in steps:
        steps[what]()
    else:
        print(__doc__)


if __name__ == "__main__":  # needed on Windows: the playtester workers start fresh copies of this program
    main()
