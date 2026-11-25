"""Experiment tracking: every run gets a folder under runs/ with its exact config,
environment, data hash, metrics over time and final summary. Fully local, no account.

    python -m gameai.track            # table of all runs
Optional live dashboard: pip install trackio, then set GAMEAI_TRACKIO=1 before training
and run `trackio show --project gameai`.
"""
import json, os, platform, subprocess, sys, time


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class Run:
    def __init__(self, name, config, root="runs", kind="train"):
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + name
        self.dir = os.path.join(root, self.id)
        self.root = root
        os.makedirs(self.dir, exist_ok=True)
        env = {"python": sys.version.split()[0], "platform": platform.platform(), "git": _git_hash(), "kind": kind}
        try:
            import torch
            env["torch"] = torch.__version__
            env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
        manifest = "data/manifest.json"
        if os.path.exists(manifest):
            env["data"] = json.load(open(manifest))
        json.dump(config, open(os.path.join(self.dir, "config.json"), "w"), indent=2)
        json.dump(env, open(os.path.join(self.dir, "env.json"), "w"), indent=2)
        self._f = open(os.path.join(self.dir, "metrics.jsonl"), "a")
        self._t0 = time.time()
        self._tk = None
        if os.environ.get("GAMEAI_TRACKIO"):
            try:
                import trackio
                trackio.init(project="gameai", name=self.id, config=config)
                self._tk = trackio
            except Exception as e:
                print("trackio not available:", e)

    def log(self, step, **metrics):
        row = {"step": step, "time": round(time.time() - self._t0, 2), **metrics}
        self._f.write(json.dumps(row) + "\n"); self._f.flush()
        if self._tk:
            self._tk.log(metrics, step=step)

    def finish(self, **summary):
        summary = {"id": self.id, "minutes": round((time.time() - self._t0) / 60, 2), **summary}
        json.dump(summary, open(os.path.join(self.dir, "summary.json"), "w"), indent=2)
        with open(os.path.join(self.root, "index.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")
        self._f.close()
        if self._tk:
            self._tk.finish()
        return summary


def main():
    path = "runs/index.jsonl"
    if not os.path.exists(path):
        print("no runs yet"); return
    rows = [json.loads(l) for l in open(path)]
    keys = ["id", "minutes", "params", "val_loss", "field_acc", "exact", "typo_exact", "instruction_match", "promoted"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
