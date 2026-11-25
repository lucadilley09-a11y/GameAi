"""Automatic updates from the project's GitHub repo (set in update.json).

    python -m gameai.update          check once and install a newer version if there is one

start.bat runs this before opening the app, and the app checks again every 15 seconds while it runs.
Only code is replaced. Your trained AI, data, games, ratings and settings are never touched.
"""
import io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEEP = {"data", "models", "runs", "bench", "checkpoints", "learn", "memory", "games", "play", "for-chromebook", "export",
        ".venv", "logs", "update.json", ".version"}  # never overwritten by an update
API = "https://api.github.com/repos/{repo}/commits/{branch}"
ZIP = "https://codeload.github.com/{repo}/zip/{sha}"


def config():
    p = os.path.join(ROOT, "update.json")
    return json.load(open(p)) if os.path.exists(p) else None


def local_version():
    p = os.path.join(ROOT, ".version")
    return json.load(open(p)) if os.path.exists(p) else {"sha": None}


def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "game-ai-updater", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest(cfg):
    d = json.loads(get(API.format(repo=cfg["repo"], branch=cfg.get("branch", "main"))))
    return {"sha": d["sha"], "message": d["commit"]["message"].split("\n")[0], "date": d["commit"]["committer"]["date"]}


def head_sha(cfg):
    """The newest version's id, in one tiny request (about 0.5 KB). This isn't GitHub's API, so it can be
    checked every few seconds without hitting GitHub's limit of 60 API calls an hour."""
    import re
    req = urllib.request.Request(f"https://github.com/{cfg['repo']}.git/info/refs?service=git-upload-pack", headers={"User-Agent": "git/2.40 game-ai"})
    with urllib.request.urlopen(req, timeout=10) as r:
        text = r.read().decode("latin1")
    m = re.search(r"([0-9a-f]{40}) refs/heads/" + re.escape(cfg.get("branch", "main")) + r"\b", text)
    return m.group(1) if m else None


def apply_zip(data, root=ROOT):
    """Copy the code from a GitHub zip into the project, skipping everything in KEEP. Returns changed paths."""
    z = zipfile.ZipFile(io.BytesIO(data))
    top = z.namelist()[0].split("/")[0] + "/"
    changed = []
    for name in z.namelist():
        if not name.startswith(top) or name.endswith("/"):
            continue
        rel = name[len(top):]
        if not rel or rel.split("/")[0] in KEEP or "__pycache__" in rel:
            continue
        dest = os.path.join(root, *rel.split("/"))
        new = z.read(name)
        if os.path.exists(dest) and open(dest, "rb").read() == new:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".new"
        open(tmp, "wb").write(new)
        os.replace(tmp, dest)  # atomic swap: a half-written file is never used
        changed.append(rel)
    return changed


def check(install=True, quiet=False):
    """-> dict with 'available', 'installed', 'message', 'changed'."""
    if os.environ.get("GAMEAI_NO_UPDATE"): return {"available": False, "error": "updates are off on this copy"}
    cfg = config()
    if not cfg:
        return {"available": False, "error": "no update.json yet"}
    try:
        new = latest(cfg)
    except Exception as e:
        try:  # GitHub's API said no (often its hourly limit): use the tiny check instead
            sha = head_sha(cfg)
            if not sha: raise RuntimeError("no version found")
            new = {"sha": sha, "message": "new version", "date": ""}
        except Exception:
            return {"available": False, "error": f"could not reach GitHub: {e}"}
    cur = local_version()
    if new["sha"] == cur.get("sha"):
        return {"available": False, "version": new["sha"][:7]}
    if not install:
        return {"available": True, "message": new["message"]}
    try:
        data = get(ZIP.format(repo=cfg["repo"], sha=new["sha"]), timeout=120)
        changed = apply_zip(data)
    except Exception as e:
        return {"available": True, "error": f"download failed, kept the current version: {e}"}
    if "requirements.txt" in changed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(ROOT, "requirements.txt")])
    json.dump({**new, "installed": time.time()}, open(os.path.join(ROOT, ".version"), "w"))
    if not quiet:
        print(f"Updated: {new['message']} ({len(changed)} files)")
    return {"available": True, "installed": True, "message": new["message"], "changed": changed, "version": new["sha"][:7]}


def main():
    r = check()
    if r.get("installed"): return
    print(r.get("error") or f"Game AI is up to date ({r.get('version', '?')}).")


if __name__ == "__main__":
    main()
