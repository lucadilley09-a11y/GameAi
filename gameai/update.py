"""Automatic updates from the project's GitHub repo (set in update.json).

    python -m gameai.update          check once and install a newer version if there is one

    python -m gameai.update --rollback   put the previous version back

start.bat runs this before opening the app, and the app checks again every 15 seconds while it runs.
Only code is replaced. Your trained AI, data, games, ratings and settings are never touched.

Safety: updates come only from the repo and exact version (commit) named in update.json, every file is checked
against the version's SHA-256 list (MANIFEST.sha256.json) and GitHub's own file hashes before anything is
changed, the previous version is kept in backup/previous, and update.json's "mode" can be "instant"
(default while the AI is being developed), "manual" (ask first) or "off" (never goes online to update).
"""
import hashlib, io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEEP = {"data", "models", "runs", "bench", "checkpoints", "learn", "memory", "games", "play", "for-chromebook", "export",
        ".venv", "logs", "update.json", ".version", "backup", "sealed", "runtime"}  # never overwritten by an update
API = "https://api.github.com/repos/{repo}/commits/{branch}"
TREE = "https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"
MANIFEST = "MANIFEST.sha256.json"
BACKUP = os.path.join(ROOT, "backup", "previous")
MODES = ("instant", "manual", "off")
ZIP = "https://codeload.github.com/{repo}/zip/{sha}"


def config():
    p = os.path.join(ROOT, "update.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save_config(**changes):
    cfg = {**(config() or {}), **changes}
    json.dump(cfg, open(os.path.join(ROOT, "update.json"), "w"), indent=1)
    return cfg


def mode(cfg=None):
    m = (cfg or config() or {}).get("mode", "instant")
    return m if m in MODES else "instant"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    """the hash git (and GitHub) use for a file's contents"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_code(rel):
    """a program file that updates install (not your AI, data, settings or the SHA-256 list itself)"""
    return rel != MANIFEST and rel.split("/")[0] not in KEEP and "__pycache__" not in rel and not rel.endswith((".pyc", ".new"))


def code_files(root=None):
    """every program file of the project on this PC"""
    root, out = root or ROOT, []
    for d, dirs, files in os.walk(root):
        rel_d = os.path.relpath(d, root).replace(os.sep, "/")
        if rel_d == ".": dirs[:] = [x for x in dirs if x not in KEEP and x not in (".git", ".venv")]
        for f in files:
            rel = f if rel_d == "." else f"{rel_d}/{f}"
            if is_code(rel): out.append(rel)
    return sorted(out)


def make_manifest(root=None):
    """Written before every published version: the SHA-256 of every program file git tracks."""
    root = root or ROOT
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split("\n")
    files = {rel: sha256(open(os.path.join(root, rel), "rb").read()) for rel in tracked if rel and is_code(rel)}
    json.dump({"files": files}, open(os.path.join(root, MANIFEST), "w"), indent=0, sort_keys=True)
    return len(files)


def verify(z, top, sha, cfg):
    """Refuse a download that isn't exactly the pinned version. Returns a list of checks that passed."""
    passed = []
    if sha and sha not in top and sha[:7] not in top:
        raise ValueError("update refused: the download is not the version that was asked for")
    passed.append("pinned version")
    files = {n[len(top):]: n for n in z.namelist() if n.startswith(top) and not n.endswith("/") and n[len(top):]}
    for n in files.values():
        if (z.getinfo(n).external_attr >> 16) & 0o170000 == 0o120000:  # a link pointing somewhere else
            raise ValueError(f"update refused: {n!r} is a link, not a file")
    if MANIFEST in files:
        listed = json.loads(z.read(files[MANIFEST]))["files"]
        code = {rel for rel in files if is_code(rel)}
        if set(listed) != code:
            raise ValueError("update refused: the file list doesn't match the version's SHA-256 list")
        for rel in code:
            if sha256(z.read(files[rel])) != listed[rel]:
                raise ValueError(f"update refused: {rel} doesn't match its SHA-256")
        passed.append("SHA-256 list")
    if sha and cfg:
        try:
            tree = json.loads(get(TREE.format(repo=cfg["repo"], sha=sha)))
        except Exception:
            tree = None  # GitHub's API can be busy: the SHA-256 list still checked every file
        if tree and not tree.get("truncated"):
            blobs = {t["path"]: t["sha"] for t in tree.get("tree", []) if t.get("type") == "blob"}
            for rel, n in files.items():
                if blobs.get(rel) != git_blob(z.read(n)):
                    raise ValueError(f"update refused: {rel} doesn't match GitHub's copy of that version")
            passed.append("GitHub file hashes")
    return passed


def backup(changed_by_update):
    """Keep a copy of the current version (program files only) so it can be put back."""
    if os.path.exists(BACKUP): shutil.rmtree(BACKUP)
    os.makedirs(BACKUP)
    for rel in code_files():
        dest = os.path.join(BACKUP, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, *rel.split("/")), dest)
    json.dump({"version": local_version(), "files": code_files(), "time": time.time()}, open(os.path.join(BACKUP, ".backup.json"), "w"))


def rollback(bad=True):
    """Put the previous version back. bad=True also remembers not to install the broken version again."""
    info_p = os.path.join(BACKUP, ".backup.json")
    if not os.path.exists(info_p): return {"ok": False, "error": "there is no previous version saved yet"}
    info, now = json.load(open(info_p)), local_version()
    man = os.path.join(ROOT, MANIFEST)
    added = set(json.load(open(man))["files"]) - set(info["files"]) if os.path.exists(man) else set()
    for rel in added:  # only files the update itself added (never anything of yours)
        p = os.path.join(ROOT, *rel.split("/"))
        if os.path.exists(p): os.remove(p)
    for rel in info["files"]:
        dest = os.path.join(ROOT, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(BACKUP, *rel.split("/")), dest)
    json.dump(info["version"], open(os.path.join(ROOT, ".version"), "w"))
    if bad and now.get("sha"): save_config(skip=now["sha"])
    return {"ok": True, "version": (info["version"].get("sha") or "?")[:7], "skipped": (now.get("sha") or "?")[:7]}


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


def apply_zip(data, root=ROOT, sha=None, cfg=None, keep_backup=False):
    """Copy the code from a GitHub zip into the project, skipping everything in KEEP. Returns changed paths."""
    z = zipfile.ZipFile(io.BytesIO(data))
    top = z.namelist()[0].split("/")[0] + "/"
    changed = []
    if sum(i.file_size for i in z.infolist()) > 200 * 1024 * 1024:
        raise ValueError("update refused: the download is far bigger than the program should be")
    for name in z.namelist():  # check every path first, so a bad download changes nothing at all
        rel = name[len(top):] if name.startswith(top) else name
        if rel.startswith("/") or ".." in rel.split("/") or ":" in rel or "\\" in rel:
            raise ValueError(f"update refused: unsafe file path in the download ({rel!r})")
    verify(z, top, sha, cfg)  # nothing is changed unless every check passes
    if keep_backup: backup(None)
    for name in z.namelist():
        if not name.startswith(top) or name.endswith("/"):
            continue
        rel = name[len(top):]
        if not rel or rel.split("/")[0] in KEEP or "__pycache__" in rel:
            continue
        parts = rel.split("/")
        if any(x in ("", ".", "..") or ":" in x or "\\" in x for x in parts):
            raise ValueError(f"update refused: unsafe file path in the download ({rel!r})")
        dest = os.path.join(root, *parts)
        real_root = os.path.realpath(root)
        if os.path.commonpath([real_root, os.path.realpath(dest)]) != real_root:  # never write outside the project folder
            raise ValueError(f"update refused: {rel!r} would be written outside the project")
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
    if mode(cfg) == "off":
        return {"available": False, "error": "updates are off (type /updates instant to turn them back on)"}
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
    if new["sha"] == cfg.get("skip"):
        return {"available": False, "version": (cur.get("sha") or "?")[:7], "error": "the newest version failed to start, so it was put back; waiting for a fixed one"}
    if not install:
        return {"available": True, "message": new["message"]}
    try:
        data = get(ZIP.format(repo=cfg["repo"], sha=new["sha"]), timeout=120)
        changed = apply_zip(data, sha=new["sha"], cfg=cfg, keep_backup=True)
    except Exception as e:
        return {"available": True, "error": f"download failed, kept the current version: {e}"}
    if "requirements.txt" in changed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(ROOT, "requirements.txt")])
    json.dump({**new, "installed": time.time()}, open(os.path.join(ROOT, ".version"), "w"))
    if not quiet:
        print(f"Updated: {new['message']} ({len(changed)} files)")
    return {"available": True, "installed": True, "message": new["message"], "changed": changed, "version": new["sha"][:7]}


def main():
    if "--rollback" in sys.argv:
        print(rollback()); return
    if "--manifest" in sys.argv:  # used when publishing a new version
        print(f"SHA-256 list written for {make_manifest()} files"); return
    r = check()
    if r.get("installed"): return
    print(r.get("error") or f"Game AI is up to date ({r.get('version', '?')}).")


if __name__ == "__main__":
    main()
