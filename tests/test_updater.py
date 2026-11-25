"""Updater safety tests. No internet needed.  Run:  python -m tests.test_updater"""
import hashlib, io, json, os, shutil, sys, tempfile, zipfile

import gameai.update as U

SHA = "a" * 40


def make_zip(files, sha=SHA, manifest=True, tamper=None, extra=None, link=None):
    """A download like GitHub's: one top folder named after the version."""
    files = dict(files)
    if manifest:
        files[U.MANIFEST] = json.dumps({"files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items() if U.is_code(k)}})
    if tamper: files[tamper] = files[tamper] + " changed"
    if extra: files[extra] = "not on the list"
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in files.items(): z.writestr(f"GameAi-{sha}/{k}", v)
        if link:
            info = zipfile.ZipInfo(f"GameAi-{sha}/{link}"); info.external_attr = 0o120777 << 16
            z.writestr(info, "../../somewhere")
    return b.getvalue()


def project():
    root = tempfile.mkdtemp()
    U.ROOT, U.BACKUP = root, os.path.join(root, "backup", "previous")
    os.makedirs(os.path.join(root, "gameai")); os.makedirs(os.path.join(root, "data"))
    open(os.path.join(root, "gameai", "app.py"), "w").write("old app")
    open(os.path.join(root, "data", "mine.txt"), "w").write("your data")
    json.dump({"sha": "old"}, open(os.path.join(root, ".version"), "w"))
    json.dump({"repo": "x/GameAi"}, open(os.path.join(root, "update.json"), "w"))
    return root


def read(root, rel): return open(os.path.join(root, *rel.split("/"))).read()


NEW = {"gameai/app.py": "new app", "gameai/extra.py": "added file", "data/mine.txt": "should never be installed"}
results = []


def case(name, fn):
    try:
        fn(); results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, str(e)))


def refused(data, sha=SHA):
    root = project()
    try:
        U.apply_zip(data, root=root, sha=sha, keep_backup=True)
    except ValueError:
        assert read(root, "gameai/app.py") == "old app", "something changed before refusing"
        assert not os.path.exists(U.BACKUP), "a refused update still touched the backup"
        return True
    raise AssertionError("it installed a bad download")


def good_install():
    root = project()
    changed = U.apply_zip(make_zip(NEW), root=root, sha=SHA, keep_backup=True)
    assert read(root, "gameai/app.py") == "new app" and read(root, "gameai/extra.py") == "added file"
    assert read(root, "data/mine.txt") == "your data", "your data was overwritten"
    assert os.path.exists(os.path.join(U.BACKUP, "gameai", "app.py")), "no backup of the previous version"
    assert U.MANIFEST in changed


def rollback_works():
    root = project()
    U.apply_zip(make_zip(NEW), root=root, sha=SHA, keep_backup=True)
    json.dump({"sha": SHA}, open(os.path.join(root, ".version"), "w"))
    r = U.rollback()
    assert r["ok"], r
    assert read(root, "gameai/app.py") == "old app", "the old version didn't come back"
    assert not os.path.exists(os.path.join(root, "gameai", "extra.py")), "a file the update added was left behind"
    assert read(root, "data/mine.txt") == "your data"
    assert U.config().get("skip") == SHA, "the broken version isn't remembered"
    assert json.load(open(os.path.join(root, ".version")))["sha"] == "old"


def off_means_offline():
    project(); U.save_config(mode="off")
    def no_internet(*a, **k): raise AssertionError("it went online while updates are off")
    U.get, U.head_sha = no_internet, no_internet
    r = U.check()
    assert not r.get("installed") and "off" in r.get("error", "")


def skip_bad_version():
    project(); U.save_config(skip=SHA)
    U.latest = lambda cfg: {"sha": SHA, "message": "broken", "date": ""}
    U.get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("it downloaded the version that already failed"))
    r = U.check()
    assert not r.get("installed")


case("installs a correct download, keeps your data, keeps a backup", good_install)
case("refuses a file that doesn't match its SHA-256", lambda: refused(make_zip(NEW, tamper="gameai/app.py")))
case("refuses a file missing from the SHA-256 list", lambda: refused(make_zip(NEW, extra="gameai/sneaky.py")))
case("refuses a download of a different version", lambda: refused(make_zip(NEW, sha="b" * 40)))
case("refuses links", lambda: refused(make_zip(NEW, link="gameai/link.py")))
for name, bad in [("parent folder", "gameai/../../evil.py"), ("absolute path", "/etc/evil"), ("drive letter", "C:/evil.py"), ("backslash", "gameai\\..\\evil.py")]:
    case(f"refuses an unsafe path ({name})", lambda bad=bad: refused(make_zip({**NEW, bad: "x"}, manifest=False)))
def github_hashes(match):
    root = project(); data = make_zip(NEW)
    z = zipfile.ZipFile(io.BytesIO(data)); top = f"GameAi-{SHA}/"
    tree = [{"path": n[len(top):], "type": "blob", "sha": U.git_blob(z.read(n)) if match else "0" * 40} for n in z.namelist()]
    real_get = U.get; U.get = lambda *a, **k: json.dumps({"tree": tree, "truncated": False}).encode()
    try:
        if match:
            U.apply_zip(data, root=root, sha=SHA, cfg={"repo": "x/GameAi"})
            assert read(root, "gameai/app.py") == "new app"
        else:
            try: U.apply_zip(data, root=root, sha=SHA, cfg={"repo": "x/GameAi"}); raise AssertionError("installed files GitHub doesn't have")
            except ValueError: assert read(root, "gameai/app.py") == "old app"
    finally:
        U.get = real_get


case("checks every file against GitHub's hashes for that version", lambda: github_hashes(True))
case("refuses files that differ from GitHub's copy", lambda: github_hashes(False))
case("puts the previous version back and remembers not to reinstall the broken one", rollback_works)
case("mode off never goes online", off_means_offline)
case("won't reinstall a version that failed", skip_bad_version)

for name, ok, why in results:
    print(("PASS " if ok else "FAIL ") + name + (f": {why}" if why else ""))
print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
