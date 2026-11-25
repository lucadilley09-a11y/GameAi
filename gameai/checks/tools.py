"""Installing and finding the checker programs. Everything goes in runtime/tools (kept by the updater) and is
downloaded once; after that the checkers work offline.

    JavaScript tools (about 62 MB): TypeScript, ESLint + typescript-eslint, Prettier, StyLua (WebAssembly)
    Node.js (about 80 MB, only if this PC doesn't already have Node 20 or newer)
    ruff (about 25 MB, a Python package)
Versions are pinned to the ones that were tested.
"""
import glob, io, json, os, re, shutil, subprocess, sys, urllib.request, zipfile

DIR = os.path.join("runtime", "tools")
NODE_DIR = os.path.join(DIR, "node")          # the npm packages
NODE_DIST = os.path.join(DIR, "node-dist")    # a private Node.js, only when the PC has none
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
JS_PACKAGES = {
    "typescript": "6.0.3", "eslint": "10.15.0", "@eslint/js": "10.0.1", "typescript-eslint": "8.74.0",
    "globals": "17.14.0", "prettier": "3.10.1", "@johnnymorganz/stylua": "2.6.0",
}
RUFF = "0.16.9"
MIN_NODE = 20


def _ver(exe):
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW).stdout
        m = re.search(r"v?(\d+)\.", out); return int(m.group(1)) if m else 0
    except Exception:
        return 0


def node():
    """path of a Node.js that's new enough, or None (our own copy first, then the PC's)"""
    own = glob.glob(os.path.join(NODE_DIST, "**", "node.exe" if os.name == "nt" else "node"), recursive=True)
    for exe in own + [shutil.which("node")]:
        if exe and _ver(exe) >= MIN_NODE: return os.path.abspath(exe)
    return None


def node_modules():
    nm = os.path.join(NODE_DIR, "node_modules")
    return os.path.abspath(nm) if all(os.path.exists(os.path.join(nm, *p.split("/"))) for p in JS_PACKAGES) else None


def ruff():
    for exe in (shutil.which("ruff"), os.path.join(os.path.dirname(sys.executable), "Scripts" if os.name == "nt" else "", "ruff" + (".exe" if os.name == "nt" else ""))):
        if exe and os.path.exists(exe): return exe
    return None


def _get(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "game-ai"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()


def install_node(log=print):
    """A private Node.js (the current LTS) from nodejs.org, only for these tools. Windows and Linux x64."""
    index = json.loads(_get("https://nodejs.org/dist/index.json", 60))
    rel = next(r for r in index if r.get("lts") and int(r["version"][1:].split(".")[0]) >= MIN_NODE)
    v = rel["version"]
    if os.name == "nt":
        name = f"node-{v}-win-x64.zip"
        log(f"  downloading Node.js {v}...")
        z = zipfile.ZipFile(io.BytesIO(_get(f"https://nodejs.org/dist/{v}/{name}")))
        top = name[:-4] + "/"
        for n in z.namelist():  # node.exe, npm, and the licence; nothing else is needed
            rel_ = n[len(top):]
            if n.endswith("/") or not (rel_ in ("node.exe", "LICENSE", "npm.cmd") or rel_.startswith("node_modules/npm/")): continue
            dest = os.path.join(NODE_DIST, *rel_.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "wb").write(z.read(n))
    else:
        import tarfile
        name = f"node-{v}-linux-x64.tar.xz"
        log(f"  downloading Node.js {v}...")
        tarfile.open(fileobj=io.BytesIO(_get(f"https://nodejs.org/dist/{v}/{name}")), mode="r:xz").extractall(NODE_DIST, filter="data")
    return node()


def _npm(node_exe):
    """npm next to this node (our private copy has its npm-cli.js), else the PC's npm"""
    cli = glob.glob(os.path.join(os.path.dirname(node_exe), "**", "npm", "bin", "npm-cli.js"), recursive=True) + \
          glob.glob(os.path.join(os.path.dirname(os.path.dirname(node_exe)), "lib", "node_modules", "npm", "bin", "npm-cli.js"))
    if cli: return [node_exe, cli[0]]
    npm = shutil.which("npm")
    if not npm: raise RuntimeError("npm wasn't found next to Node.js")
    return [npm]


def install(log=print):
    """Install what's missing; safe to run again. -> status dict"""
    from ..phase0.common import need_space
    if not node_modules() or not node():
        need_space(0.5, "the code checkers")
    exe = node()
    if not exe:
        log("Node.js 20 or newer wasn't found on this PC, so a private copy goes in runtime/tools.")
        exe = install_node(log)
        if not exe: raise RuntimeError("couldn't set up Node.js")
    if not node_modules():
        os.makedirs(NODE_DIR, exist_ok=True)
        pkg = os.path.join(NODE_DIR, "package.json")
        json.dump({"name": "gameai-tools", "private": True, "description": "the code checkers' packages"}, open(pkg, "w"), indent=1)
        log("  installing TypeScript, ESLint, Prettier and StyLua (about 62 MB)...")
        cmd = _npm(exe) + ["install", "--no-audit", "--no-fund", "--save-exact", "--prefix", os.path.abspath(NODE_DIR)] + \
              [f"{k}@{v}" for k, v in JS_PACKAGES.items()]
        env = dict(os.environ, PATH=os.path.dirname(exe) + os.pathsep + os.environ.get("PATH", ""))
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, creationflags=NO_WINDOW)
        if p.returncode != 0 or not node_modules(): raise RuntimeError("installing the JavaScript tools failed:\n" + (p.stderr or p.stdout)[-800:])
    if not ruff():
        log("  installing ruff (Python formatter and linter)...")
        p = subprocess.run([sys.executable, "-m", "pip", "install", f"ruff=={RUFF}"], capture_output=True, text=True, creationflags=NO_WINDOW)
        if p.returncode != 0: raise RuntimeError("installing ruff failed:\n" + p.stderr[-800:])
    from . import status
    return status()
