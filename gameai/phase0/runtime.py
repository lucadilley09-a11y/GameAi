"""llama.cpp: the offline runtime the AI will use. Downloaded once (its official prebuilt release), then it works
without internet. Here: getting it, turning Hugging Face weights into its format (GGUF), compressing them (4-bit),
and running its server on this PC only (127.0.0.1).
"""
import glob, io, json, os, re, shutil, subprocess, sys, time, urllib.request, zipfile

DIR = os.path.join("runtime", "llama.cpp")
REL = "https://github.com/ggml-org/llama.cpp/releases"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "game-ai"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl()


def exe(name):
    """path of a llama.cpp program, or None"""
    hits = glob.glob(os.path.join(DIR, "**", name + (".exe" if os.name == "nt" else "")), recursive=True)
    if not hits and os.environ.get("GAMEAI_LLAMA_BIN"):  # a llama.cpp you built yourself
        p = os.path.join(os.environ["GAMEAI_LLAMA_BIN"], name + (".exe" if os.name == "nt" else ""))
        hits = [p] if os.path.exists(p) else []
    return hits[0] if hits else None


def convert_script():
    hits = glob.glob(os.path.join(DIR, "**", "convert_hf_to_gguf.py"), recursive=True)
    if not hits and os.environ.get("GAMEAI_LLAMA_SRC"):
        p = os.path.join(os.environ["GAMEAI_LLAMA_SRC"], "convert_hf_to_gguf.py"); hits = [p] if os.path.exists(p) else []
    return hits[0] if hits else None


def install(log=print):
    """Download llama.cpp's official prebuilt release for this PC (Windows + NVIDIA: the CUDA build) and its source
    (only for the converter script). About 150-300 MB."""
    if exe("llama-server") and exe("llama-quantize") and convert_script():
        return json.load(open(os.path.join(DIR, "release.json"))) if os.path.exists(os.path.join(DIR, "release.json")) else {"tag": "local"}
    _, final = _get(REL + "/latest")
    tag = final.rstrip("/").rsplit("/", 1)[-1]
    html = _get(f"{REL}/expanded_assets/{tag}")[0].decode("utf-8", "replace")
    assets = sorted(set(re.findall(r'href="(/ggml-org/llama\.cpp/releases/download/[^"]+\.zip)"', html)))
    names = [a.rsplit("/", 1)[-1] for a in assets]
    if os.name == "nt":
        cuda = [n for n in names if re.match(r"llama-.*-bin-win-cuda-12\.\d+-x64\.zip$", n)] or [n for n in names if re.match(r"llama-.*-bin-win-cuda-.*-x64\.zip$", n)]
        pick = sorted(cuda)[-1:] if cuda else [n for n in names if re.match(r"llama-.*-bin-win-(vulkan|cpu)-x64\.zip$", n)][:1]
        if pick and "cuda" in pick[0]:
            ver = re.search(r"cuda-([\d.]+)", pick[0]).group(1)
            pick += [n for n in names if n.startswith("cudart-") and f"cuda-{ver}" in n and "win" in n][:1]
    else:
        pick = [n for n in names if re.match(r"llama-.*-bin-ubuntu-x64\.zip$", n)][:1]
    if not pick: raise RuntimeError(f"couldn't find a llama.cpp build for this PC in release {tag}")
    from .common import need_space
    need_space(1.0, "llama.cpp")
    os.makedirs(DIR, exist_ok=True)
    for n in pick:
        log(f"  downloading {n}...")
        data, _ = _get(f"{REL}/download/{tag}/{n}", timeout=600)
        zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(DIR, "bin"))
    server = exe("llama-server")
    if server:  # the GPU runtime files (cudart) must sit next to llama-server.exe
        for dll in glob.glob(os.path.join(DIR, "bin", "**", "*.dll"), recursive=True):
            if os.path.dirname(os.path.abspath(dll)) != os.path.dirname(os.path.abspath(server)):
                shutil.copy2(dll, os.path.dirname(server))
    log("  downloading the converter (llama.cpp source)...")
    data, _ = _get(f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip", timeout=600)
    z = zipfile.ZipFile(io.BytesIO(data)); top = z.namelist()[0].split("/")[0] + "/"
    keep = ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "gguf-py/", "conversion/", "LICENSE", "requirements/")
    for n in z.namelist():
        rel = n[len(top):]
        if rel and any(rel == k or rel.startswith(k) for k in keep) and not n.endswith("/"):
            dest = os.path.join(DIR, "src", *rel.split("/")); os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "wb").write(z.read(n))
    if os.name != "nt":
        for f in glob.glob(os.path.join(DIR, "bin", "**", "llama-*"), recursive=True): os.chmod(f, 0o755)
    rel = {"tag": tag, "files": pick, "installed": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(rel, open(os.path.join(DIR, "release.json"), "w"), indent=1)
    return rel


def to_gguf(hf_dir, out_dir, log=print, vision=False):
    """Hugging Face weights -> one 16-bit GGUF file (+ the vision part as its own file, if the model has one)."""
    script = convert_script()
    if not script: raise RuntimeError("the llama.cpp converter isn't installed")
    src = os.path.dirname(script)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(src, "gguf-py"), src, os.environ.get("PYTHONPATH", "")]), NO_LOCAL_GGUF="")
    os.makedirs(out_dir, exist_ok=True)
    f16 = os.path.join(out_dir, "model-f16.gguf")
    r = subprocess.run([sys.executable, script, hf_dir, "--outfile", f16, "--outtype", "f16"], capture_output=True, text=True, env=env, creationflags=NO_WINDOW)
    if r.returncode: raise RuntimeError("converting failed: " + (r.stderr or r.stdout)[-800:])
    mm = None
    if vision:
        mm = os.path.join(out_dir, "mmproj-f16.gguf")
        r = subprocess.run([sys.executable, script, hf_dir, "--outfile", mm, "--outtype", "f16", "--mmproj"], capture_output=True, text=True, env=env, creationflags=NO_WINDOW)
        if r.returncode or not os.path.exists(mm):
            log("  no separate vision file could be made: " + (r.stderr or r.stdout)[-300:].strip().replace("\n", " ")); mm = None
    return f16, mm


def lora_to_gguf(adapter_dir, hf_dir, out):
    """A trained adapter (PEFT) -> llama.cpp's adapter format, loaded next to the compressed model with --lora."""
    script = convert_script()
    lora = os.path.join(os.path.dirname(script), "convert_lora_to_gguf.py") if script else None
    if not lora or not os.path.exists(lora): raise RuntimeError("the llama.cpp adapter converter isn't installed")
    src = os.path.dirname(lora)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(src, "gguf-py"), src, os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, lora, adapter_dir, "--base", hf_dir, "--outfile", out, "--outtype", "f16"], capture_output=True, text=True, env=env, creationflags=NO_WINDOW)
    if r.returncode or not os.path.exists(out): raise RuntimeError("converting the adapter failed: " + (r.stderr or r.stdout)[-600:])
    return out


def make_imatrix(f16, calib_txt, out, chunks=100):
    """an importance matrix: which weights matter most, measured on text like the AI's real work, so compression keeps
    those more precise (Optimise phase: lower error at the same file size). Takes a few minutes on the GPU."""
    q = exe("llama-imatrix")
    if not q: raise RuntimeError("llama-imatrix isn't installed")
    r = subprocess.run([q, "-m", f16, "-f", calib_txt, "-o", out, "--chunks", str(chunks), "-ngl", "99"],
                       capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode: raise RuntimeError("measuring the importance matrix failed: " + (r.stderr or r.stdout)[-800:])
    return out


def calibration_text(path, max_chars=600_000):
    """text for the importance matrix: the AI's own kinds of work from the TRAINING splits only (never validation or
    sealed sets): game requests, checked code, everyday conversation"""
    import random
    from .. import splits
    rng, parts = random.Random(0), []
    for coll in ("game_requests", "general", "real_talk"):
        try: rows = splits.rows(coll, "train")
        except Exception: rows = []
        rng.shuffle(rows)
        for r in rows[:1500]:
            if r.get("messages"): parts.append("\n".join(m.get("content", "") for m in r["messages"]))
            else: parts.append(" ".join(str(r.get(k, "")) for k in ("prompt", "text", "answer") if r.get(k)))
    first = os.path.join("data", "coding", "first_pass.jsonl")
    if os.path.exists(first):
        for l in list(open(first, encoding="utf-8"))[:800]:
            d = json.loads(l); parts.append(d["prompt"][-1500:] + "\n" + d["answer"])
    rng.shuffle(parts)
    text = "\n\n".join(p for p in parts if p.strip())[:max_chars]
    if len(text) < 20_000:  # too little of our own text yet: GameKit's code and docs fill the gap
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f in ("kit/gamekit.js", "kit/gamekit.d.ts"):
            text += "\n\n" + open(os.path.join(here, *f.split("/")), encoding="utf-8").read()
    open(path, "w", encoding="utf-8").write(text)
    return path


def quantize(f16, out, kind="Q4_K_M", imatrix=None):
    q = exe("llama-quantize")
    if not q: raise RuntimeError("llama-quantize isn't installed")
    r = subprocess.run([q] + (["--imatrix", imatrix] if imatrix else []) + [f16, out, kind], capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode: raise RuntimeError("compressing failed: " + (r.stderr or r.stdout)[-800:])
    return out


TUNING = os.path.join("runtime", "tuning.json")


def tuned_flags(gguf):
    """the speed settings measured best for this model file (Optimise phase), or none if it hasn't been measured"""
    try: t = json.load(open(TUNING))
    except (OSError, ValueError): return []
    e = t.get(_model_id(gguf))
    return list(e["flags"]) if e else []


def _model_id(gguf):
    return f"{os.path.basename(gguf)}:{os.path.getsize(gguf) if os.path.exists(gguf) else 0}"


class Server:
    """llama-server on 127.0.0.1 only. Use as: with Server(gguf) as s: s.chat(...)
    tuned: add the speed settings measured for this model (tune()); cache_prompt: reuse the processed start of the
    prompt between requests (the same instructions aren't processed again). The bake-off turns both off so every
    candidate is measured the same plain way."""

    def __init__(self, gguf, mmproj=None, ctx=8192, port=8790, gpu_layers=99, extra=(), tuned=True, cache_prompt=True):
        extra = list(extra)
        if tuned:
            have = {x for x in extra if str(x).startswith("-")}
            for i, f in enumerate(t := tuned_flags(gguf)):
                if str(f).startswith("-") and f not in have:
                    extra.append(f)
                    if i + 1 < len(t) and not str(t[i + 1]).startswith("-"): extra.append(t[i + 1])
        self.cache_prompt = cache_prompt
        self.args = [exe("llama-server"), "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx), "-ngl", str(gpu_layers), "--jinja", "-np", "1", *extra]
        if mmproj: self.args += ["--mmproj", mmproj]
        if not self.args[0]: raise RuntimeError("llama-server isn't installed")
        self.url, self.p, self.log = f"http://127.0.0.1:{port}", None, os.path.join("logs", "llama-server.log")

    def __enter__(self):
        os.makedirs("logs", exist_ok=True)
        low = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)  # your programs come first
        self.p = subprocess.Popen(self.args, stdout=open(self.log, "w"), stderr=subprocess.STDOUT, creationflags=NO_WINDOW | low)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.p.poll() is not None:
                raise RuntimeError("llama-server stopped while starting: " + open(self.log, errors="replace").read()[-600:])
            try:
                if json.loads(_get(self.url + "/health", 3)[0]).get("status") == "ok": return self
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server didn't start within 10 minutes")

    def __exit__(self, *a):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try: self.p.wait(20)
            except Exception: self.p.kill()

    def chat(self, messages, max_tokens=256, grammar=None, thinking=None, temperature=0.0, timeout=600):
        """-> (text, info) where info has tokens per second, time to the first token and the token counts"""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "cache_prompt": self.cache_prompt}
        if grammar: body["grammar"] = grammar
        if thinking is not None: body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        t = time.perf_counter()
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        wall = time.perf_counter() - t
        msg = d["choices"][0]["message"]
        tm, us = d.get("timings") or {}, d.get("usage") or {}
        info = {"wall_s": round(wall, 3), "tokens_out": us.get("completion_tokens") or tm.get("predicted_n"),
                "tokens_in": us.get("prompt_tokens") or tm.get("prompt_n"),
                "gen_tok_per_s": round(tm["predicted_per_second"], 1) if tm.get("predicted_per_second") else None,
                "prompt_tok_per_s": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
                "first_token_s": round(tm["prompt_ms"] / 1000, 3) if tm.get("prompt_ms") else None,
                "thinking_chars": len(msg.get("reasoning_content") or "")}
        return (msg.get("content") or ""), info
