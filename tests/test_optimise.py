"""Tests for the Optimise phase pieces that can run without a GPU: answer-speed tuning's choice logic, the server's use
of measured settings, length-grouped batches, imatrix compression arguments and the compile fallback."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = []
def t(name, ok, info=None):
    ok = bool(ok); R.append(ok); print(("PASS " if ok else "FAIL ") + name)
    if not ok and info is not None: print("   ", str(info)[:1200])


def main():
    here = os.getcwd(); work = tempfile.mkdtemp(); os.chdir(work)
    try:
        from gameai.phase0 import runtime as Rt, tune as Tu
        gguf = os.path.join(work, "m.gguf"); open(gguf, "wb").write(b"x" * 100)
        # the server takes the measured settings, without repeating flags it was given
        os.makedirs("runtime", exist_ok=True)
        json.dump({Rt._model_id(gguf): {"flags": ["-fa", "on", "--spec-type", "ngram-mod"]}}, open(Rt.TUNING, "w"))
        Rt.exe = lambda name: "llama-server"
        s = Rt.Server(gguf, extra=["--lora", "a.gguf"])
        t("server adds the measured settings", s.args[-6:] == ["--lora", "a.gguf", "-fa", "on", "--spec-type", "ngram-mod"], s.args)
        s = Rt.Server(gguf, extra=["--spec-type", "draft-mtp"])
        t("an explicit setting wins over the measured one", s.args.count("--spec-type") == 1 and "draft-mtp" in s.args, s.args)
        s = Rt.Server(gguf, tuned=False, cache_prompt=False)
        t("the bake-off can turn tuning and caching off", "-fa" not in s.args and not s.cache_prompt)
        t("prompt caching is on by default", Rt.Server(gguf).cache_prompt)
        # tuning's choice: fake servers with known speeds and answers
        speeds = {"plain": 10, "flash attention": 8, "flash attention + 8-bit KV cache": 6, "n-gram speculation": 7,
                  "flash attention + n-gram speculation": 6.5, "flash attention + 8-bit KV cache + n-gram speculation": 5}
        class Fake:
            def __init__(self, gguf, port=0, extra=(), tuned=False, cache_prompt=True): self.flags = list(extra)
            def __enter__(self): return self
            def __exit__(self, *a): pass
        def fake_run(srv, items):
            name = next(n for n, f in Tu.CONFIGS if f == srv.flags)
            kv = "-ctk" in srv.flags
            return speeds[name], ["same"] * 10, {"recipes_fully_right": 10, "code_tasks_solved": 5 if kv else 6}
        Tu.R.Server, Tu.run, Tu.workload = Fake, fake_run, lambda: [None] * 10
        res = Tu.tune(gguf, log=lambda *a: None)
        t("tuning picks the fastest setting that keeps the answers (8-bit cache rejected: it solved fewer code tasks)",
          res["chosen"] == "flash attention + n-gram speculation" and res["flags"][:2] == ["-fa", "on"], res)
        speeds.update({k: 9.8 for k in speeds if k != "plain"})
        res = Tu.tune(gguf, log=lambda *a: None)
        t("a gain under 5% isn't adopted", res["chosen"] == "plain" and res["flags"] == [], res)
        # length-grouped batches
        from gameai.lora_train import grouped_batch
        import random
        rows = [{"n": n} for n in [5] * 50 + [500] * 50]
        rng = random.Random(1)
        b = grouped_batch(rows, [], 0.0, 8, rng, length=lambda r: r["n"])
        t("a batch is drawn from rows of similar length", len(b) == 8 and len({r["n"] for r in b}) == 1, [r["n"] for r in b])
        gen = [{"n": 5, "g": 1}] * 40 + [{"n": 500, "g": 1}] * 40
        share = sum(sum("g" in r for r in grouped_batch(rows, gen, 0.3, 10, rng, length=lambda r: r["n"])) for _ in range(300)) / 3000
        t("the general-data share is kept (about 30%)", 0.25 < share < 0.35, share)
        # imatrix compression arguments
        calls = []
        Rt.exe = lambda name: name
        Rt.subprocess.run = lambda cmd, **k: calls.append(cmd) or type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        Rt.quantize("f16.gguf", "out.gguf", imatrix="m.imatrix")
        t("compression can use an importance matrix", calls[-1] == ["llama-quantize", "--imatrix", "m.imatrix", "f16.gguf", "out.gguf", "Q4_K_M"], calls[-1])
        open("calib.txt", "w").write("hello " * 100)
        Rt.make_imatrix("f16.gguf", "calib.txt", "m.imatrix")
        t("the importance matrix is computed from calibration text", calls[-1][:5] == ["llama-imatrix", "-m", "f16.gguf", "-f", "calib.txt"], calls[-1])
        # compile: falls back cleanly when it can't work
        from gameai.train import maybe_compile
        import torch
        m = torch.nn.Linear(2, 2)
        out, how = maybe_compile(m, "auto", device="cpu")
        t("compile is skipped without a GPU, and the model still works", out is m and "skipped" in how, how)
        # storage cleaning
        import time as _t
        from gameai import storage as St, registry
        os.environ["HF_HUB_CACHE"] = os.path.join(work, "hub")
        os.makedirs("hub/models--someone--old-model/blobs", exist_ok=True); open("hub/models--someone--old-model/blobs/x", "wb").write(b"0" * 1000)
        os.makedirs("hub/models--Qwen--Qwen3.5-4B/blobs", exist_ok=True); open("hub/models--Qwen--Qwen3.5-4B/blobs/y", "wb").write(b"0" * 1000)
        os.makedirs("checkpoints", exist_ok=True); os.makedirs("models/versions", exist_ok=True); os.makedirs("logs", exist_ok=True)
        for n in ("run1.pt", "latest.pt"):
            open("checkpoints/" + n, "wb").write(b"0" * 500); os.utime("checkpoints/" + n, (_t.time() - 7200, _t.time() - 7200))
        for i in range(1, 16): open(f"models/versions/v{i:03d}.pt", "wb").write(b"0" * 10); open(f"models/versions/v{i:03d}.json", "w").write("{}")
        open("logs/big.log", "wb").write(b"x" * (St.BIG_LOG + 10 * 2**20))
        St._busy = lambda: False; St._training_pending = lambda: False
        os.makedirs("models/base/m/hf", exist_ok=True); open("models/base/m/hf/model.safetensors", "wb").write(b"0" * 2000)
        os.utime("models/base/m/hf/model.safetensors", (_t.time() - 5 * 86400, _t.time() - 5 * 86400))
        registry.register("core_hf", "base", "training copy", "models/base/m/hf", 0, "apache-2.0", "abc", "hf", repo="x/y", key="m", revision="abcdef")
        freed = St.clean(log=lambda *a: None)
        t("unused downloads go, the learning teacher's stays", not os.path.exists("hub/models--someone--old-model") and os.path.exists("hub/models--Qwen--Qwen3.5-4B"))
        t("old checkpoints go, the last tested one stays", not os.path.exists("checkpoints/run1.pt") and os.path.exists("checkpoints/latest.pt"))
        t("versions beyond the last 10 go", len([f for f in os.listdir("models/versions") if f.endswith(".pt")]) == 10 and os.path.exists("models/versions/v015.pt"))
        t("big logs keep their end", os.path.getsize("logs/big.log") < 6 * 2**20)
        t("the unused training copy goes, and the report knows", not os.path.exists("models/base/m/hf/model.safetensors") and freed > 2**20, freed)
        got = []
        import gameai.phase0.candidates as C
        C.by_key = lambda k: {"key": k}; C.record = lambda c: {"download_gb": 0.001, "revision": "latest", "repo": "x/y"}
        C.download = lambda rec, dest: got.append(rec["revision"]) or os.makedirs(dest, exist_ok=True) or open(os.path.join(dest, "model.safetensors"), "wb").write(b"1")
        import gameai.phase0.common as Co; Co.need_space = lambda *a, **k: None
        St.ensure_training_copy(log=lambda *a: None)
        t("it's downloaded again at exactly the same version when training needs it", got == ["abcdef"] and os.path.exists("models/base/m/hf/model.safetensors"), got)
        # sharing the PC
        from gameai import polite
        slept = []
        polite.time.sleep = lambda s: slept.append(s)
        polite.state = lambda: "using"
        g = polite.Gate(); g.checked = 0; g.last = polite.time.perf_counter() - 0.2; g.after_step()
        t("while you use the PC, training rests as long as it worked (half speed)", slept and 0.15 < slept[-1] < 0.3, slept)
        polite.state = lambda: "heavy"
        g.checked = 0; g.last = polite.time.perf_counter() - 0.1; g.after_step()
        t("with a game running, it mostly rests", slept[-1] > 0.4, slept)
        polite.state = lambda: "away"
        g.checked = 0; n = len(slept); g.after_step()
        t("when you're away, full speed", len(slept) == n)
    finally:
        os.chdir(here); shutil.rmtree(work, ignore_errors=True)
    print(f"{sum(R)}/{len(R)} passed")
    if not all(R): sys.exit(1)


main()
