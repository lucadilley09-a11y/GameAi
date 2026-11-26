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
        # clarifying questions: ask only when the answer would matter
        from gameai import clarify as Cl
        os.makedirs("memory", exist_ok=True)
        with open("memory/questions.jsonl", "w") as f:
            for _ in range(20): f.write(json.dumps({"field": "theme", "cared": False}) + "\n")
            for _ in range(10): f.write(json.dumps({"field": "goal", "cared": True}) + "\n")
        cares = Cl.care()
        t("what you care about is learned from your answers", cares["theme"] < 0.1 and cares["goal"] > 0.8, cares)
        rec = {f: "auto" for f in Cl.QUESTION}
        t("it asks about what you care about, never about what you always skip", Cl.choose_questions(rec, cares)[0] == "goal" and "theme" not in Cl.choose_questions(rec, cares))
        t("nothing to ask when the request says it all", Cl.choose_questions({f: "x" for f in Cl.QUESTION}, cares) == [])
        import gameai.guard as Gd
        class NoIx:
            def match(self, t): return (False, None)
        Gd.build_index = lambda: NoIx()
        res = Cl.generate(300, cares=cares)
        rows = [json.loads(l) for l in open("data/clarify/sft.jsonl")]
        asked = [r for r in rows if r["asked"]]
        t("simulated conversations: question, the user's reply, then the game", asked and all(r["messages"][2]["content"].endswith("surprise me") and r["messages"][-1]["content"].startswith("theme=") for r in asked), asked[:1])
        pairs = [json.loads(l) for l in open("data/clarify/pairs.jsonl")]
        t("preference pairs: asking vs answering, the better one chosen", pairs and all(p["chosen"] != p["rejected"] for p in pairs))
        # the lighter learning teacher answers several at once
        from gameai import teacher as Te
        class FakeSrv:
            def chat(self, c, **k): return ("<think>x</think> " + c[-1]["content"].upper(), {})
        out = Te.chat(None, FakeSrv(), [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], 20)
        t("the llama.cpp teacher's answers come back in order, without thinking text", out == ["A", "B"], out)
        speedups_1_to_3()
    finally:
        os.chdir(here); shutil.rmtree(work, ignore_errors=True)
    print(f"{sum(R)}/{len(R)} passed")
    if not all(R): sys.exit(1)


def speedups_1_to_3():
    """training speed-ups 1-3: growing, learning from the current AI, fastest batch pieces"""
    import torch
    from gameai.model import GameAI, make_config
    from gameai.grow import grow
    from gameai import train as Tr, all as All
    torch.manual_seed(0)
    for a, b, exact in [("tiny", "small", True), ("small", "large", True), ("base", "large", True), ("small", "base", False)]:
        m = GameAI(make_config(a, 120, 48)); m.eval()
        with torch.no_grad():
            for p in m.parameters(): p.add_(torch.randn_like(p) * 0.05)
        g, info = grow(m, b); g.eval()
        x = torch.randint(1, 120, (2, 16)); x[0, :3] = 0
        with torch.no_grad(): diff = float((m(x) - g(x)).abs().max())
        if exact:
            t(f"growing {a} -> {b} keeps the same answers", info["exact"] and diff < 0.01 and g.num_params() > m.num_params(), (info, diff))
        else:
            t(f"growing {a} -> {b} is an approximate start and says so", not info["exact"] and info["heads_dropped"] > 0, info)
    g, _ = grow(GameAI(make_config("tiny", 120, 48)), "small")
    xb = torch.randint(1, 120, (4, 12)); xb[:, 0] = 0
    opt = torch.optim.SGD(g.parameters(), lr=0.1)
    for _ in range(2):  # new layers' zero outputs pick up on the first step, then everything behind them learns
        opt.zero_grad(); Tr.recipe_loss(g, xb, xb, 5, 3).backward(); opt.step()
    opt.zero_grad(); Tr.recipe_loss(g, xb, xb, 5, 3).backward()
    dead = [n for n, p in g.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    t("every part of a grown AI can learn (nothing stuck at zero)", not dead, dead)
    teacher = GameAI(make_config("tiny", 120, 48)); student = GameAI(make_config("tiny", 120, 48))
    ce = Tr.recipe_loss(student, xb, xb, 5, 3)
    t("with the teacher's share at zero, the loss is the usual one", abs(Tr.distill_loss(student, teacher, xb, xb, 5, 3, 0.0).item() - ce.item()) < 1e-6)
    student.load_state_dict(teacher.state_dict())
    t("matching the teacher exactly adds nothing to the loss", abs(Tr.distill_loss(student, teacher, xb, xb, 5, 3, 1.0).item()) < 1e-4)
    ws = [Tr.distill_weight(s, 1000, 0.5) for s in (0, 300, 600, 900)]
    t("the teacher's share fades out by 60% of training", ws[0] == 0.5 and 0 < ws[1] < 0.5 and ws[2] == 0 and ws[3] == 0, ws)
    t("batch-piece tuning is skipped without a GPU", Tr.tune_micro(student, None, 256, 32, "cpu", 3) == 32)
    os.makedirs("models", exist_ok=True)
    m = GameAI(make_config("small", 120, 48))
    torch.save({"model": m.state_dict(), "config": m.config_dict(), "size": "small"}, "models/current.pt")
    t("a bigger size starts from the current smaller AI", All.smaller_current("large") == "models/current.pt" and All.smaller_current("base") == "models/current.pt")
    t("the same or a smaller size starts fresh", All.smaller_current("small") is None and All.smaller_current("tiny") is None)
    os.remove("models/current.pt")
    # gaming: detected before "away" (controllers don't count as input), pauses learning, blocks new work
    from gameai import polite as Po, autopilot as Ap
    saved = (Po.gaming, Po.idle_seconds, Po.CHECK_S, Po.OFF, Po.state)
    Po.state = Po.REAL_STATE  # earlier tests swap state() for stand-ins
    try:
        Po.OFF = False; Po.gaming = lambda: True; Po.idle_seconds = lambda: 10**6
        t("a game in front counts as gaming even with no keyboard or mouse input", Po.state() == "gaming")
        t("no new long work starts while you're gaming", not Ap.may_start({"answers": {}}, 10**6))
        n = [0]
        def flip():
            n[0] += 1; return n[0] < 3
        Po.gaming = flip; Po.CHECK_S = 0.01
        g = Po.Gate(); g.after_step()
        t("learning waits while you're gaming, then carries on by itself", n[0] >= 3 and g.state == "away", (n, g.state))
    finally:
        Po.gaming, Po.idle_seconds, Po.CHECK_S, Po.OFF, Po.state = saved


main()
