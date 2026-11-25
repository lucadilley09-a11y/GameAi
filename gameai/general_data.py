"""The general collection: everyday chat, instructions, other languages and reasoning, so the AI stays good at normal
conversation while it learns games. Imported once from permissively licensed datasets (licence checked first),
sampled small (lean storage), split three ways, and the raw downloads deleted.

    python -m gameai.general_data [--human-only]

    training / validation (30% permanent anchor + rotating)   from each dataset's own training part
    sealed                                                       from each dataset's own TEST part (written by them,
                                                                 not by Claude), locked away encrypted
"""
import glob, gzip, json, os, random, shutil, sys

from .phase0.candidates import ALLOWED
from .phase0.common import need_space, say

SOURCES = {
    "aya": {"repos": ["CohereLabs/aya_dataset", "CohereForAI/aya_dataset"], "human": True, "train": "data/train-*.parquet", "test": "data/test-*.parquet", "take": 15000},
    "oasst2": {"repos": ["OpenAssistant/oasst2"], "human": True, "trees": "2023-11-05_oasst2_ready.trees.jsonl.gz", "take": 10000},
    "gsm8k": {"repos": ["openai/gsm8k"], "human": True, "train": "main/train-*.parquet", "test": "main/test-*.parquet", "take": 7473},
    "smoltalk": {"repos": ["HuggingFaceTB/smol-smoltalk"], "human": False, "train": "data/train-00000-*.parquet", "test": "data/test-*.parquet", "take": 40000},
}
OUT = os.path.join("data", "splits", "general")
TMP = os.path.join("data", "tmp_download")
VAL_SHARE, VAL_MAX, ANCHOR, SEALED_N = 0.03, 600, 0.3, 200


def licence(repo):
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo)
    card = getattr(info, "card_data", None) or {}
    lic = (card.get("license") if hasattr(card, "get") else None) or next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), "")
    if isinstance(lic, list): lic = lic[0] if lic else ""
    return info, str(lic).lower()


def fetch(repo, pattern, rev):
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=[pattern], local_dir=os.path.join(TMP, repo.replace("/", "_")))
    return sorted(glob.glob(os.path.join(d, pattern)))


def parquet_rows(files):
    import pyarrow.parquet as pq
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=2000):
            yield from batch.to_pylist()


def to_chat(name, r):
    """-> {"messages": [...]} in one shape for every source"""
    if name == "aya": return {"messages": [{"role": "user", "content": r["inputs"]}, {"role": "assistant", "content": r["targets"]}], "language": r.get("language")}
    if name == "gsm8k": return {"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"].replace("####", "Answer:")}]}
    if name == "smoltalk": return {"messages": [m for m in r["messages"] if m["role"] in ("system", "user", "assistant")]}
    raise KeyError(name)


def oasst_chats(path):
    """each conversation tree -> one conversation following the best-ranked reply at every turn"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line); node, msgs = t["prompt"], []
            while node:
                msgs.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
                kids = [k for k in node.get("replies") or [] if not k.get("deleted")]
                node = min(kids, key=lambda k: k.get("rank") if k.get("rank") is not None else 99) if kids else None
            if len(msgs) >= 2:
                if msgs[-1]["role"] != "assistant": msgs = msgs[:-1]
                yield {"messages": msgs, "language": t["prompt"].get("lang"), "tree": t["message_tree_id"]}


def run(human_only=False, seed=7, log=say):
    from . import sealed
    need_space(3, "the general collection")
    rng = random.Random(seed)
    os.makedirs(OUT, exist_ok=True)
    train, val, sealed_rows, report = [], [], [], {}
    for name, src in SOURCES.items():
        if human_only and not src["human"]: continue
        info = lic = None
        for repo in src["repos"]:
            try: info, lic = licence(repo); break
            except Exception as e: last = e
        if info is None: report[name] = f"not found ({last})"; log(f"  {name}: not found"); continue
        if lic not in ALLOWED: report[name] = f"skipped: licence '{lic}'"; log(f"  {name}: skipped, licence '{lic}' isn't on the allowed list"); continue
        log(f"  {name}: licence {lic}, downloading a sample...")
        try:
            if "trees" in src:
                rows = list(oasst_chats(fetch(info.id, src["trees"], info.sha)[0]))
                by_tree = {}
                for r in rows: by_tree.setdefault(r["tree"], r)
                rows = list(by_tree.values()); rng.shuffle(rows)
                test, rows = rows[:SEALED_N], rows[SEALED_N:]
            else:
                rows = [to_chat(name, r) for r in parquet_rows(fetch(info.id, src["train"], info.sha))]
                test = [to_chat(name, r) for r in parquet_rows(fetch(info.id, src["test"], info.sha))]
                rng.shuffle(test); test = test[:SEALED_N]
            rng.shuffle(rows); rows = rows[:src["take"]]
            nv = min(VAL_MAX, max(20, int(len(rows) * VAL_SHARE)))
            for r in rows: r["source"] = name
            val += rows[:nv]; train += rows[nv:]
            for r in test:
                user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
                ans = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
                if user and ans: sealed_rows.append({"request": user, "answer": ans, "source": name})
            report[name] = {"licence": lic, "repo": info.id, "revision": info.sha, "train": len(rows) - nv, "validation": nv, "sealed": len(test)}
            log(f"    {report[name]}")
        finally:
            shutil.rmtree(TMP, ignore_errors=True)  # lean storage: only the sample stays
    rng.shuffle(val)
    k = int(len(val) * ANCHOR)
    for fn, rows in (("train.jsonl", train), ("validation_anchor.jsonl", val[:k]), ("validation_rotating.jsonl", val[k:])):
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if sealed_rows:
        tmp = os.path.join(OUT, "_sealed_import.jsonl")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in sealed_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sealed.add("general", tmp, kind="general"); os.remove(tmp)
    json.dump({"sources": report, "human_only": human_only}, open(os.path.join(OUT, "sources.json"), "w"), indent=1)
    return report


if __name__ == "__main__":
    print(run(human_only="--human-only" in sys.argv))
