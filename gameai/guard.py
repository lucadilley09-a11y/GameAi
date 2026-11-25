"""Keeps test questions out of training.

Every test example (validation, frozen tests, your corrections kept for testing, the teacher's test share, and the
sealed sets) leaves a fingerprint here. Training data goes through filter_training(), which drops anything that is
the same request, or a near-copy of one (a reworded twin), as a test example.

    exact      the text, lower-cased with spacing and punctuation removed, hashed (SHA-256)
    near-copy  MinHash of word 3-grams; two texts count as near-copies when their estimated overlap
               (Jaccard similarity) is at least NEAR. Signatures can't be turned back into the text, so the
               sealed set's fingerprints can be used here without anyone reading it.

    python -m gameai.guard report     how much of today's training data is a near-copy of a test example
"""
import hashlib, json, os, re

import numpy as np

NEAR = 0.8          # overlap at which two requests count as the same question reworded
PERMS = 64          # MinHash size
BANDS, ROWS = 16, 4  # locality-sensitive hashing: finds candidate pairs fast; each is then checked properly
INDEX = "data/guard_index.npz"
_P = np.uint64((1 << 31) - 1)  # (a*h + b) mod p with a, h below 2^32 never overflows 64 bits
_rng = np.random.default_rng(1234)  # fixed: fingerprints must be the same on every run and every PC
_A = _rng.integers(1, int(_P), PERMS, dtype=np.uint64)
_B = _rng.integers(0, int(_P), PERMS, dtype=np.uint64)


def norm(text):
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def exact_key(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()[:32]


def shingles(text, n=3):
    w = norm(text).split()
    if len(w) < n: return {" ".join(w)}
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def signature(text):
    hs = np.array([int.from_bytes(hashlib.blake2b(x.encode(), digest_size=4).digest(), "little") for x in shingles(text)], dtype=np.uint64)
    return ((np.outer(hs, _A) + _B) % _P).min(axis=0)


def similarity(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))


def text_of(row):
    for k in ("prompt", "request", "text", "input"):
        if k in row: return row[k]
    if "messages" in row:  # a conversation: what the person asked is what makes it the same question
        return " ".join(m.get("content", "") for m in row["messages"] if m.get("role") == "user")
    return json.dumps(row, sort_keys=True)


class Index:
    """Fingerprints of every test example: exact keys plus MinHash signatures, bucketed for fast lookup."""

    def __init__(self):
        self.exact, self.sigs, self.buckets = set(), [], {}

    def add(self, text=None, sig=None, key=None):
        if key is None: key = exact_key(text)
        if sig is None: sig = signature(text)
        self.exact.add(key)
        i = len(self.sigs); self.sigs.append(sig)
        for b in range(BANDS):
            self.buckets.setdefault((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), []).append(i)

    def match(self, text):
        """-> (is_test_question, why)"""
        if exact_key(text) in self.exact: return True, "same request as a test example"
        sig = signature(text)
        seen = set()
        for b in range(BANDS):
            for i in self.buckets.get((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), ()):
                if i in seen: continue
                seen.add(i)
                if similarity(sig, self.sigs[i]) >= NEAR: return True, "a reworded copy of a test example"
        return False, ""

    def save(self, path=INDEX):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, exact=np.array(sorted(self.exact)), sigs=np.array(self.sigs, dtype=np.uint64).reshape(-1, PERMS))

    @classmethod
    def load(cls, path=INDEX):
        ix = cls()
        d = np.load(path)
        for k, s in zip(d["exact"], d["sigs"]): ix.add(sig=s, key=str(k))
        ix.exact |= set(map(str, d["exact"]))
        return ix


TEST_FILES = ["data/val.jsonl", "data/test_frozen.jsonl", "data/test_typo.jsonl", "data/test_teacher.jsonl",
              "memory/corrections_test.jsonl", "gameai/gold.jsonl"]


def test_texts():
    from .splits import all_validation_texts
    for p in TEST_FILES:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip(): yield text_of(json.loads(line))
    yield from all_validation_texts()


def build_index():
    """Fingerprint every test example on this PC, plus the sealed sets' fingerprints (stored without their text)."""
    ix = Index()
    for t in test_texts(): ix.add(t)
    from .sealed import fingerprints
    for key, sig in fingerprints(): ix.add(sig=sig, key=key)
    return ix


_cached = {}


def filter_training(rows, label="training", ix=None, log=print):
    """Drop every row that is a test question or a reworded copy of one. -> kept rows"""
    if ix is None:
        if "ix" not in _cached: _cached["ix"] = build_index()
        ix = _cached["ix"]
    kept, dropped = [], {"same request as a test example": 0, "a reworded copy of a test example": 0}
    for r in rows:
        hit, why = ix.match(text_of(r))
        if hit: dropped[why] += 1
        else: kept.append(r)
    n = sum(dropped.values())
    if n and log: log(f"guard: left out {n} of {len(rows)} {label} examples that match test questions "
                      f"({dropped['same request as a test example']} the same, {dropped['a reworded copy of a test example']} reworded)")
    return kept


def main():
    import sys, time
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        t0 = time.time(); ix = build_index()
        print(f"{len(ix.sigs)} test fingerprints in {time.time() - t0:.0f}s")
        rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(rows)
        t0 = time.time(); kept = filter_training(rows[:limit], ix=ix)
        print(f"kept {len(kept)} of {limit} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
