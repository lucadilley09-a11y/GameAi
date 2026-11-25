"""The three permanent splits, for every collection.

    training      examples the AI may learn from
    validation    used for every decision: choosing versions, settings, mixes, weightings, release setups.
                  ANCHOR files never change, so scores stay comparable across every version, forever;
                  ROTATING files may be refreshed on a schedule so decisions don't slowly fit one fixed set.
    sealed        never used for training, tuning or feedback; encrypted, only totals come out (see sealed.py)

Collections that exist today map onto files the app already has; new collections live in data/splits/<name>/.

    python -m gameai.splits        show every collection and how big each split is
"""
import json, os

ROOT_DIR = "data/splits"

# anchor = permanent validation (about 30% or more of a collection's validation; never regenerated)
COLLECTIONS = {
    "game_requests": {"train": ["data/train.jsonl"], "anchor": ["data/test_frozen.jsonl", "data/test_typo.jsonl", "gameai/gold.jsonl"],
                      "rotating": ["data/val.jsonl"], "note": "generated requests; the anchor is the frozen, typo and hand-written gold tests"},
    "real_talk": {"train": ["data/teacher.jsonl"], "anchor": ["data/test_teacher.jsonl:500"], "rotating": ["data/test_teacher.jsonl:500:"],
                  "note": "requests the teacher wrote the way people talk; every 10th is kept for testing"},
    "yours": {"train": ["memory/corrections.jsonl"], "anchor": ["memory/corrections_test.jsonl"], "rotating": [],
              "note": "your corrections; every 5th is kept for testing"},
    # filled in later phases (each gets train / validation_anchor / validation_rotating files under data/splits/<name>/)
    "general": {"note": "general chat, instructions, other languages, reasoning (protects general ability)"},
    "intent": {"note": "people-written requests: slang, typos, ambiguity, multi-turn, edits to existing projects"},
    "code": {"note": "coding tasks in JavaScript/TypeScript, Python and Lua: data/coding (tasks_val.jsonl plus the hand-written "
                     "seed tasks are validation, tasks_train.jsonl is for self-teaching); multi-file and existing-project tasks too"},
    "ui": {"note": "interface tasks and screenshots"},
    "end_to_end": {"note": "practice and validation versions of the end-to-end tasks (the sealed ones live in sealed/)"},
}


def _files(c, part):
    spec = COLLECTIONS[c]
    if part in spec: return spec[part]
    name = {"train": "train.jsonl", "anchor": "validation_anchor.jsonl", "rotating": "validation_rotating.jsonl"}[part]
    return [os.path.join(ROOT_DIR, c, name)]


def read(path_spec):
    """'file.jsonl', 'file.jsonl:500' (first 500) or 'file.jsonl:500:' (from 500 on)"""
    import re
    path, n, tail = re.match(r"^(.*?)(?::(\d+)(:?))?$", path_spec).groups()
    if not os.path.exists(path): return []
    out = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if n is None: return out
    return out[int(n):] if tail else out[:int(n)]


def rows(collection, part):
    out = []
    for f in _files(collection, part): out += read(f)
    return out


def all_validation_texts():
    from .guard import text_of
    for c in COLLECTIONS:
        for part in ("anchor", "rotating"):
            if c in ("game_requests", "real_talk", "yours"): continue  # guard.TEST_FILES already covers these
            for r in rows(c, part): yield text_of(r)


def refresh_rotating(collection, new_rows):
    """Replace only the rotating part of a collection's validation. The anchor is never touched."""
    path = _files(collection, "rotating")[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in new_rows: f.write(json.dumps(r) + "\n")
    hist = os.path.join(ROOT_DIR, "refresh_log.jsonl")
    import time
    with open(hist, "a") as f: f.write(json.dumps({"collection": collection, "rows": len(new_rows), "time": time.time()}) + "\n")


def describe():
    from .sealed import summary
    sealed = summary()
    out = []
    for c, spec in COLLECTIONS.items():
        n = {p: len(rows(c, p)) for p in ("train", "anchor", "rotating")}
        s = sealed.get(c, {})
        out.append({"collection": c, **n, "sealed_active": s.get("active", 0), "sealed_reserve": s.get("reserve", 0), "note": spec["note"]})
    return out


def main():
    print(f"{'collection':14} {'train':>8} {'anchor':>8} {'rotating':>9} {'sealed':>7} {'reserve':>8}")
    for d in describe():
        print(f"{d['collection']:14} {d['train']:8} {d['anchor']:8} {d['rotating']:9} {d['sealed_active']:7} {d['sealed_reserve']:8}  {d['note']}")


if __name__ == "__main__":
    main()
