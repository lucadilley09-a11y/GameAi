"""Export the current AI for the browser terminal (Chromebook).

    python -m gameai.export_web

Writes ONE file: for-chromebook/gameai-model.json. Put it on your Chromebook, open the
Game AI Terminal, and drag the file onto it (or type /load). It is remembered after that.
Also writes export/web/ (index.html, model.json, weights.json) for publishing the page."""
import base64, json, os
import numpy as np
from .generate import load
from .recipe import FIELD_NAMES, OUT_FIELDS, CONSTRAINTS
from .tokenizer import MAX_PROMPT
from .datagen import SYN

HERE = os.path.dirname(__file__)


def export_model(checkpoint="models/current.pt"):
    """-> (manifest dict, raw float32 bytes) for the JavaScript model."""
    model, tok, ck = load(checkpoint, "cpu")
    tensors, parts, off = [], [], 0
    for name, t in model.state_dict().items():
        a = t.float().numpy().ravel().astype("<f4")
        tensors.append({"name": name, "shape": list(t.shape), "offset": off, "size": a.size})
        parts.append(a); off += a.size
    manifest = {"config": model.config_dict(), "run": ck.get("run"), "params": model.num_params(), "tensors": tensors,
                "tokenizer": {"itos": tok.itos, "field_names": OUT_FIELDS, "recipe_fields": FIELD_NAMES,
                              "constraints": {k: [d, t] for k, (d, t) in CONSTRAINTS.items()}, "field_ids": tok.field_ids, "pad": tok.pad,
                              "bos": tok.bos, "sep": tok.sep, "unk": tok.unk, "max_prompt": MAX_PROMPT},
                "synonyms": SYN}
    return manifest, np.concatenate(parts).tobytes()


def single_file(checkpoint="models/current.pt"):
    m, raw = export_model(checkpoint)
    return json.dumps({**m, "b64": base64.b64encode(raw).decode()})


def build_page(builtin=None, pc=None):
    play = open(os.path.join(HERE, "play_template.html"), encoding="utf-8").read()
    engine = play[play.index("// ---------- engine ----------"):play.index("// ---------- prompt -> recipe -> game ----------")]
    engine = engine.replace("Describe a game and press New game", "Type a game idea in the terminal").replace("Training the AI…", "Loading the AI…")
    page = open(os.path.join(HERE, "web_template.html"), encoding="utf-8").read()
    page = page.replace("/*WEB_AI*/", open(os.path.join(HERE, "web_ai.js"), encoding="utf-8").read()).replace("/*ENGINE*/", engine)
    if builtin:  # one-file Chromebook app: the AI and the game template travel inside the page
        tpl = json.dumps(play).replace("</", "<\\/")
        page = page.replace("/*BUILTIN*/", "window.GAMEAI_BUILTIN=" + builtin + ";window.GAMEAI_GAME_TPL=" + tpl + ";")
    if pc:  # the Chromebook file knows where your PC is, so it can use it when it's on
        page = page.replace("<script>/*BUILTIN*/", "<script>window.GAMEAI_PC=" + json.dumps(pc) + ";/*BUILTIN*/", 1) if "/*BUILTIN*/" in page else page.replace("<script>window.GAMEAI_BUILTIN", "<script>window.GAMEAI_PC=" + json.dumps(pc) + ";window.GAMEAI_BUILTIN", 1)
    return page


def main(checkpoint="models/current.pt", out="export/web", pc=None):
    os.makedirs(out, exist_ok=True); os.makedirs("for-chromebook", exist_ok=True)
    m, raw = export_model(checkpoint)
    json.dump(m, open(os.path.join(out, "model.json"), "w"))
    json.dump({"b64": base64.b64encode(raw).decode()}, open(os.path.join(out, "weights.json"), "w"))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(build_page())
    single = json.dumps({**m, "b64": base64.b64encode(raw).decode()})
    open("for-chromebook/gameai-model.json", "w").write(single)
    open("for-chromebook/GameAI.html", "w", encoding="utf-8").write(build_page(single, pc))
    print(f"Chromebook app ready: {os.path.abspath('for-chromebook/GameAI.html')}  ({len(raw) * 1.34 / 1e6:.1f} MB)")
    print("Put it on your Chromebook (Google Drive or USB) and open it in Chrome. Your AI is built in.")


if __name__ == "__main__":
    main()
