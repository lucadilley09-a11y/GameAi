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


def export_model(checkpoint="models/current.pt", compact=True, weights=True):
    """-> (manifest dict, raw bytes) for the JavaScript model.
    compact: big weight tables are stored as 1 byte per number plus one scale per row ("q8"), about 4x smaller;
    the browser turns them back into normal numbers when it loads them. Small tensors stay exact."""
    model, tok, ck = load(checkpoint, "cpu")
    tensors, f32, i8 = [], [], []
    n32 = n8 = 0
    for name, t in (model.state_dict().items() if weights else []):
        a = t.float().numpy()
        if compact and a.ndim == 2 and a.size >= 4096:
            scale = np.maximum(np.abs(a).max(1), 1e-12) / 127.0
            q = np.clip(np.round(a / scale[:, None]), -127, 127).astype(np.int8).ravel()
            tensors.append({"name": name, "shape": list(t.shape), "q8": n8, "scale": n32, "size": int(a.size)})
            i8.append(q); f32.append(scale.astype("<f4")); n8 += q.size; n32 += scale.size
        else:
            a = a.ravel().astype("<f4")
            tensors.append({"name": name, "shape": list(t.shape), "offset": n32, "size": int(a.size)})
            f32.append(a); n32 += a.size
    raw = (np.concatenate(f32).astype("<f4").tobytes() if f32 else b"") + (np.concatenate(i8).tobytes() if i8 else b"")
    manifest = {"format": "q8" if i8 else "f32", "f32_count": n32, "config": model.config_dict(), "run": ck.get("run"), "params": model.num_params(), "tensors": tensors,
                "tokenizer": {"itos": tok.itos, "field_names": OUT_FIELDS, "recipe_fields": FIELD_NAMES,
                              "constraints": {k: [d, t] for k, (d, t) in CONSTRAINTS.items()}, "field_ids": tok.field_ids, "pad": tok.pad,
                              "bos": tok.bos, "sep": tok.sep, "unk": tok.unk, "max_prompt": MAX_PROMPT},
                "synonyms": SYN}
    return manifest, raw


def single_file(checkpoint="models/current.pt", compact=True):
    m, raw = export_model(checkpoint, compact)
    return json.dumps({**m, "b64": base64.b64encode(raw).decode()})


def build_page(builtin=None, pc=None):
    play = open(os.path.join(HERE, "play_template.html"), encoding="utf-8").read()
    engine = play[play.index("// ---------- engine ----------"):play.index("// ---------- prompt -> recipe -> game ----------")]
    engine = engine.replace("Describe a game and press New game", "Type a game idea in the terminal").replace("Training the AI…", "Loading the AI…")
    page = open(os.path.join(HERE, "web_template.html"), encoding="utf-8").read()
    page = page.replace("/*WEB_AI*/", open(os.path.join(HERE, "web_ai.js"), encoding="utf-8").read()).replace("/*ENGINE*/", engine + "\nwindow.GAMEAI_ENGINE_SRC=" + json.dumps(engine).replace("</", "<\\/") + ";")
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
