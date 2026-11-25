"""Export the promoted model for the browser / Chromebook (doc section 10).

    python -m gameai.export_onnx

Writes export/gameai.onnx (float), export/gameai.int8.onnx (8-bit, ~4x smaller) and
export/vocab.json, then checks the INT8 model gives the same recipes as PyTorch.
Browser use: ONNX Runtime Web with the WebGPU backend, WebAssembly as fallback.
"""
import json, os
import numpy as np
import torch
from .generate import load, decode
from .recipe import FIELD_NAMES
from .tokenizer import MAX_PROMPT


def ort_decode(sess, tok, prompt):
    x = np.array([tok.encode_prompt(prompt)], dtype=np.int64)
    out = {}
    for slot, f in enumerate(FIELD_NAMES):
        logits = sess.run(None, {"idx": x})[0][0, -1]
        ids = tok.field_ids[slot]
        t = ids[int(np.argmax(logits[ids]))]
        out[f] = tok.decode_value(slot, t)
        x = np.concatenate([x, [[t]]], 1)
    return out


def main(checkpoint="models/current.pt"):
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    model, tok, _ = load(checkpoint, "cpu")
    os.makedirs("export", exist_ok=True)
    dummy = torch.tensor([tok.encode_prompt("a fast cat in a dungeon")])
    seq = torch.export.Dim("seq", min=2, max=tok.max_len)
    torch.onnx.export(model, (dummy,), "export/gameai.onnx", input_names=["idx"], output_names=["logits"],
                      dynamic_shapes={"idx": {1: seq}}, opset_version=18, dynamo=True, external_data=False)
    quantize_dynamic("export/gameai.onnx", "export/gameai.int8.onnx", weight_type=QuantType.QInt8)
    json.dump({"itos": tok.itos, "field_names": FIELD_NAMES, "field_ids": tok.field_ids, "max_prompt": MAX_PROMPT,
               "pad": tok.pad, "bos": tok.bos, "sep": tok.sep}, open("export/vocab.json", "w"))

    rows = [json.loads(l) for l in open("data/test_frozen.jsonl")][:300]
    ref = decode(model, tok, [r["prompt"] for r in rows], "cpu", gate=0.0)
    sess = ort.InferenceSession("export/gameai.int8.onnx", providers=["CPUExecutionProvider"])
    agree = sum(ort_decode(sess, tok, r["prompt"]) == {f: ref[i][f][0] for f in FIELD_NAMES} for i, r in enumerate(rows))
    for f in ("gameai.onnx", "gameai.int8.onnx"):
        print(f"{f}: {os.path.getsize('export/' + f) / 1e6:.1f} MB")
    print(f"INT8 matches PyTorch on {agree}/{len(rows)} test prompts")


if __name__ == "__main__":
    main()
