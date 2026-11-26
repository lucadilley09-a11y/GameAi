"""Compressing a model to 4 bits the best way (Optimise phase, approved 2026-11-25): plain Q4_K_M and Q4_K_M with an
importance matrix measured on the AI's own kind of text are both made and scored on validation (game requests and code
tasks); the better one is kept and the other deleted. Same file size either way, so no storage cost.
"""
import os

from . import runtime as R


def score(gguf, extra=(), log=print):
    from . import tasks as T
    from ..coding import tasks as CT, prompts as CP
    from ..coding.verify import verify
    items = T.recipe_items()[:60]
    with R.Server(gguf, extra=list(extra), tuned=False, port=8796) as srv:
        rec = T.test_recipes(srv.chat, items, thinking=False)["fully_right"]
        code = sum(verify(t, srv.chat(CP.messages(t), max_tokens=700, thinking=False, temperature=0.0)[0])["accepted"] for t in CT.load("val"))
    return {"recipes_fully_right": rec, "code_tasks_solved": code}


def best_q4(f16, out, log=print, extra=()):
    """-> (path, report). out is where the winner ends up."""
    plain, imat = out + ".plain.gguf", out + ".imatrix.gguf"
    R.quantize(f16, plain)
    report = {}
    try:
        calib = R.calibration_text(out + ".calib.txt")
        im = R.make_imatrix(f16, calib, out + ".imatrix")
        R.quantize(f16, imat, imatrix=im)
        os.remove(calib); os.remove(im)
        report["plain"] = score(plain, extra, log); report["imatrix"] = score(imat, extra, log)
        a, b = report["plain"], report["imatrix"]
        win = imat if (b["recipes_fully_right"], b["code_tasks_solved"]) >= (a["recipes_fully_right"] - 0.005, a["code_tasks_solved"]) else plain
    except Exception as e:  # the importance matrix couldn't be made here: plain compression, as before
        report["imatrix_error"] = str(e)[:300]; win = plain
    for f in (plain, imat):
        if f != win and os.path.exists(f): os.remove(f)
    os.replace(win, out)
    report["kept"] = "imatrix" if win == imat else "plain"
    log(f"compression: kept {report['kept']} " + (f"(plain {report.get('plain')}, imatrix {report.get('imatrix')})" if "plain" in report else ""))
    return out, report
