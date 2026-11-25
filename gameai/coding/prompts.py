"""How a coding task is put to a model (the AI itself, or the coding teacher), first attempts and repairs alike."""
import os

LANG_NAME = {"js": "JavaScript", "py": "Python", "lua": "Lua"}
FENCE = {"js": "javascript", "py": "python", "lua": "lua"}


def system(lang):
    return (f"You write {LANG_NAME[lang]} code for games. Reply with the complete contents of the requested file in one "
            f"```{FENCE[lang]} code block, and nothing else. Write code that works for every input, not just the tests shown.")


_kit_map = None
def kit_reference():
    """GameKit's functions and their signatures (from the project map of gamekit.d.ts): what the code may call"""
    global _kit_map
    if _kit_map is None:
        from ..repair import repomap
        dts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kit", "gamekit.d.ts")
        _kit_map = repomap.render(repomap.build({"gamekit.d.ts": open(dts, encoding="utf-8").read()}))
    return _kit_map


def first(task):
    parts = [f"Task: {task['prompt']}", f"File to write: {task['file']}"]
    if task["lang"] == "js" and "GameKit" in task["prompt"] + "".join(task["tests"].values()):
        parts.append("GameKit (a global) has these functions:\n" + kit_reference())
    for name, text in sorted(task.get("starter", {}).items()):
        parts.append(f"Current {name}:\n```{FENCE[task['lang']]}\n{text.rstrip()}\n```")
    for name, text in sorted(task["tests"].items()):
        parts.append(f"Tests it must pass ({name}):\n```{FENCE[task['lang']]}\n{text.rstrip()}\n```")
    return "\n\n".join(parts)


def repair(task, code, report):
    """the first prompt, the failed code and what went wrong (from the visible tests only)"""
    return (first(task) + f"\n\nYour previous {task['file']}:\n```{FENCE[task['lang']]}\n{code.rstrip()}\n```\n\n"
            f"What went wrong:\n{report.strip()}\n\nWrite the corrected {task['file']}.")


def messages(task, code=None, report=None):
    return [{"role": "system", "content": system(task["lang"])},
            {"role": "user", "content": first(task) if code is None else repair(task, code, report)}]


def text_of_reply(reply):
    """a model's reply text: llama.cpp's Server.chat gives (text, timings); a plain function may give just the text"""
    return reply[0] if isinstance(reply, tuple) else reply
