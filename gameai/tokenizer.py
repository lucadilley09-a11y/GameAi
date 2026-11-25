"""Our own tokenizer. Prompts are read letter by letter (robust to typos and slang);
recipes are written as one token per field value (e.g. 'theme=dungeon').

Sequence layout:  [BOS] p r o m p t ... [SEP] theme=x player=y ... powerup=z [EOS]
"""
from .recipe import OUT_FIELDS, out_options

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 .,'!?-&")
MAX_PROMPT = 240  # Phase 2: prompts can carry rules too


class Tokenizer:
    def __init__(self):
        self.itos = SPECIALS + CHARS + [f"{f}={v}" for f in OUT_FIELDS for v in out_options(f)]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad, self.bos, self.sep, self.eos, self.unk = range(5)
        # allowed token ids for each output slot: this IS the grammar, as a mask
        self.field_ids = [[self.stoi[f"{f}={v}"] for v in out_options(f)] for f in OUT_FIELDS]
        self.max_len = 1 + MAX_PROMPT + 1 + len(OUT_FIELDS) + 1

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode_prompt(self, text):
        t = " ".join(text.lower().split())[:MAX_PROMPT]
        return [self.bos] + [self.stoi.get(c, self.unk) for c in t] + [self.sep]

    def encode_recipe(self, recipe):
        return [self.stoi[f"{f}={recipe[f]}"] for f in OUT_FIELDS] + [self.eos]

    def decode_value(self, slot, token_id):
        return self.itos[token_id].split("=", 1)[1]
