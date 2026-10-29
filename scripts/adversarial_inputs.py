"""Inputs for pre-tokenization and encoding benchmarks: the corpus and adversarial cases.

Each generator is deterministic (seeded) and returns about ``n`` bytes of UTF-8, except
``many_specials``, which also returns the special tokens it inserted.
"""

import base64
import random
import re
import sys
import unicodedata
from collections.abc import Callable
from functools import cache

from corpus import SEPARATOR, load_corpus


def regular(n: int) -> str:
    return load_corpus(n)


def dna(n: int) -> str:
    # Letters only: the whole string is one pre-token.
    return "".join(random.Random(0).choices("ACGT", k=n))


def base64_text(n: int) -> str:
    # Letters, digits and "+/=" alternate: many short, mostly distinct pieces.
    return base64.b64encode(random.Random(0).randbytes(n * 3 // 4)).decode()


def minified(n: int) -> str:
    # The corpus (mostly Python code) with all whitespace removed, like minified JS.
    return re.sub(r"\s+", "", load_corpus(n * 6 // 5))[:n]


def cjk(n: int) -> str:
    # ~20k distinct ideographs (3 bytes each), with CJK punctuation and some spaces.
    r = random.Random(0)
    chars = [
        chr(r.randint(0x4E00, 0x9FFF)) if r.random() < 0.9 else r.choice(" ，。\n")
        for _ in range(n // 3)
    ]
    return "".join(chars)


@cache
def _assigned() -> list[str]:
    return [
        c
        for c in map(chr, range(sys.maxunicode + 1))
        if unicodedata.category(c) not in ("Cn", "Cs")
    ]


def unicode_all(n: int) -> str:
    # Assigned code points from every plane: as many distinct characters as possible.
    r = random.Random(0)
    assigned = _assigned()
    out: list[str] = []
    size = 0
    while size < n:
        c = " " if r.random() < 0.1 else r.choice(assigned)
        out.append(c)
        size += len(c.encode("utf-8", "surrogatepass"))
    return "".join(out)


_EMOJI = ["👍", "👍🏽", "👨‍👩‍👧‍👦", "🇨🇳", "🌍", "🏳️‍🌈", "😀", "❤️", "🧑🏿‍💻"]


def emoji(n: int) -> str:
    r = random.Random(0)
    out: list[str] = []
    size = 0
    while size < n:
        piece = r.choice(_EMOJI) * r.randint(1, 3) + r.choice([" ", " ok ", "\n", ""])
        out.append(piece)
        size += len(piece.encode())
    return "".join(out)


_SPACES = [" ", "\t", "\n", "\r\n", "\u3000", "\xa0", "\u2003", "\x0b"]


def whitespace(n: int) -> str:
    # Whitespace runs of up to 200 mixed characters between short words.
    r = random.Random(0)
    out: list[str] = []
    size = 0
    while size < n:
        piece = "".join(r.choices(_SPACES, k=r.randint(1, 200))) + r.choice(["a", "x1", "!"])
        out.append(piece)
        size += len(piece.encode())
    return "".join(out)


def high_repetition(n: int) -> str:
    unit = "the cat sat on the mat. "
    return (unit * (n // len(unit) + 1))[:n]


def low_repetition(n: int) -> str:
    # Random lowercase words of 1-10 letters: almost every piece is distinct.
    r = random.Random(0)
    out: list[str] = []
    size = 0
    while size < n:
        word = " " + "".join(r.choices("abcdefghijklmnopqrstuvwxyz", k=r.randint(1, 10)))
        out.append(word)
        size += len(word)
    return "".join(out)


def many_specials(n: int) -> tuple[str, list[str]]:
    # The corpus cut every 8-24 characters by one of 50 special tokens: tiny chunks.
    r = random.Random(0)
    specials = [f"<|s{i}|>" for i in range(50)]
    base = load_corpus(n)
    out: list[str] = []
    i = 0
    while i < len(base):
        k = r.randint(8, 24)
        out.append(base[i : i + k])
        out.append(r.choice(specials))
        i += k
    return "".join(out), specials


def stream(n: int) -> list[str]:
    """Documents for a long-running encoder, ``n`` bytes in all, encoded one by one.

    Half is the corpus (``n // 4`` bytes of it) twice over, its documents in random
    order each time; the other half is base64 and random words cut into chunks, one
    between every two documents, that never repeat. The never-repeating pieces far
    exceed the encode cache, so it is emptied or thinned again and again, while the
    corpus pieces come back after a long gap.
    """
    r = random.Random(0)
    docs = [d for d in load_corpus(n // 4).split(SEPARATOR) if d]
    order = r.sample(docs, len(docs)) + r.sample(docs, len(docs))
    filler = base64_text(n // 4) + low_repetition(n // 4)
    step = len(filler) // len(order) + 1
    out: list[str] = []
    for i, doc in enumerate(order):
        out += [doc, filler[i * step : (i + 1) * step]]
    return out


# Inputs without special tokens of their own; the corpus uses corpus.SEPARATOR.
INPUTS: dict[str, Callable[[int], str]] = {
    "regular": regular,
    "dna": dna,
    "base64": base64_text,
    "minified": minified,
    "cjk": cjk,
    "unicode_all": unicode_all,
    "emoji": emoji,
    "whitespace": whitespace,
    "high_rep": high_repetition,
    "low_rep": low_repetition,
}
