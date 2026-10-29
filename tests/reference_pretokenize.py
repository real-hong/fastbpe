"""The pre-tokenizer as it was before review finding #7, kept as the oracle for tests.

Copied unchanged from ``src/fastbpe/pretokenize.py`` at commit 5d19a4c: it walks the
text one character at a time, trying the GPT-2 pattern's alternatives in order, and
classifies characters with ``unicodedata``. Slow, but each step is easy to check
against the pattern.
"""

import unicodedata
from collections.abc import Callable

_CONTRACTIONS = ("s", "d", "m", "t", "ll", "ve", "re")


def _is_space(char: str) -> bool:
    # ``\s`` in the pattern is Unicode White_Space, which unlike str.isspace() excludes
    # the ASCII separators U+001C..U+001F.
    return char.isspace() and not "\x1c" <= char <= "\x1f"


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_number(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _is_other(char: str) -> bool:
    return not (_is_space(char) or _is_letter(char) or _is_number(char))


def _run_end(text: str, start: int, predicate: Callable[[str], bool]) -> int:
    end = start
    while end < len(text) and predicate(text[end]):
        end += 1
    return end


def _piece_end(text: str, i: int) -> int:
    """Return where the piece starting at ``i`` ends, trying the pattern's alternatives in order."""
    # '(?:[sdmt]|ll|ve|re)
    if text[i] == "'":
        for suffix in _CONTRACTIONS:
            if text.startswith(suffix, i + 1):
                return i + 1 + len(suffix)

    # ` ?\p{L}++`, ` ?\p{N}++`, ` ?[^\s\p{L}\p{N}]++`. Without the optional space none of
    # these can match at a space, so trying only the with-space variant is enough.
    start = i + 1 if text[i] == " " else i
    for predicate in (_is_letter, _is_number, _is_other):
        end = _run_end(text, start, predicate)
        if end > start:
            return end

    # What is left starts with whitespace.
    end = _run_end(text, i, _is_space)
    if end == len(text):  # \s++$
        return end
    if end - i > 1:  # \s+(?!\S): leave the last whitespace char to lead the next piece
        return end - 1
    return i + 1  # \s


def pretokenize(text: str) -> list[str]:
    """Split ``text`` exactly as ``regex.findall(GPT2_PATTERN, text)`` would."""
    pieces = []
    i = 0
    while i < len(text):
        end = _piece_end(text, i)
        pieces.append(text[i:end])
        i = end
    return pieces
