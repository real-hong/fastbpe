"""Pre-tokenization: split text into GPT-2 style pieces and count them."""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

# The GPT-2 split pattern as used by tiktoken. It needs the third-party ``regex`` module
# (``\p{..}`` classes, possessive quantifiers), so it is kept here only as the reference
# that ``pretokenize`` must match; the implementation itself uses the standard library.
# Character classes come from ``unicodedata``, so code points this Python's Unicode
# database does not assign yet count as "other" rather than as letters or numbers.
GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)

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


def split_on_special_tokens(text: str, special_tokens: Iterable[str]) -> list[str]:
    """Split ``text`` around special tokens, keeping each one as its own chunk.

    Where special tokens overlap, the longest one wins. Empty chunks are dropped.
    """
    tokens = sorted(special_tokens, key=len, reverse=True)
    if not tokens:
        return [text] if text else []
    pattern = "(" + "|".join(re.escape(token) for token in tokens) + ")"
    return [chunk for chunk in re.split(pattern, text) if chunk]


def _count(texts: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        counts.update(piece.encode() for piece in pretokenize(text))
    return counts


def count_pretokens(texts: Iterable[str], *, workers: int = 1) -> Counter[bytes]:
    """Count the UTF-8 encoded pieces of every text; ``workers > 1`` counts in parallel."""
    texts = list(texts)
    if workers <= 1:
        return _count(texts)
    chunks = [texts[k::workers] for k in range(workers)]
    total: Counter[bytes] = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for counts in pool.map(_count, chunks):
            total.update(counts)
    return total
