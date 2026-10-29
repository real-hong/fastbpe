"""Pre-tokenization: split text into GPT-2 style pieces and count them."""

import multiprocessing
import re
import unicodedata
import warnings
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import Executor, ProcessPoolExecutor
from multiprocessing.context import BaseContext

# The GPT-2 split pattern as used by tiktoken. It needs the third-party ``regex`` module
# (``\p{..}`` classes, possessive quantifiers), so it is kept here only as the reference
# that ``pretokenize`` must match; the implementation itself uses the standard library.
# Character classes come from ``unicodedata``, so code points this Python's Unicode
# database does not assign yet count as "other" rather than as letters or numbers.
GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)

# Fast path: ``pretokenize`` maps every character to an ASCII representative of its class
# with ``str.translate`` and runs an ASCII version of the pattern with ``re``, which only
# knows ASCII classes but is far faster than classifying characters in Python.
#
# This relies on what the GPT-2 pattern can tell apart, and on nothing else:
# - letters, numbers, whitespace and everything else, never which letter or number:
#   numbers are not split into groups of digits, and no class depends on case;
# - the characters it names literally: ``'``, the lowercase contraction letters
#   ``sdmtlvre`` (contractions are case-sensitive) and U+0020 (the ``" ?"`` prefix);
# - whitespace is one class: newlines are not told apart from other whitespace.
# Patterns that see more, such as cl100k's case-insensitive contractions or its
# ``[\r\n]`` classes, would need other representatives, so ``_compile_mapped`` accepts
# only the patterns listed in ``_ASCII_PATTERNS``, each checked against this mapping.
_LITERAL = frozenset("'sdmtlvre ")

_ASCII_PATTERNS = {
    # \p{L} -> [a-zA-Z], \p{N} -> [0-9]; $ -> \Z, which is what \s++$ amounts to (the
    # possessive \s++ has already taken any final newline).
    GPT2_PATTERN: (
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]++| ?[0-9]++| ?[^\sa-zA-Z0-9]++|\s++\Z|\s+(?!\S)|\s"""
    ),
}


def _is_space(char: str) -> bool:
    # ``\s`` in the pattern is Unicode White_Space, which unlike str.isspace() excludes
    # the ASCII separators U+001C..U+001F.
    return char.isspace() and not "\x1c" <= char <= "\x1f"


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_number(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _representative(char: str) -> str:
    """The single ASCII character that stands for ``char`` in the ASCII pattern."""
    if char in _LITERAL:
        return char
    if _is_space(char):
        return "\t"  # not " ": only U+0020 itself may match the optional leading space
    if _is_letter(char):
        return "a"
    if _is_number(char):
        return "0"
    return "!"


class _RepresentativeTable(dict[int, str]):
    """``str.translate`` table from code point to representative, filled on first use.

    Classifying all 0x110000 code points at import would take about 0.9 s; text uses few
    of them, and each is classified once, the first time it is seen.
    """

    def __missing__(self, code_point: int) -> str:
        representative = self[code_point] = _representative(chr(code_point))
        return representative


def _compile_mapped(pattern: str) -> re.Pattern[str]:
    """Compile the ASCII version of ``pattern`` that runs on representatives."""
    try:
        ascii_pattern = _ASCII_PATTERNS[pattern]
    except KeyError:
        raise ValueError(
            "the category-map fast path only supports GPT2_PATTERN; other patterns may "
            "tell apart characters that share a representative"
        ) from None
    return re.compile(ascii_pattern, re.ASCII)


_TABLE = _RepresentativeTable()
_SPLIT = _compile_mapped(GPT2_PATTERN)


def pretokenize(text: str) -> list[str]:
    """Split ``text`` exactly as ``regex.findall(GPT2_PATTERN, text)`` would.

    The pattern runs on the representatives of ``text``'s characters. Each character
    maps to one character, so a match of length ``n`` in the mapped text is the next
    ``n`` characters of ``text``.
    """
    mapped = text.translate(_TABLE)
    if len(mapped) != len(text):
        raise RuntimeError("character table must map each character to one character")
    pieces = []
    start = 0
    for match in _SPLIT.findall(mapped):
        end = start + len(match)
        pieces.append(text[start:end])
        start = end
    if start != len(text):
        raise RuntimeError("pattern matches must cover the whole text")
    return pieces


def split_on_special_tokens(text: str, special_tokens: Iterable[str]) -> list[str]:
    """Split ``text`` around special tokens, keeping each one as its own chunk.

    Where special tokens overlap, the longest one wins. Empty chunks are dropped.
    Raises ``ValueError`` for an empty special token, which would match between every
    pair of characters.
    """
    tokens = sorted(special_tokens, key=len, reverse=True)
    if tokens and not tokens[-1]:
        raise ValueError("special tokens must not be empty")
    if not tokens:
        return [text] if text else []
    pattern = "(" + "|".join(re.escape(token) for token in tokens) + ")"
    return [chunk for chunk in re.split(pattern, text) if chunk]


def _count(texts: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        counts.update(piece.encode() for piece in pretokenize(text))
    return counts


def _batches(texts: list[str], n: int) -> list[list[str]]:
    """Group consecutive texts into about ``n`` batches of similar total length."""
    target = sum(len(text) for text in texts) / n
    batches: list[list[str]] = [[]]
    size = 0
    for text in texts:
        if size >= target:
            batches.append([])
            size = 0
        batches[-1].append(text)
        size += len(text)
    return batches


def default_mp_context() -> BaseContext:
    """forkserver where the platform has it, else spawn; never fork.

    Forking a process that has other threads running can deadlock the child.
    forkserver forks workers from a clean single-threaded server instead, and is cheaper
    to start than spawn.
    """
    if "forkserver" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def count_pretokens(
    texts: Iterable[str],
    *,
    workers: int = 1,
    mp_context: BaseContext | None = None,
    executor: Executor | None = None,
) -> Counter[bytes]:
    """Count the UTF-8 encoded pieces of every text; ``workers > 1`` counts in parallel.

    Each text is pre-tokenized whole, so the texts must already be cut where no piece
    can cross, such as at special tokens. Parallelism is only as good as that split:
    a single text runs in a single process.

    Parallel counting is opt-in (``workers=1`` by default). It starts ``workers``
    processes with ``mp_context`` (default: ``default_mp_context()``), so a script
    calling it needs an ``if __name__ == "__main__":`` guard. Inside a daemonic process
    (e.g. a ``multiprocessing.Pool`` worker), which may not start children, it warns and
    counts in the current process instead. Pass ``executor`` to run the batches on an
    executor you own (it is left open); ``workers`` then only sets how many batches.
    """
    if executor is not None and mp_context is not None:
        raise ValueError("pass either executor or mp_context, not both")
    texts = list(texts)
    if workers <= 1:
        return _count(texts)
    # A few batches per worker keeps them all busy when text lengths vary.
    batches = _batches(texts, workers * 4)
    if len(batches) == 1:
        return _count(texts)
    if executor is not None:
        return _count_batches(executor, batches)
    if multiprocessing.current_process().daemon:
        warnings.warn(
            "daemonic processes cannot start workers; counting pre-tokens in this process",
            RuntimeWarning,
            stacklevel=2,
        )
        return _count(texts)
    with ProcessPoolExecutor(workers, mp_context=mp_context or default_mp_context()) as pool:
        return _count_batches(pool, batches)


def _count_batches(executor: Executor, batches: list[list[str]]) -> Counter[bytes]:
    total: Counter[bytes] = Counter()
    for counts in executor.map(_count, batches):
        total.update(counts)
    return total
