"""Differential tests: ``count_pretokens`` against encoding every piece, then counting.

``reference_count`` is the plain serial count. Any change to how ``count_pretokens``
counts or encodes (such as counting ``str`` pieces and encoding only the distinct ones,
see TODO.md) must give the same counts in the same order, serial and in batches, and
on text UTF-8 cannot encode (lone surrogates) the same ``UnicodeEncodeError``.
"""

import unicodedata
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import texts

from fastbpe.pretokenize import count_pretokens, pretokenize


def reference_count(texts: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        counts.update(piece.encode() for piece in pretokenize(text))
    return counts


def threaded_count(texts: list[str]) -> Counter[bytes]:
    # Batches as with workers=3, in threads: the parallel path without process start-up.
    with ThreadPoolExecutor(3) as executor:
        return count_pretokens(texts, workers=3, executor=executor)


def outcome(count: Callable[[list[str]], Counter[bytes]], texts: list[str]) -> object:
    """The counts as an ordered list, or the error's type and arguments."""
    try:
        return list(count(texts).items())
    except UnicodeEncodeError as error:
        return type(error), error.args


@given(st.lists(texts, max_size=8))
@pytest.mark.parametrize("count", [count_pretokens, threaded_count])
def test_matches_reference(count: Callable[[list[str]], Counter[bytes]], texts: list[str]) -> None:
    assert outcome(count, texts) == outcome(reference_count, texts)


@pytest.mark.parametrize("count", [count_pretokens, threaded_count])
def test_composed_and_decomposed_are_different_pieces(
    count: Callable[[list[str]], Counter[bytes]],
) -> None:
    composed = "caf\xe9"
    decomposed = unicodedata.normalize("NFD", composed)
    texts = [composed, decomposed, composed]
    counts = count(texts)
    # The combining accent (Mn) is a piece of its own, so "cafe" and "́" appear.
    assert counts == {composed.encode(): 2, b"cafe": 1, "́".encode(): 1}
    assert list(counts.items()) == list(reference_count(texts).items())


@pytest.mark.parametrize("count", [count_pretokens, threaded_count])
def test_lone_surrogate_raises(count: Callable[[list[str]], Counter[bytes]]) -> None:
    with pytest.raises(UnicodeEncodeError):
        count(["fine", "bad \ud800 piece"])
