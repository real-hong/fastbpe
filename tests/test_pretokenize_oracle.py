"""Differential tests: ``pretokenize`` against the reference pre-tokenizer.

The reference (``tests/reference_pretokenize.py``) is the character-by-character
implementation from before review finding #7. Both classify characters with this
Python's ``unicodedata``, so they must agree on every input, unassigned code points and
lone surrogates included.
"""

import re
import sys

import pytest
import reference_pretokenize as reference
from hypothesis import given
from hypothesis import strategies as st
from reference_pretokenize import pretokenize as reference_pretokenize
from strategies import INTERESTING, texts

import fastbpe.pretokenize as pretokenize_module
from fastbpe.pretokenize import GPT2_PATTERN, _compile_mapped, _representative, pretokenize


@given(texts)
def test_matches_reference(text: str) -> None:
    pieces = pretokenize(text)
    assert pieces == reference_pretokenize(text)
    assert "".join(pieces) == text


@pytest.mark.slow
def test_every_code_point_matches_reference() -> None:
    # Each code point alone and between other classes, as in scripts/review/fuzz_pre.py.
    mismatches = []
    for cp in range(sys.maxunicode + 1):
        c = chr(cp)
        for text in (c, " " + c + "a", "a" + c + " ", "'" + c + c + "\n"):
            if pretokenize(text) != reference_pretokenize(text):
                mismatches.append((hex(cp), text))
                break
    assert mismatches == []


# --- the category map behind the fast path ---------------------------------------------

ASCII_SPACE = frozenset(" \t\n\r\x0b\x0c")  # \s under re.ASCII
LITERAL = frozenset("'sdmtlvre ")  # characters the pattern names, kept as they are


def check_representative(char: str) -> None:
    rep = _representative(char)
    # One character for one character, so match offsets in the mapped text are offsets
    # in the original.
    assert len(rep) == 1
    assert rep.isascii()
    # Same class under the ASCII pattern as under the reference predicates.
    assert (rep in ASCII_SPACE) == reference._is_space(char)
    assert rep.isalpha() == reference._is_letter(char)
    assert rep.isdigit() == reference._is_number(char)
    # Literal characters stay themselves; nothing else may become one.
    assert (rep == char) if char in LITERAL else (rep not in LITERAL)


@given(st.one_of(st.characters(exclude_categories=()), st.sampled_from(INTERESTING)))
def test_representative(char: str) -> None:
    check_representative(char)


@pytest.mark.slow
def test_representative_of_every_code_point() -> None:
    for cp in range(sys.maxunicode + 1):
        check_representative(chr(cp))


def test_fast_path_accepts_only_the_gpt2_pattern() -> None:
    assert _compile_mapped(GPT2_PATTERN).pattern != GPT2_PATTERN  # the ASCII rewrite
    cl100k = (
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"""
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )
    for pattern in (cl100k, GPT2_PATTERN.replace("[sdmt]", "[sdmtx]"), r"\w+|\s+"):
        with pytest.raises(ValueError, match="GPT2_PATTERN"):
            _compile_mapped(pattern)


def test_table_entry_of_wrong_length_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(pretokenize_module._TABLE, ord("x"), "ab")
    with pytest.raises(RuntimeError, match="one character"):
        pretokenize("x y")


def test_unmatched_character_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # findall skips what the pattern cannot match; offsets would silently shift.
    monkeypatch.setattr(pretokenize_module, "_SPLIT", re.compile(r"[a-z]+"))
    with pytest.raises(RuntimeError, match="cover"):
        pretokenize("ab!cd")
