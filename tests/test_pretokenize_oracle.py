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

import fastbpe.pretokenize as pretokenize_module
from fastbpe.pretokenize import GPT2_PATTERN, _compile_mapped, _representative, pretokenize

# Characters at the edges of the pattern's classes, written as escapes to keep them
# readable. Hypothesis rarely draws most of them from the full range on its own.
WHITESPACE = [
    *"\t\n\x0b\x0c\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000",
    *map(chr, range(0x2000, 0x200B)),  # EN QUAD .. HAIR SPACE
]
# Not White_Space, though str.isspace() or their names suggest otherwise.
NOT_WHITESPACE = [*"\x1c\x1d\x1e\x1f\u180e\u200b\u200c\u200d\u2060\ufeff"]
CONTRACTION = [*"'sdmtlvreSDMTLVRE", "\u2019"]  # and RIGHT SINGLE QUOTATION MARK
LETTERS = [
    *"aZ\xe9\xdf\u01c5\u02b0\xaa\u4e2d\u3042\ud55c\u05d0\u0627",  # Lu Ll Lt Lm Lo, CJK
    "\U0001d400",  # MATHEMATICAL BOLD CAPITAL A, outside the BMP
    "\U00020000",  # CJK EXTENSION B
]
MARKS = [*"\u0301\u0903\u20dd\u0e31"]  # Mn Mc Me, Thai vowel sign
NUMBERS = [*"09\u0663\u216b\xb2\xbd\u3007", "\U0001d7ce"]  # Nd Nl No, astral digit
OTHER = [
    *"!_-.,$^\xac\xa9\uff0c\u3002\x00\x7f",
    "\ue000",  # private use (Co)
    "\U000f0000",  # supplementary private use
    "\u0378",  # unassigned (Cn)
    "\U0010ffff",  # last code point, a noncharacter
    "\ud800",  # lone surrogates (Cs)
    "\udfff",
]
EMOJI = ["\U0001f44d", "\U0001f3fd", "\u200d", "\u2764", "\ufe0f", "\U0001f1e8", "\U0001f1f3"]
INTERESTING = WHITESPACE + NOT_WHITESPACE + CONTRACTION + LETTERS + MARKS + NUMBERS + OTHER + EMOJI

# Multi-character pieces around the contraction and whitespace alternatives.
FRAGMENTS = [
    *("'s", "'d", "'m", "'t", "'ll", "'ve", "'re", "'S", "'LL", "'l", "'v", "'r"),
    *(" ", "  ", "   ", "\n", "\n\n", " \n", "\n ", "\r\n", " \t ", "\u3000 "),
    *("ab", " ab", "12", " 12", "!!", " !!", "e\u0301", "\U0001f468\u200d\U0001f469"),
]

texts = st.one_of(
    st.text(st.characters(exclude_categories=())),  # any code point, surrogates included
    st.text(st.sampled_from(INTERESTING)),
    st.lists(st.sampled_from(FRAGMENTS + INTERESTING)).map("".join),
)


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
