"""The known differences from tiktoken listed in README.md, pinned as tests.

If one of these fails after a tiktoken or Python upgrade, the behaviour changed:
update README.md ("Known differences from tiktoken") together with the test.
"""

import itertools
import sys
import unicodedata

import pytest
import regex
import tiktoken

from fastbpe import Tokenizer
from fastbpe.pretokenize import pretokenize

pytestmark = pytest.mark.oracle

BYTE_VOCAB = {i: bytes([i]) for i in range(256)}
BYTE_RANKS = {token: i for i, token in BYTE_VOCAB.items()}

# (shorter, longer, whether tiktoken 0.14.0 picks the longer one). tiktoken tries the
# special tokens in its hash map's order, which depends on the strings alone.
PREFIX_PAIRS = [
    ("xy", "xyz", True),
    ("q", "qq", True),
    ("<|x|>", "<|x|>!", True),
    ("ab", "abc", True),
    ("b", "ba", True),
    ("a", "ab", False),
    ("<|a|>", "<|a|><|b|>", False),
]
ID_PAIRS = [(256, 257), (257, 256), (300, 1000), (1000, 300)]


def special_assignments(short: str, long: str) -> list[dict[str, int]]:
    """Every id assignment in both insertion orders."""
    out = []
    for (short_id, long_id), long_first in itertools.product(ID_PAIRS, (False, True)):
        items = [(short, short_id), (long, long_id)]
        out.append(dict(reversed(items) if long_first else items))
    return out


@pytest.mark.parametrize(("short", "long", "tiktoken_longest"), PREFIX_PAIRS)
def test_prefix_special_tokens_longest_here_hash_order_in_tiktoken(
    gpt2: tiktoken.Encoding, short: str, long: str, tiktoken_longest: bool
) -> None:
    for specials in special_assignments(short, long):
        tokenizer = Tokenizer(BYTE_VOCAB, [], specials)
        assert tokenizer.encode(long) == [specials[long]]

        enc = tiktoken.Encoding(
            "bytes", pat_str=gpt2._pat_str, mergeable_ranks=BYTE_RANKS, special_tokens=specials
        )
        expected = (
            [specials[long]]
            if tiktoken_longest
            else [specials[short], *long[len(short) :].encode()]
        )
        # The same for every id and order: only the strings decide.
        assert enc.encode(long, allowed_special="all") == expected, specials


def test_prefix_special_tokens_agree_without_prefixes(gpt2: tiktoken.Encoding) -> None:
    specials = {"<|a|>": 256, "<|b|>": 257}
    tokenizer = Tokenizer(BYTE_VOCAB, [], specials)
    enc = tiktoken.Encoding(
        "bytes", pat_str=gpt2._pat_str, mergeable_ranks=BYTE_RANKS, special_tokens=specials
    )
    text = "x<|a|><|b|>y<|b|>"
    assert tokenizer.encode(text) == enc.encode(text, allowed_special="all")


# The Unicode versions README.md names for each supported Python.
UNICODE_VERSIONS = {(3, 12): "15.0.0", (3, 13): "15.1.0", (3, 14): "16.0.0"}


def test_readme_unicode_versions() -> None:
    expected = UNICODE_VERSIONS.get(sys.version_info[:2])
    if expected is None:
        pytest.fail(f"add Python {sys.version_info[:2]} to README.md and UNICODE_VERSIONS")
    assert unicodedata.unidata_version == expected


HIEROGLYPH = "\U00013460"  # Egyptian Hieroglyphs Extended-A, assigned in Unicode 16.0


def test_unassigned_code_points_split_differently_until_python_knows_them(
    gpt2: tiktoken.Encoding,
) -> None:
    text = ":" + HIEROGLYPH  # found by tests/test_vocab_invariants.py
    # tiktoken's tables (and the regex package's) know the code point as a letter.
    assert regex.findall(gpt2._pat_str, text) == [":", HIEROGLYPH]
    # Trained on this text, every pair occurs once and the tie rule picks merges across
    # ":" and the hieroglyph's bytes, so the split shows in the ids. (With GPT-2's
    # merges, or a repeated hieroglyph, it happens not to.)
    tokenizer = Tokenizer.train(text, vocab_size=256 + 4)
    ranks = {token: token_id for token_id, token in tokenizer.vocab.items()}
    enc = tiktoken.Encoding(
        "trained", pat_str=gpt2._pat_str, mergeable_ranks=ranks, special_tokens={}
    )
    if unicodedata.unidata_version < "16.0.0":  # Python 3.12 (15.0), 3.13 (15.1)
        assert unicodedata.category(HIEROGLYPH) == "Cn"
        assert pretokenize(text) == [text]  # "other": joins the preceding ":"
        assert tokenizer.encode(text) != enc.encode(text)
    else:  # Python 3.14 (16.0): a letter here too, no difference
        assert unicodedata.category(HIEROGLYPH) == "Lo"
        assert pretokenize(text) == [":", HIEROGLYPH]
        assert tokenizer.encode(text) == enc.encode(text)


def test_single_byte_special_token_shares_bytes_like_tiktoken(gpt2: tiktoken.Encoding) -> None:
    specials = {"a": 256}
    tokenizer = Tokenizer(BYTE_VOCAB, [], specials)
    enc = tiktoken.Encoding(
        "bytes", pat_str=gpt2._pat_str, mergeable_ranks=BYTE_RANKS, special_tokens=specials
    )
    assert tokenizer.encode("abc") == enc.encode("abc", allowed_special="all") == [256, 98, 99]
    assert (
        tokenizer.encode("abc", allowed_special=set())
        == enc.encode("abc", allowed_special=set(), disallowed_special=())
        == [97, 98, 99]
    )
