"""Invariants of trained vocabularies.

Regular tokens (the 256 bytes and one per merge) have distinct bytes: a token forms in
every word from the same merge, so no later merge can build its bytes again (see
``_train_merges``). Special tokens are a namespace of their own, as in tiktoken: a
single-byte special token has the bytes of a base token, and both ids stay valid.
"""

import unicodedata
import warnings

import pytest
import tiktoken
from bpe_strategies import overlapping_words, piece_counts
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import texts

from fastbpe import Tokenizer, train_bpe
from fastbpe.train import _train_merges

ALL_MERGES = 10_000
SPECIALS = ["a", "b", " ", "\x00", "\x7f", "<|endoftext|>", "ab", "\xe9"]


def _encodable(text: str) -> bool:
    try:
        text.encode()
    except UnicodeEncodeError:  # lone surrogates, which training rejects
        return False
    return True


words = st.lists(overlapping_words().map(bytes.decode), min_size=1, max_size=8).map(" ".join)
plain_text = st.one_of(texts.filter(_encodable), words)


@st.composite
def corpora_with_specials(draw: st.DrawFn) -> tuple[str, list[str]]:
    """Text with special tokens, single-byte ones included, between its parts."""
    specials = draw(st.lists(st.sampled_from(SPECIALS), unique=True, max_size=3))
    parts = draw(st.lists(plain_text, min_size=1, max_size=4))
    separators = st.sampled_from(specials) if specials else st.just("")
    text = parts[0] + "".join(draw(separators) + part for part in parts[1:])
    return text, specials


def train_quietly(text: str, vocab_size: int, specials: list[str]) -> Tokenizer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # single-byte special tokens
        return Tokenizer.train(text, vocab_size, specials)


@given(piece_counts(overlapping_words()))
def test_merges_build_distinct_tokens(counts: dict[bytes, int]) -> None:
    merges = _train_merges(counts, ALL_MERGES)
    assert len(set(merges)) == len(merges)
    tokens = [left + right for left, right in merges]
    assert len(set(tokens)) == len(tokens)
    assert all(len(token) > 1 for token in tokens)  # never one of the 256 bytes


@given(corpora_with_specials(), st.integers(0, 120))
def test_regular_tokens_have_distinct_bytes(case: tuple[str, list[str]], merges: int) -> None:
    text, specials = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        vocab, merge_list = train_bpe(text, 256 + merges + len(specials), specials)
    regular = [vocab[i] for i in range(256 + len(merge_list))]
    assert len(set(regular)) == len(regular)
    # Special tokens never coincide with a merged token: those are contiguous bytes of
    # text between special tokens, which (UTF-8 being self-synchronizing) cannot contain
    # a special token's bytes. Only a single-byte special token meets a regular token.
    for special in specials:
        clash = [token for token in regular if token == special.encode()]
        assert clash == ([special.encode()] if len(special.encode()) == 1 else [])


@pytest.mark.oracle
@given(corpora_with_specials(), st.integers(0, 60))
def test_special_tokens_sharing_bytes_with_regular_tokens_follow_tiktoken(
    gpt2: tiktoken.Encoding, case: tuple[str, list[str]], merges: int
) -> None:
    text, specials = case
    # Where one special token is a prefix of another, both match at one position.
    # tiktoken then takes whichever its hash map lists first ("a"/"ab": the shorter,
    # "xy"/"xyz": the longer), so it is no oracle there; this tokenizer takes the longest.
    assume(not any(a != b and b.startswith(a) for a in specials for b in specials))
    # As in test_oracle.py: code points this Python's Unicode database does not know
    # may be letters to tiktoken's newer one.
    assume(all(unicodedata.category(c) != "Cn" for c in text))
    tokenizer = train_quietly(text, 256 + merges + len(specials), specials)
    special_ids = tokenizer.special_tokens
    ranks = {
        token: token_id
        for token_id, token in tokenizer.vocab.items()
        if token_id not in special_ids.values()
    }
    enc = tiktoken.Encoding(
        "trained", pat_str=gpt2._pat_str, mergeable_ranks=ranks, special_tokens=special_ids
    )
    assert tokenizer.encode(text) == enc.encode(text, allowed_special="all")
    assert tokenizer.encode(text, allowed_special=set()) == enc.encode(
        text, allowed_special=set(), disallowed_special=()
    )
    assert tokenizer.decode_bytes(tokenizer.encode(text)) == text.encode()
