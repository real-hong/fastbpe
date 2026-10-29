"""A piece that is itself a token encodes to that token, as in tiktoken.

tiktoken looks the whole piece up before running BPE; the tokenizer does the same.
For vocabularies where BPE on a token's own bytes already ends in that token, the
lookup changes nothing; these tests check that this holds for GPT-2 and for trained
vocabularies, and that for any other vocabulary the tokenizer follows tiktoken.
"""

import pytest
import tiktoken
from bpe_strategies import overlapping_words
from hypothesis import given
from hypothesis import strategies as st
from strategies import texts

from fastbpe import Tokenizer
from fastbpe.encode import bpe_parts
from fastbpe.pretokenize import pretokenize

pytestmark = pytest.mark.oracle

BYTE_VOCAB = {i: bytes([i]) for i in range(256)}


def test_gpt2_tokens_that_are_whole_pieces_match_tiktoken(
    gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer
) -> None:
    checked = 0
    for token in gpt2._mergeable_ranks:
        try:
            text = token.decode()
        except UnicodeDecodeError:
            continue
        if pretokenize(text) != [text] or text in gpt2.special_tokens_set:
            continue
        assert gpt2_tokenizer.encode(text) == gpt2.encode(text, allowed_special="all"), text
        checked += 1
    assert checked > 40_000  # nearly all of the 50,257 tokens


def test_whole_piece_token_wins_even_if_bpe_cannot_reach_it(gpt2: tiktoken.Encoding) -> None:
    # No ranked pair builds b"abc": BPE alone gives [a, b, c]; tiktoken gives the token.
    vocab = BYTE_VOCAB | {256: b"abc"}
    ranks = {token: token_id for token_id, token in vocab.items()}
    enc = tiktoken.Encoding("abc", pat_str=gpt2._pat_str, mergeable_ranks=ranks, special_tokens={})
    tokenizer = Tokenizer(vocab, [])
    for text in ["abc", "abc abcd", " abc"]:
        assert tokenizer.encode(text) == enc.encode(text)
    assert tokenizer.encode("abc") == [256]


def _encodable(text: str) -> bool:
    try:
        text.encode()
    except UnicodeEncodeError:  # lone surrogates, which training rejects
        return False
    return True


corpora = st.one_of(
    texts.filter(_encodable),
    st.lists(overlapping_words().map(bytes.decode), min_size=1, max_size=8).map(" ".join),
)


@given(corpora, st.integers(0, 80))
def test_trained_tokens_encode_to_themselves(corpus: str, merges: int) -> None:
    tokenizer = Tokenizer.train(corpus, vocab_size=256 + merges)
    ranks = {token: token_id for token_id, token in sorted(tokenizer.vocab.items(), reverse=True)}
    for token in ranks:
        assert bpe_parts(token, ranks) == [token]
