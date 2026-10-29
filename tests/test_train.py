import warnings

import pytest
from naive_bpe import naive_train_bpe

from fastbpe import Tokenizer, train_bpe

EOT = "<|endoftext|>"


def test_base_vocab_is_single_bytes() -> None:
    vocab, merges = train_bpe("hello", vocab_size=256)
    assert merges == []
    assert vocab == {i: bytes([i]) for i in range(256)}


def test_single_merge_picks_most_frequent_pair() -> None:
    # pieces: "ab", " ab", " ab", " cd" -> (a, b) occurs 3 times
    vocab, merges = train_bpe("ab ab ab cd", vocab_size=257)
    assert merges == [(b"a", b"b")]
    assert vocab[256] == b"ab"
    assert len(vocab) == 257


@pytest.mark.parametrize("text", ["ab cd", "cd ab"])
def test_tie_breaks_to_lexicographically_larger_pair(text: str) -> None:
    # (a, b), (" ", c) and (c, d) all occur once; (c, d) is the largest
    _, merges = train_bpe(text, vocab_size=257)
    assert merges == [(b"c", b"d")]


def test_merge_sequence_and_early_stop() -> None:
    vocab, merges = train_bpe("ab ab ab cd", vocab_size=1000)
    assert merges == [
        (b"a", b"b"),
        (b" ", b"ab"),
        (b"c", b"d"),  # ties with (" ", c); c > " "
        (b" ", b"cd"),
    ]
    assert [vocab[i] for i in range(256, 260)] == [b"ab", b" ab", b"cd", b" cd"]
    assert len(vocab) == 260


def test_special_tokens_are_not_merged_and_come_last() -> None:
    vocab, merges = train_bpe(f"hello{EOT}world", vocab_size=1000, special_tokens=[EOT])
    assert merges
    for left, right in merges:
        merged = left + right
        assert merged in b"hello" or merged in b"world"
    assert vocab[max(vocab)] == EOT.encode()
    assert max(vocab) == 256 + len(merges)


def test_special_tokens_count_toward_vocab_size() -> None:
    vocab, merges = train_bpe("ab ab ab cd", vocab_size=258, special_tokens=[EOT, "<|pad|>"])
    assert merges == []
    assert vocab[256] == EOT.encode()
    assert vocab[257] == b"<|pad|>"


@pytest.mark.parametrize(("vocab_size", "special_tokens"), [(255, []), (0, []), (256, [EOT])])
def test_vocab_size_too_small_raises(vocab_size: int, special_tokens: list[str]) -> None:
    with pytest.raises(ValueError):
        train_bpe("abc", vocab_size=vocab_size, special_tokens=special_tokens)


def test_tokenizer_train_matches_train_bpe() -> None:
    tok = Tokenizer.train("ab ab ab cd", vocab_size=300, special_tokens=[EOT])
    vocab, merges = train_bpe("ab ab ab cd", vocab_size=300, special_tokens=[EOT])
    assert tok.vocab == vocab
    assert tok.merges == merges
    assert tok.special_tokens == {EOT: 260}


def test_parallel_training_matches_serial() -> None:
    docs = [f"document {i}: the quick brown fox jumps over the dog #{i % 7}" for i in range(200)]
    text = EOT.join(docs)
    serial = train_bpe(text, vocab_size=400, special_tokens=[EOT])
    assert train_bpe(text, vocab_size=400, special_tokens=[EOT], workers=4) == serial


def test_empty_special_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        train_bpe("ab ab ab cd", vocab_size=300, special_tokens=[""])


def test_duplicate_special_tokens_are_rejected() -> None:
    # Would otherwise give two vocab ids for one token, one of them a ghost regular token.
    with pytest.raises(ValueError, match="duplicate"):
        train_bpe("abc <x> abc", vocab_size=300, special_tokens=["<x>", "<y>", "<x>"])


@pytest.mark.parametrize("special", ["a", "\x00", " ", "\x7f"])
def test_single_byte_special_token_warns(special: str) -> None:
    text = f"abc{special}ab a\x00b\x7f"
    with pytest.warns(UserWarning, match="will not take part in merges") as caught:
        warned = train_bpe(text, vocab_size=300, special_tokens=[special, EOT])
    assert len(caught) == 1
    assert repr(special) in str(caught[0].message)
    # The warning changes nothing: same result with it filtered out, and as the naive
    # reference trainer, which does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert train_bpe(text, vocab_size=300, special_tokens=[special, EOT]) == warned
    assert naive_train_bpe(text, vocab_size=300, special_tokens=[special, EOT]) == warned


def test_tokenizer_train_warns_for_single_byte_special_token() -> None:
    with pytest.warns(UserWarning, match="will not take part in merges"):
        Tokenizer.train("abc abc", vocab_size=300, special_tokens=["a"])


@pytest.mark.parametrize("special", [EOT, "ab", "\xe9", "中"])
def test_multi_byte_special_tokens_do_not_warn(special: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train_bpe(f"abc{special}abc", vocab_size=300, special_tokens=[special])
