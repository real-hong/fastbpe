from collections.abc import Set as AbstractSet
from pathlib import Path

import pytest

from fastbpe import Tokenizer

EOT = "<|endoftext|>"
SAMPLES = ["hello world", f"before{EOT}after", "你好 👍🏽", ""]


def test_vocab_size(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.vocab_size == 300
    assert len(small_tokenizer.vocab) == 300


def test_constructor_matches_trained(small_tokenizer: Tokenizer) -> None:
    rebuilt = Tokenizer(
        small_tokenizer.vocab, small_tokenizer.merges, small_tokenizer.special_tokens
    )
    for text in SAMPLES:
        assert rebuilt.encode(text) == small_tokenizer.encode(text)


def test_special_token_alone(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.encode(EOT) == [small_tokenizer.special_tokens[EOT]]


def test_special_token_inside_text(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.encode(f"hello{EOT} world") == [
        *small_tokenizer.encode("hello"),
        eot,
        *small_tokenizer.encode(" world"),
    ]


def test_overlapping_special_tokens_use_longest_match(small_corpus: str) -> None:
    tok = Tokenizer.train(small_corpus, vocab_size=300, special_tokens=["<|a|>", "<|a|><|b|>"])
    long_id = tok.special_tokens["<|a|><|b|>"]
    short_id = tok.special_tokens["<|a|>"]
    assert tok.encode("<|a|><|b|>") == [long_id]
    assert tok.encode("<|a|><|a|>") == [short_id, short_id]


def test_disallowed_special_is_plain_text(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    ids = small_tokenizer.encode(EOT, allowed_special=set())
    assert eot not in ids
    assert len(ids) > 1
    assert small_tokenizer.decode(ids) == EOT


def test_allowed_special_subset(small_corpus: str) -> None:
    tok = Tokenizer.train(small_corpus, vocab_size=300, special_tokens=[EOT, "<|pad|>"])
    ids = tok.encode(f"{EOT}<|pad|>", allowed_special={"<|pad|>"})
    assert tok.special_tokens[EOT] not in ids
    assert ids[-1] == tok.special_tokens["<|pad|>"]


@pytest.mark.parametrize("bad", ["none", EOT, "", "ALL", [EOT], (EOT,), None])
def test_allowed_special_rejects_anything_but_all_or_a_set(
    small_tokenizer: Tokenizer, bad: object
) -> None:
    # As in tiktoken: a string other than "all" is not silently treated as "all".
    with pytest.raises(TypeError):
        small_tokenizer.encode(f"hi{EOT}", allowed_special=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("allowed", [frozenset({EOT}), {EOT: 0}.keys()])
def test_allowed_special_accepts_any_abstract_set(
    small_tokenizer: Tokenizer, allowed: AbstractSet[str]
) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.encode(EOT, allowed_special=allowed) == [eot]


def test_decode_special_token(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.decode([eot]) == EOT
    assert small_tokenizer.decode_bytes([eot]) == EOT.encode()


def test_decode_replaces_invalid_utf8(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.decode([0xFF]) == "�"


def test_save_load_roundtrip(small_tokenizer: Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    small_tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocab == small_tokenizer.vocab
    assert loaded.merges == small_tokenizer.merges
    assert loaded.special_tokens == small_tokenizer.special_tokens
    for text in SAMPLES:
        assert loaded.encode(text) == small_tokenizer.encode(text)


def test_save_load_accepts_str_path(small_tokenizer: Tokenizer, tmp_path: Path) -> None:
    path = str(tmp_path / "tok.json")
    small_tokenizer.save(path)
    assert Tokenizer.load(path).vocab == small_tokenizer.vocab


@pytest.mark.oracle
def test_from_ranks_gpt2(gpt2_tokenizer: Tokenizer) -> None:
    assert gpt2_tokenizer.vocab_size == 50257
    assert gpt2_tokenizer.special_tokens == {EOT: 50256}
