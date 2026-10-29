import random
import string

import pytest
import tiktoken

from fastbpe import Tokenizer, train_bpe

EOT = "<|endoftext|>"

LONG_INPUTS = [
    "a" * 100_000,
    " " * 100_000,
    "x" + " " * 10_000 + "y",
    "中" * 20_000,
    "1234567890" * 10_000,
]

UNICODE_INPUTS = [
    "👍🏽",
    "👨‍👩‍👧‍👦",
    "🇨🇳🇯🇵",
    "你好，世界！",
    "日本語のテキスト",
    "hello🌍世界",
    "\u0000\u001f\u007f",
]


# --- empty input ---------------------------------------------------------------------


def test_encode_empty(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.encode("") == []


def test_decode_empty(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.decode([]) == ""
    assert small_tokenizer.decode_bytes([]) == b""


def test_train_on_empty_corpus() -> None:
    vocab, merges = train_bpe("", vocab_size=1000)
    assert merges == []
    assert len(vocab) == 256


# --- single bytes --------------------------------------------------------------------


@pytest.mark.parametrize("code", range(128))
def test_single_ascii_char_is_its_byte_id(small_tokenizer: Tokenizer, code: int) -> None:
    assert small_tokenizer.encode(chr(code), allowed_special=set()) == [code]


def test_every_byte_id_decodes_to_its_byte(small_tokenizer: Tokenizer) -> None:
    for i in range(256):
        assert small_tokenizer.decode_bytes([i]) == bytes([i])


@pytest.mark.oracle
@pytest.mark.parametrize("code", range(128))
def test_single_ascii_char_gpt2(
    gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer, code: int
) -> None:
    assert gpt2_tokenizer.encode(chr(code)) == gpt2.encode(chr(code))


# --- very long inputs without spaces -------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("text", LONG_INPUTS, ids=lambda t: f"{t[:3]!r}x{len(t)}")
def test_long_input_roundtrip(small_tokenizer: Tokenizer, text: str) -> None:
    assert small_tokenizer.decode(small_tokenizer.encode(text)) == text


@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("text", LONG_INPUTS, ids=lambda t: f"{t[:3]!r}x{len(t)}")
def test_long_input_gpt2(gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer, text: str) -> None:
    assert gpt2_tokenizer.encode(text) == gpt2.encode(text)


@pytest.mark.slow
def test_train_on_long_input_without_spaces() -> None:
    # Not periodic: a repeated unit collapses after a few dozen merges and runs out of pairs.
    text = "".join(random.Random(0).choices(string.ascii_lowercase, k=100_000))
    vocab, merges = train_bpe(text, vocab_size=300)
    assert len(vocab) == 300
    assert len(set(merges)) == len(merges)


# --- emoji / CJK ---------------------------------------------------------------------


@pytest.mark.parametrize("text", UNICODE_INPUTS)
def test_unicode_roundtrip(small_tokenizer: Tokenizer, text: str) -> None:
    assert small_tokenizer.decode(small_tokenizer.encode(text)) == text


@pytest.mark.oracle
@pytest.mark.parametrize("text", UNICODE_INPUTS)
def test_unicode_gpt2(gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer, text: str) -> None:
    assert gpt2_tokenizer.encode(text) == gpt2.encode(text)


# --- only special tokens -------------------------------------------------------------


def test_only_special_tokens(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.encode(EOT * 3) == [eot, eot, eot]
    assert small_tokenizer.decode([eot, eot, eot]) == EOT * 3


@pytest.mark.oracle
def test_only_special_tokens_gpt2(gpt2_tokenizer: Tokenizer) -> None:
    assert gpt2_tokenizer.encode(EOT * 3) == [50256, 50256, 50256]
