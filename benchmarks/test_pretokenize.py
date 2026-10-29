"""Pre-tokenization on the corpus and adversarial inputs, alone and inside train/encode.

Each case is about 1 MB of UTF-8 (``scripts/adversarial_inputs.py``) and runs four ways:

- ``train_pretok``: what ``train_bpe`` does before merging (split on special tokens,
  ``count_pretokens``), without the merges, which pre-tokenization does not touch.
- ``pretokenize``: ``pretokenize`` alone over the chunks between special tokens.
- ``encode_cold``: ``Tokenizer.encode`` with GPT-2 ranks and an empty piece cache.
- ``encode_warm``: the same tokenizer encoding the same text again (one untimed warm-up).
  Note: base64, low_rep and unicode_all are effectively cold here as well. Their
  distinct pieces (103,599, 118,253 and 135,180) exceed ``CACHE_SIZE`` (100k), so the
  cache is emptied during every encode (see TODO.md).

``encode_stream`` encodes ``adversarial_inputs.stream`` (4 MB) document by document with
one tokenizer, as a long-running encoder would: corpus documents twice over between
never-repeating filler, far more distinct pieces than the cache holds. It is the case
to judge cache policies by, not the warm 1 MB inputs, whose second pass is a cyclic scan
that favours whatever keeps the most entries. For comparisons between variants use
``scripts/time_variants.py`` (one process per measurement), not this session-wide run.

``pretokenize_cold_table`` times the first ``pretokenize`` call after the character
table is emptied, on inputs with many distinct code points.
"""

from functools import cache
from typing import Any

import pytest
from adversarial_inputs import INPUTS, many_specials, stream
from corpus import SEPARATOR
from pytest_benchmark.fixture import BenchmarkFixture

import fastbpe.pretokenize as pretokenize_module
from fastbpe import Tokenizer
from fastbpe.pretokenize import count_pretokens, pretokenize, split_on_special_tokens

SIZE = 1_000_000
ROUNDS = 3
CASES = [*INPUTS, "specials"]


@cache
def case(name: str) -> tuple[str, list[str]]:
    """The text and the special tokens it contains."""
    if name == "specials":
        return many_specials(SIZE)
    return INPUTS[name](SIZE), [SEPARATOR] if name == "regular" else []


@cache
def stream_docs() -> list[str]:
    return stream(4 * SIZE)


def encode_docs(tokenizer: Tokenizer, docs: list[str]) -> None:
    for doc in docs:
        tokenizer.encode(doc)


def chunks(name: str) -> list[str]:
    text, specials = case(name)
    return [c for c in split_on_special_tokens(text, specials) if c not in specials]


def train_pretok(text: str, specials: list[str]) -> None:
    count_pretokens([c for c in split_on_special_tokens(text, specials) if c not in specials])


def pretokenize_all(texts: list[str]) -> None:
    for text in texts:
        pretokenize(text)


def fresh_tokenizer(gpt2: Tokenizer, specials: list[str]) -> Tokenizer:
    extra = {s: gpt2.vocab_size + i for i, s in enumerate(specials) if s != SEPARATOR}
    return Tokenizer(gpt2.vocab, gpt2.merges, gpt2.special_tokens | extra)


def cold_table() -> None:
    """Empty the pre-tokenizer's lazily filled character table (none before finding #7)."""
    table = getattr(pretokenize_module, "_TABLE", None)
    if table is not None:
        table.clear()


@pytest.mark.parametrize("name", CASES)
def test_train_pretok(benchmark: BenchmarkFixture, name: str) -> None:
    benchmark.pedantic(train_pretok, args=case(name), rounds=ROUNDS)


@pytest.mark.parametrize("name", CASES)
def test_pretokenize(benchmark: BenchmarkFixture, name: str) -> None:
    benchmark.pedantic(pretokenize_all, args=(chunks(name),), rounds=ROUNDS)


@pytest.mark.parametrize("name", CASES)
def test_encode_cold(benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, name: str) -> None:
    text, specials = case(name)

    def setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (fresh_tokenizer(gpt2_tokenizer, specials), text), {}

    benchmark.pedantic(Tokenizer.encode, setup=setup, rounds=ROUNDS)


@pytest.mark.parametrize("name", CASES)
def test_encode_warm(benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, name: str) -> None:
    text, specials = case(name)
    tokenizer = fresh_tokenizer(gpt2_tokenizer, specials)
    benchmark.pedantic(tokenizer.encode, args=(text,), rounds=ROUNDS, warmup_rounds=1)


def test_encode_stream(benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer) -> None:
    docs = stream_docs()

    def setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (fresh_tokenizer(gpt2_tokenizer, []), docs), {}

    benchmark.pedantic(encode_docs, setup=setup, rounds=ROUNDS)


@pytest.mark.parametrize("name", ["cjk", "unicode_all"])
def test_pretokenize_cold_table(benchmark: BenchmarkFixture, name: str) -> None:
    texts = chunks(name)
    benchmark.pedantic(pretokenize_all, args=(texts,), setup=cold_table, rounds=ROUNDS)
