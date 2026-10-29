"""Fixed per-call cost of ``Tokenizer.encode`` with many special tokens.

``encode_short``: a short text with a warm piece cache, so the time is what ``encode``
spends around the pieces: choosing the allowed special tokens and splitting on them.
``construct`` and ``first_encode`` time building a tokenizer and its first encode, with
the ``re`` cache emptied so that no special-token pattern is compiled already. Each
case runs with 1 and with 1000 special tokens (GPT-2 ranks plus ``<|special_i|>``).
"""

import re
from functools import cache
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from fastbpe import Tokenizer

TEXT = "hello world"
SUBSET = 10


def specials(n: int) -> list[str]:
    return [f"<|special_{i}|>" for i in range(n)]


@cache
def tokenizer_with(gpt2: Tokenizer, n: int) -> Tokenizer:
    extra = {s: gpt2.vocab_size + i for i, s in enumerate(specials(n))}
    return Tokenizer(gpt2.vocab, gpt2.merges, gpt2.special_tokens | extra)


ALLOWED = {
    "all": lambda n: "all",
    "subset": lambda n: set(specials(min(n, SUBSET))),
    "none": lambda n: set(),
}


@pytest.mark.parametrize("allowed", ALLOWED)
@pytest.mark.parametrize("n", [1, 1000])
def test_encode_short(
    benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, n: int, allowed: str
) -> None:
    tokenizer = tokenizer_with(gpt2_tokenizer, n)
    allowed_special = ALLOWED[allowed](n)
    tokenizer.encode(TEXT, allowed_special=allowed_special)
    benchmark(tokenizer.encode, TEXT, allowed_special=allowed_special)


@pytest.mark.parametrize("n", [1, 1000])
def test_construct(benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, n: int) -> None:
    tokenizer = tokenizer_with(gpt2_tokenizer, n)
    args = (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)

    def setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        re.purge()
        return args, {}

    benchmark.pedantic(Tokenizer, setup=setup, rounds=20)


@pytest.mark.parametrize("n", [1, 1000])
def test_first_encode(benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, n: int) -> None:
    tokenizer = tokenizer_with(gpt2_tokenizer, n)
    args = (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)

    def setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        fresh = Tokenizer(*args)
        re.purge()
        return (fresh, TEXT), {}

    benchmark.pedantic(Tokenizer.encode, setup=setup, rounds=20)
