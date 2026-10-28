import random
import string
from collections.abc import Callable
from typing import Any

from pytest_benchmark.fixture import BenchmarkFixture

from fastbpe import Tokenizer, train_bpe
from fastbpe.pretokenize import count_pretokens

TRAIN_VOCAB_SIZE = 1000
ROUNDS = 3


def test_train(benchmark: BenchmarkFixture, train_corpus: str, separator: str) -> None:
    benchmark.pedantic(train_bpe, args=(train_corpus, TRAIN_VOCAB_SIZE, [separator]), rounds=ROUNDS)


def test_count_pretokens(benchmark: BenchmarkFixture, train_corpus: str) -> None:
    benchmark.pedantic(count_pretokens, args=([train_corpus],), rounds=ROUNDS)


PedanticArgs = tuple[tuple[Any, ...], dict[str, Any]]


def cold_encode(tokenizer: Tokenizer, text: str) -> Callable[[], PedanticArgs]:
    """pedantic() setup: a fresh tokenizer per round, so no round reuses another's cache."""

    def setup() -> PedanticArgs:
        fresh = Tokenizer(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)
        return (fresh, text), {}

    return setup


def test_encode_gpt2(
    benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, encode_corpus: str
) -> None:
    benchmark.pedantic(
        Tokenizer.encode, setup=cold_encode(gpt2_tokenizer, encode_corpus), rounds=ROUNDS
    )


def test_encode_gpt2_long_piece(benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer) -> None:
    # One 20 kB pre-token: random letters, so it holds many distinct mergeable pairs.
    text = "".join(random.Random(0).choices(string.ascii_lowercase, k=20_000))
    benchmark.pedantic(Tokenizer.encode, setup=cold_encode(gpt2_tokenizer, text), rounds=ROUNDS)
