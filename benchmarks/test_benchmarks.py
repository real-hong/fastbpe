from pytest_benchmark.fixture import BenchmarkFixture

from fastbpe import Tokenizer, train_bpe
from fastbpe.pretokenize import count_pretokens

TRAIN_VOCAB_SIZE = 1000
ROUNDS = 3


def test_train(benchmark: BenchmarkFixture, train_corpus: str, separator: str) -> None:
    benchmark.pedantic(train_bpe, args=(train_corpus, TRAIN_VOCAB_SIZE, [separator]), rounds=ROUNDS)


def test_count_pretokens(benchmark: BenchmarkFixture, train_corpus: str) -> None:
    benchmark.pedantic(count_pretokens, args=([train_corpus],), rounds=ROUNDS)


def test_encode_gpt2(
    benchmark: BenchmarkFixture, gpt2_tokenizer: Tokenizer, encode_corpus: str
) -> None:
    benchmark.pedantic(gpt2_tokenizer.encode, args=(encode_corpus,), rounds=ROUNDS)
