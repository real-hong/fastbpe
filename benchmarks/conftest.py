import sys
from pathlib import Path

import pytest
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus import SEPARATOR, load_corpus

from fastbpe import Tokenizer


@pytest.fixture(scope="session")
def train_corpus() -> str:
    return load_corpus(1_000_000)


@pytest.fixture(scope="session")
def encode_corpus() -> str:
    return load_corpus(1_000_000)


@pytest.fixture(scope="session")
def gpt2_tokenizer() -> Tokenizer:
    enc = tiktoken.get_encoding("gpt2")
    return Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)


@pytest.fixture(scope="session")
def separator() -> str:
    return SEPARATOR
