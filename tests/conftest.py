import os

import pytest
import tiktoken
from hypothesis import settings

from fastbpe import Tokenizer

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


# Slow tests are opt-in, as in the pytest docs: CI passes --runslow.
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


EOT = "<|endoftext|>"

SMALL_CORPUS = """\
The quick brown fox jumps over the lazy dog. The dog didn't care; it's been there before.
Numbers like 123, 4567 and 89 show up too, as do symbols: (a + b) * c == d!
<|endoftext|>
Tokenizers split text into pieces. They'll merge frequent pairs until the vocabulary is full.
你好，世界。Emoji too: 👍🏽 🌍
"""


@pytest.fixture(scope="session")
def small_corpus() -> str:
    return SMALL_CORPUS


@pytest.fixture(scope="session")
def small_tokenizer() -> Tokenizer:
    return Tokenizer.train(SMALL_CORPUS, vocab_size=300, special_tokens=[EOT])


@pytest.fixture(scope="session")
def gpt2() -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as exc:  # e.g. offline on the first run, before the data is cached
        # CI sets FASTBPE_REQUIRE_ORACLE so that a failed download cannot pass as skips.
        if os.environ.get("FASTBPE_REQUIRE_ORACLE"):
            pytest.fail(f"tiktoken gpt2 encoding unavailable: {exc}")
        pytest.skip(f"tiktoken gpt2 encoding unavailable: {exc}")


@pytest.fixture(scope="session")
def gpt2_tokenizer(gpt2: tiktoken.Encoding) -> Tokenizer:
    return Tokenizer.from_ranks(gpt2._mergeable_ranks, gpt2._special_tokens)
