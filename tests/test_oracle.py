"""Encoding must match tiktoken's gpt2 encoding exactly when using the same merges."""

import pytest
import tiktoken
from hypothesis import given
from hypothesis import strategies as st

from fastbpe import Tokenizer

pytestmark = pytest.mark.oracle

# Characters assigned in this Python's Unicode database. Code points assigned only in newer
# Unicode versions are letters/numbers to ``regex`` and tiktoken but unknown to the stdlib.
assigned_text = st.text(st.characters(exclude_categories=("Cs", "Cn")))

CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll've done it; it's what we're here for. Don't you think I'd know?",
    "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "Pi is 3.14159, e is 2.71828 and 1,000,000 is a million.",
    "lots   of    spaces\t\ttabs\n\n\nand newlines   ",
    "   leading and trailing   ",
    "emoji: 👍🏽 👨‍👩‍👧‍👦 🇨🇳 🌍",
    "中文分词测试：你好，世界！日本語のテキストも。한국어도.",
    "mixed hello世界 café naïve Ünïcödé",
    "special <|endoftext|> in the middle<|endoftext|>",
]


@pytest.mark.parametrize("text", CORPUS)
def test_matches_tiktoken(gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer, text: str) -> None:
    assert gpt2_tokenizer.encode(text) == gpt2.encode(text, allowed_special="all")


@given(assigned_text)
def test_matches_tiktoken_on_arbitrary_text(
    gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer, text: str
) -> None:
    assert gpt2_tokenizer.encode(text) == gpt2.encode(text, allowed_special="all")


def test_decode_matches_tiktoken(gpt2: tiktoken.Encoding, gpt2_tokenizer: Tokenizer) -> None:
    ids = gpt2.encode("\n".join(CORPUS), allowed_special="all")
    assert gpt2_tokenizer.decode_bytes(ids) == gpt2.decode_bytes(ids)
