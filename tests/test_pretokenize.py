from collections import Counter

import pytest

from fastbpe.pretokenize import count_pretokens, pretokenize, split_on_special_tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I'm here", ["I", "'m", " here"]),
        ("they'll've", ["they", "'ll", "'ve"]),
        ("it's we're I'd don't", ["it", "'s", " we", "'re", " I", "'d", " don", "'t"]),
        ("hello world 123 !!", ["hello", " world", " 123", " !!"]),
        ("abc123", ["abc", "123"]),
        ("a  b", ["a", " ", " b"]),
        ("a   ", ["a", "   "]),
        ("a\n\nb", ["a", "\n", "\n", "b"]),
        ("x\t\ty", ["x", "\t", "\t", "y"]),
        ("héllo wörld", ["héllo", " wörld"]),
        ("你好世界 ok", ["你好世界", " ok"]),
        ("١٢٣ ٤", ["١٢٣", " ٤"]),
    ],
)
def test_pretokenize(text: str, expected: list[str]) -> None:
    assert pretokenize(text) == expected


def test_pretokenize_empty() -> None:
    assert pretokenize("") == []


def test_count_pretokens() -> None:
    assert count_pretokens(["the cat", "the dog"]) == Counter({b"the": 2, b" cat": 1, b" dog": 1})


def test_count_pretokens_encodes_utf8() -> None:
    assert count_pretokens(["héllo héllo"]) == Counter({"héllo".encode(): 1, " héllo".encode(): 1})


def test_count_pretokens_parallel_matches_serial() -> None:
    texts = [f"line {i}: the quick brown fox #{i % 7}" for i in range(1000)]
    assert count_pretokens(texts, workers=4) == count_pretokens(texts, workers=1)


def test_empty_special_token_is_rejected() -> None:
    # An empty alternative matches between every pair of characters.
    with pytest.raises(ValueError, match="empty"):
        split_on_special_tokens("ab cd", ["<|endoftext|>", ""])
