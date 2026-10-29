"""Differential tests: the incremental trainer must produce exactly the naive merge list.

The strategies lean on tiny alphabets and long runs or repeated motifs, where pairs
overlap (``aaaa``, ``abab``, ``aabaa``) and an incremental update is easiest to get wrong.
Every case asks for more merges than can happen, so training runs until no pairs are left.
"""

from bpe_strategies import overlapping_words, piece_counts
from hypothesis import example, given
from hypothesis import strategies as st
from naive_bpe import naive_merges, naive_train_bpe

from fastbpe import train_bpe
from fastbpe.train import _train_merges

EOT = "<|endoftext|>"
ALL_MERGES = 10_000


@given(piece_counts(overlapping_words()))
@example({b"aaa": 1})
@example({b"aaaa": 1})
@example({b"aaaaa": 1})
@example({b"abab": 1})
@example({b"ababa": 1})
@example({b"aabaa": 1})
@example({b"aaa": 1, b"aaaa": 1, b"aaaaa": 1})
@example({b"a" * 1000: 1})
@example({b"ab" * 500: 1})
@example({b"ab": 1, b"ba": 1})  # every pair ties
def test_merges_match_naive_on_overlapping_words(counts: dict[bytes, int]) -> None:
    assert _train_merges(counts, ALL_MERGES) == naive_merges(counts, ALL_MERGES)


@given(
    st.dictionaries(
        st.text("ab", min_size=200, max_size=3000).map(str.encode),
        st.integers(1, 3),
        min_size=1,
        max_size=2,
    )
)
def test_merges_match_naive_on_long_words(counts: dict[bytes, int]) -> None:
    assert _train_merges(counts, ALL_MERGES) == naive_merges(counts, ALL_MERGES)


@given(
    piece_counts(st.binary(min_size=1, max_size=20)),
    st.integers(0, 60),
)
def test_merges_match_naive_on_arbitrary_bytes(counts: dict[bytes, int], n_merges: int) -> None:
    # Also stops part-way through, not only when the pairs run out.
    assert _train_merges(counts, n_merges) == naive_merges(counts, n_merges)


@given(
    st.lists(st.sampled_from(["a", "b", "aa", "ab", " ", "\n", EOT]), max_size=200).map("".join),
    st.integers(256 + 1, 256 + 1 + 100),
)
def test_train_bpe_matches_naive(text: str, vocab_size: int) -> None:
    assert train_bpe(text, vocab_size, [EOT]) == naive_train_bpe(text, vocab_size, [EOT])


@given(st.text(max_size=300), st.integers(256, 256 + 100))
def test_train_bpe_matches_naive_on_any_text(text: str, vocab_size: int) -> None:
    assert train_bpe(text, vocab_size) == naive_train_bpe(text, vocab_size)
