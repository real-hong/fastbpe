"""Hypothesis strategies for training inputs, shared by the oracle and internals tests."""

from hypothesis import strategies as st

MOTIFS = ["ab", "ba", "aab", "aba", "abb", "aabaa", "abab", "abc"]


@st.composite
def overlapping_words(draw: st.DrawFn) -> bytes:
    """A word of runs (``aaaa``) and repeated motifs (``ababab``) over a/b, sometimes c."""
    alphabet = draw(st.sampled_from(["ab", "ab", "abc"]))
    runs = st.builds(lambda c, n: c * n, st.sampled_from(alphabet), st.integers(1, 30))
    motifs = st.builds(lambda m, n: m * n, st.sampled_from(MOTIFS), st.integers(1, 15))
    parts = draw(st.lists(st.one_of(runs, motifs), min_size=1, max_size=6))
    return "".join(parts).encode()


def piece_counts(words: st.SearchStrategy[bytes]) -> st.SearchStrategy[dict[bytes, int]]:
    return st.dictionaries(words, st.integers(1, 5), min_size=1, max_size=6)
