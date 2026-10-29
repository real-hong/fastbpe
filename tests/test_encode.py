from hypothesis import given
from hypothesis import strategies as st
from tiktoken._educational import bpe_encode

from fastbpe.encode import encode_bytes

BYTE_RANKS = {bytes([i]): i for i in range(256)}


def test_no_merges_gives_byte_ids() -> None:
    assert encode_bytes(b"xyz", BYTE_RANKS) == [120, 121, 122]


def test_empty_piece() -> None:
    assert encode_bytes(b"", BYTE_RANKS) == []


def test_merges_until_no_ranked_pair_left() -> None:
    ranks = BYTE_RANKS | {b"ab": 256, b"bc": 257, b"abc": 258}
    assert encode_bytes(b"abc", ranks) == [258]


def test_lowest_rank_merges_first() -> None:
    ranks = BYTE_RANKS | {b"bc": 256, b"ab": 257}
    assert encode_bytes(b"abc", ranks) == [97, 256]


def test_equal_ranks_merge_left_to_right() -> None:
    ranks = BYTE_RANKS | {b"aa": 256}
    assert encode_bytes(b"aaa", ranks) == [256, 97]
    assert encode_bytes(b"aaaa", ranks) == [256, 256]
    assert encode_bytes(b"aaaa", ranks | {b"aaaa": 257}) == [257]


def test_new_lower_ranked_pair_is_merged_before_other_occurrences() -> None:
    # Merging the first "ab" forms "aba", which outranks the second "ab". Merging every
    # "ab" at once would give [ab, ab]; one-at-a-time (tiktoken) gives [aba, b].
    ranks = BYTE_RANKS | {b"ab": 300, b"aba": 256}
    assert encode_bytes(b"abab", ranks) == [256, 98]


# Over a three-letter alphabet, random rank tables produce every awkward case: equal
# ranks, merges that create lower-ranked pairs, and many overlapping candidates.
abc_bytes = st.lists(st.sampled_from(b"abc"), max_size=40).map(bytes)
abc_tokens = st.lists(st.sampled_from(b"abc"), min_size=2, max_size=4).map(bytes)


@given(piece=abc_bytes, tokens=st.lists(abc_tokens, unique=True, max_size=30), data=st.data())
def test_matches_reference_bpe(piece: bytes, tokens: list[bytes], data: st.DataObject) -> None:
    order = data.draw(st.permutations(range(len(tokens))))
    ranks = BYTE_RANKS | {token: 256 + rank for token, rank in zip(tokens, order, strict=True)}
    assert encode_bytes(piece, ranks) == bpe_encode(ranks, piece, visualise=None)
