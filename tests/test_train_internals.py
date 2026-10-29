"""White-box checks of the trainer's state after every merge.

The checks run here only, by driving the merge loop step by step; ``_train_merges`` has
no hooks for them.
"""

import random
from collections import Counter

from bpe_strategies import overlapping_words, piece_counts
from hypothesis import example, given
from hypothesis import strategies as st

from fastbpe.train import _DEAD, _NONE, Pair, _PairStats, _train_merges


def recount(stats: _PairStats) -> tuple[Counter[Pair], dict[Pair, set[int]]]:
    """Pair counts and positions recomputed from the linked lists."""
    counts: Counter[Pair] = Counter()
    positions: dict[Pair, set[int]] = {}
    tokens, next_ = stats.tokens, stats.next
    for start in range(len(tokens)):
        if stats.prev[start] != _NONE or tokens[start] == _DEAD:
            continue
        node = start
        while next_[node] != _NONE:
            after = next_[node]
            assert stats.prev[after] == node
            assert stats.weight[after] == stats.weight[node]
            pair = (tokens[node], tokens[after])
            counts[pair] += stats.weight[node]
            positions.setdefault(pair, set()).add(node)
            node = after
    return counts, positions


def check(stats: _PairStats) -> None:
    counts, positions = recount(stats)
    # Pairs whose count reached 0 must be gone from both, or len(counts) overstates the
    # live pairs and the heap rebuild (len(heap) > 3 * len(counts)) never fires.
    assert all(count > 0 for count in stats.counts.values())
    assert stats.counts == counts
    assert stats.positions == positions
    in_heap = {(entry.pair, entry.count) for entry in stats._heap}
    assert set(stats.counts.items()) <= in_heap
    assert len(stats._heap) <= 3 * len(stats.counts)


def train_checked(pieces: dict[bytes, int]) -> tuple[list[Pair], int]:
    """Train to exhaustion, checking the state after each merge; also count rebuilds."""
    stats = _PairStats()
    stats.add_words(pieces)
    check(stats)
    order: list[Pair] = []
    rebuilds = 0
    while (best := stats.pop_best()) is not None:
        heap = stats._heap
        stats.merge(best)
        assert best not in stats.counts
        assert best not in stats.positions
        rebuilds += stats._heap is not heap
        check(stats)
        order.append(best)
    return order, rebuilds


def unique(order: list[Pair]) -> list[Pair]:
    return list(dict.fromkeys(order))


@given(piece_counts(overlapping_words()))
@example({b"aaa": 1})
@example({b"aaaa": 2})
@example({b"aabaa": 1, b"abab": 3})
def test_state_is_consistent_after_every_merge(pieces: dict[bytes, int]) -> None:
    order, _ = train_checked(pieces)
    assert unique(order) == _train_merges(pieces, 10_000)


@given(piece_counts(st.binary(min_size=1, max_size=20)))
def test_state_is_consistent_on_arbitrary_bytes(pieces: dict[bytes, int]) -> None:
    train_checked(pieces)


def test_merged_pair_leaves_counts_and_positions() -> None:
    stats = _PairStats()
    stats.add_words({b"aaa": 1})
    assert stats.counts == {(b"a", b"a"): 2}
    stats.merge((b"a", b"a"))
    assert stats.counts == {(b"aa", b"a"): 1}
    assert stats.positions == {(b"aa", b"a"): {0}}


def test_heap_is_rebuilt_and_bounded_on_a_long_word() -> None:
    word = "".join(random.Random(0).choices("abcdefghij", k=3000)).encode()
    _, rebuilds = train_checked({word: 1})
    assert rebuilds > 0
