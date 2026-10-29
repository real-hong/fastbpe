"""Rank-based BPE encoding of single pre-tokenized pieces."""

import heapq


def bpe_parts(piece: bytes, ranks: dict[bytes, int]) -> list[bytes]:
    """Split ``piece`` into the tokens BPE ends with, merging lowest ranks first.

    Equal ranks merge leftmost first, as tiktoken does. Runs in O(n log n): tokens form
    a linked list keyed by their start offset in ``piece``. ``end_of[i]`` is where the
    token starting at ``i`` ends (-1 once it has been merged into its left neighbour)
    and ``start_before[i]`` is where the previous token starts. The heap holds candidate
    merges ``(rank, start, mid, end)`` of ``piece[start:mid]`` with ``piece[mid:end]``;
    an entry is stale once either of its two tokens has changed.
    """
    n = len(piece)
    end_of = list(range(1, n + 1))
    start_before = list(range(-1, n - 1))
    # The initial pairs are known up front: one O(n) heapify instead of n pushes.
    heap = [
        (rank, i, i + 1, i + 2)
        for i in range(n - 1)
        if (rank := ranks.get(piece[i : i + 2])) is not None
    ]
    heapq.heapify(heap)

    # After a merge the new token pairs with its left and right neighbours. Both pushes
    # are written out instead of calling a helper: the Python call per push made the
    # whole function 7-11% slower on the distinct pieces of 1 MB inputs.
    while heap:
        _, start, mid, end = heapq.heappop(heap)
        if end_of[start] != mid or end_of[mid] != end:
            continue
        end_of[start] = end
        end_of[mid] = -1
        if start > 0:
            left = start_before[start]
            if (rank := ranks.get(piece[left:end])) is not None:
                heapq.heappush(heap, (rank, left, start, end))
        if end < n:
            start_before[end] = start
            right = end_of[end]
            if (rank := ranks.get(piece[start:right])) is not None:
                heapq.heappush(heap, (rank, start, end, right))

    parts = []
    i = 0
    while i < n:
        parts.append(piece[i : end_of[i]])
        i = end_of[i]
    return parts


def encode_bytes(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    """Encode one piece by repeatedly merging the adjacent pair with the lowest rank."""
    return [ranks[part] for part in bpe_parts(piece, ranks)]
