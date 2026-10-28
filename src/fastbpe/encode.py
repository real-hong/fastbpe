"""Rank-based BPE encoding of single pre-tokenized pieces."""

from itertools import pairwise


def _lowest_rank(parts: list[bytes], ranks: dict[bytes, int]) -> int | None:
    pair_ranks = [ranks.get(left + right) for left, right in pairwise(parts)]
    return min((rank for rank in pair_ranks if rank is not None), default=None)


def _merge_rank(parts: list[bytes], ranks: dict[bytes, int], rank: int) -> list[bytes]:
    """Merge, left to right, the adjacent pairs whose concatenation has ``rank``.

    This is the same as merging the leftmost lowest-ranked pair one at a time, as
    tiktoken does, as long as no merge forms a new neighbouring pair ranked below
    ``rank``. When one does, stop there so the caller merges that pair next.
    """
    merged: list[bytes] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and ranks.get(parts[i] + parts[i + 1]) == rank:
            token = parts[i] + parts[i + 1]
            merged.append(token)
            i += 2
            left = ranks.get(merged[-2] + token) if len(merged) > 1 else None
            right = ranks.get(token + parts[i]) if i < len(parts) else None
            if any(r is not None and r < rank for r in (left, right)):
                return merged + parts[i:]
        else:
            merged.append(parts[i])
            i += 1
    return merged


def bpe_parts(piece: bytes, ranks: dict[bytes, int]) -> list[bytes]:
    """Split ``piece`` into the tokens BPE ends with, merging lowest ranks first."""
    parts = [bytes([b]) for b in piece]
    while (rank := _lowest_rank(parts, ranks)) is not None:
        parts = _merge_rank(parts, ranks, rank)
    return parts


def encode_bytes(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    """Encode one piece by repeatedly merging the adjacent pair with the lowest rank."""
    return [ranks[part] for part in bpe_parts(piece, ranks)]
