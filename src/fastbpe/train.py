"""Byte-level BPE training with incremental pair-count updates.

Pair counts are kept up to date across rounds, and an inverted index from each pair to
the words containing it means a merge only rewrites the words it actually touches.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from fastbpe.pretokenize import count_pretokens, split_on_special_tokens

Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


class _PairStats:
    """Pair frequencies over a set of weighted words, plus which words hold each pair."""

    def __init__(self) -> None:
        self.counts: dict[Pair, int] = {}
        self.words: defaultdict[Pair, set[int]] = defaultdict(set)

    def add(self, index: int, word: Word, count: int) -> None:
        for pair in pairwise(word):
            self.counts[pair] = self.counts.get(pair, 0) + count
            self.words[pair].add(index)

    def remove(self, index: int, word: Word, count: int) -> None:
        for pair in pairwise(word):
            self.counts[pair] -= count
            if self.counts[pair] == 0:
                del self.counts[pair]
            holders = self.words.get(pair)
            if holders is not None:
                holders.discard(index)
                if not holders:
                    del self.words[pair]


def _merge_word(word: Word, pair: Pair) -> Word:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    text: str, vocab_size: int, special_tokens: Sequence[str] = ()
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary on ``text``.

    Ids 0-255 are the single bytes, then one id per merge in merge order, then the
    special tokens in the order given. Any pair occurring at least once may be merged;
    ties on frequency go to the lexicographically larger pair of byte strings. Special
    token text never takes part in pair counting. Training stops early when no pairs
    are left. Raises ``ValueError`` if ``vocab_size < 256 + len(special_tokens)``.
    """
    n_merges = vocab_size - 256 - len(special_tokens)
    if n_merges < 0:
        raise ValueError(
            f"vocab_size must be at least {256 + len(special_tokens)} "
            f"(256 bytes + {len(special_tokens)} special tokens), got {vocab_size}"
        )

    specials = set(special_tokens)
    chunks = [c for c in split_on_special_tokens(text, special_tokens) if c not in specials]
    piece_counts = count_pretokens(chunks)
    words: list[Word] = [tuple(bytes([b]) for b in piece) for piece in piece_counts]
    word_counts = list(piece_counts.values())

    stats = _PairStats()
    for index, word in enumerate(words):
        stats.add(index, word, word_counts[index])

    merges: list[Pair] = []
    merged_pairs: set[Pair] = set()
    while len(merges) < n_merges and stats.counts:
        best = max(stats.counts, key=lambda pair: (stats.counts[pair], pair))
        # A pair can come back after being merged when a later merge rebuilds one of its
        # sides from different parts. Merge it again, but it is not a new vocab entry.
        if best not in merged_pairs:
            merged_pairs.add(best)
            merges.append(best)
        # Merging every occurrence left to right leaves no `best` pair behind, so its
        # index entry can go; remove() skips pairs that are no longer indexed.
        for index in stats.words.pop(best):
            stats.remove(index, words[index], word_counts[index])
            words[index] = _merge_word(words[index], best)
            stats.add(index, words[index], word_counts[index])

    vocab = {i: bytes([i]) for i in range(256)}
    for left, right in merges:
        vocab[len(vocab)] = left + right
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    return vocab, merges
