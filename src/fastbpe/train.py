"""Byte-level BPE training. Naive on purpose: all pair counts are recomputed every round."""

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from fastbpe.pretokenize import count_pretokens, split_on_special_tokens

Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def _count_pairs(word_counts: Counter[Word]) -> Counter[Pair]:
    pair_counts: Counter[Pair] = Counter()
    for word, count in word_counts.items():
        for pair in pairwise(word):
            pair_counts[pair] += count
    return pair_counts


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


def _apply_merge(word_counts: Counter[Word], pair: Pair) -> Counter[Word]:
    merged: Counter[Word] = Counter()
    for word, count in word_counts.items():
        merged[_merge_word(word, pair)] += count
    return merged


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
    word_counts: Counter[Word] = Counter()
    for piece, count in count_pretokens(chunks).items():
        word_counts[tuple(bytes([b]) for b in piece)] += count

    merges: list[Pair] = []
    while len(merges) < n_merges:
        pair_counts = _count_pairs(word_counts)
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        # A pair can come back after being merged when a later merge rebuilds one of its
        # sides from different parts. Merge it again, but it is not a new vocab entry.
        if best not in merges:
            merges.append(best)
        word_counts = _apply_merge(word_counts, best)

    vocab = {i: bytes([i]) for i in range(256)}
    for left, right in merges:
        vocab[len(vocab)] = left + right
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    return vocab, merges
