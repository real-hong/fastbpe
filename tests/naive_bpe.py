"""Naive BPE training from 762305e, kept as the reference for tests and benchmarks.

Every round recounts every pair over all words, so it is slow but obviously correct.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
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


def naive_merges(piece_counts: Mapping[bytes, int], n_merges: int) -> list[Pair]:
    """Merges learned from pre-token byte strings and their counts (no pre-tokenizing)."""
    word_counts: Counter[Word] = Counter()
    for piece, count in piece_counts.items():
        word_counts[tuple(bytes([b]) for b in piece)] += count

    merges: list[Pair] = []
    while len(merges) < n_merges:
        pair_counts = _count_pairs(word_counts)
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        # Never true for a trained merge list: a pair cannot come back once merged (see
        # fastbpe.train._train_merges). Kept as written in 762305e, the reference.
        if best not in merges:
            merges.append(best)
        word_counts = _apply_merge(word_counts, best)
    return merges


def naive_train_bpe(
    text: str, vocab_size: int, special_tokens: Sequence[str] = ()
) -> tuple[dict[int, bytes], list[Pair]]:
    """Same contract as ``fastbpe.train_bpe`` (serial only)."""
    n_merges = vocab_size - 256 - len(special_tokens)
    if n_merges < 0:
        raise ValueError(f"vocab_size too small: {vocab_size}")
    specials = set(special_tokens)
    chunks = [c for c in split_on_special_tokens(text, special_tokens) if c not in specials]
    merges = naive_merges(count_pretokens(chunks), n_merges)

    vocab = {i: bytes([i]) for i in range(256)}
    for left, right in merges:
        vocab[len(vocab)] = left + right
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    return vocab, merges
