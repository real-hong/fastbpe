"""Byte-level BPE training with incremental pair-count updates.

All words are stored as one doubly linked list of tokens (word boundaries end the
links), with an index from each pair to the nodes where it starts. Merging a pair
visits only its occurrences and updates the pairs next to each one, so a round costs
O(k log k) for k occurrences, independent of word length. The most frequent pair comes
from a max-heap with lazy deletion: a pair is pushed again only when a round changed its
count, and the heap is rebuilt when outdated entries outnumber live pairs 2 to 1.
"""

import heapq
import warnings
from array import array
from collections.abc import Mapping, Sequence
from multiprocessing.context import BaseContext

from fastbpe.pretokenize import count_pretokens, split_on_special_tokens

Pair = tuple[bytes, bytes]

_BYTES = [bytes([b]) for b in range(256)]
_DEAD = b""  # token of a node absorbed into its left neighbour; equals no real token
_NONE = -1  # prev/next link past either end of a word


class _Candidate:
    """Heap entry. heapq pops the smallest item, so "less than" means "more frequent,
    then lexicographically larger pair" -- the order max() used before the heap."""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: Pair) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_Candidate") -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


class _PairStats:
    """Words as linked token lists, pair counts, pair positions and the candidate heap.

    Invariants between merges (checked by tests/test_train_internals.py):
    ``counts`` holds exactly the pairs adjacent somewhere, each with the summed weight
    of its occurrences (overlaps included: aaa holds (a, a) twice), never 0;
    ``positions[pair]`` is the set of nodes where it starts; every pair in ``counts``
    has a heap entry with its current count.
    """

    def __init__(self) -> None:
        self.counts: dict[Pair, int] = {}
        self.positions: dict[Pair, set[int]] = {}
        self.tokens: list[bytes] = []
        # Machine-int arrays: a list would hold one int object per link (~25% more peak
        # memory on long words) for no measurable speed gain.
        self.prev = array("q")
        self.next = array("q")
        self.weight = array("q")  # count of the word the node belongs to
        self._heap: list[_Candidate] = []

    def add_words(self, piece_counts: Mapping[bytes, int]) -> None:
        tokens, prev, next_, weight = self.tokens, self.prev, self.next, self.weight
        counts, positions = self.counts, self.positions
        for piece, count in piece_counts.items():
            start, end = len(tokens), len(tokens) + len(piece)
            tokens.extend([_BYTES[b] for b in piece])
            prev.append(_NONE)
            prev.extend(range(start, end - 1))
            next_.extend(range(start + 1, end))
            next_.append(_NONE)
            weight.extend([count] * len(piece))
            for node in range(start, end - 1):
                pair = (tokens[node], tokens[node + 1])
                counts[pair] = counts.get(pair, 0) + count
                positions.setdefault(pair, set()).add(node)
        self._heap = [_Candidate(count, pair) for pair, count in counts.items()]
        heapq.heapify(self._heap)

    def pop_best(self) -> Pair | None:
        """Return the most frequent pair (ties: the larger pair), or None if none is left."""
        while self._heap:
            top = heapq.heappop(self._heap)
            if self.counts.get(top.pair) == top.count:
                return top.pair
        return None

    def _sub(self, pair: Pair, node: int, weight: int, changed: dict[Pair, int]) -> None:
        count = self.counts[pair] - weight
        if count:
            self.counts[pair] = count
        else:
            del self.counts[pair]
        holders = self.positions.get(pair)  # None for the pair being merged (popped)
        if holders is not None:
            holders.discard(node)
            if not holders:
                del self.positions[pair]
        changed[pair] = changed.get(pair, 0) - weight

    def _add(self, pair: Pair, node: int, weight: int, changed: dict[Pair, int]) -> None:
        self.counts[pair] = self.counts.get(pair, 0) + weight
        self.positions.setdefault(pair, set()).add(node)
        changed[pair] = changed.get(pair, 0) + weight

    def merge(self, best: Pair) -> None:
        """Merge every occurrence of ``best``, left to right within each word.

        At each occurrence x A B y -> x AB y, only (x, A), (A, B) and (B, y) lose one
        occurrence and (x, AB) and (AB, y) gain one.
        """
        left, right = best
        new = left + right
        tokens, prev, next_, weight = self.tokens, self.prev, self.next, self.weight
        changed: dict[Pair, int] = {}
        # Node ids grow left to right within a word, so sorting merges overlapping
        # occurrences leftmost first: aaa -> (aa) a, aaaa -> (aa) (aa).
        for i in sorted(self.positions.pop(best)):
            # Absorbed by the previous occurrence in this loop (the middle a of aaa).
            if tokens[i] != left:
                continue
            j = next_[i]
            w = weight[i]
            x, y = prev[i], next_[j]
            if x != _NONE:
                self._sub((tokens[x], left), x, w, changed)
                self._add((tokens[x], new), x, w, changed)
            self._sub(best, i, w, changed)
            if y != _NONE:
                self._sub((right, tokens[y]), j, w, changed)
                self._add((new, tokens[y]), i, w, changed)
                prev[y] = i
            tokens[i] = new
            tokens[j] = _DEAD
            next_[i] = y

        counts, heap = self.counts, self._heap
        for pair, delta in changed.items():
            if delta and pair in counts:
                heapq.heappush(heap, _Candidate(counts[pair], pair))
        # Bound the heap: rebuild once outdated entries outnumber live pairs 2 to 1.
        # Rebuilding costs O(live pairs), so it stays amortised O(1) per push.
        if len(heap) > 3 * len(counts):
            self._heap = [_Candidate(count, pair) for pair, count in counts.items()]
            heapq.heapify(self._heap)


def _train_merges(piece_counts: Mapping[bytes, int], n_merges: int) -> list[Pair]:
    """Up to ``n_merges`` merges learned from pre-token byte strings and their counts."""
    stats = _PairStats()
    stats.add_words(piece_counts)
    merges: list[Pair] = []
    # Every merge builds a token that did not exist before, so a merged pair never comes
    # back either. Wherever a token forms, no merge crossed its edges (tokens are never
    # split), so the merges inside it, overlaps resolved leftmost first, depend only on
    # its own bytes: it forms in every word at the same step, from the same pair. A
    # merge that rebuilt existing bytes would mean the pair statistics are broken.
    known = set(_BYTES)
    while len(merges) < n_merges and (best := stats.pop_best()) is not None:
        token = best[0] + best[1]
        if token in known:
            raise RuntimeError(f"internal error: merge {best!r} rebuilds existing token {token!r}")
        known.add(token)
        merges.append(best)
        stats.merge(best)
    return merges


def train_bpe(
    text: str,
    vocab_size: int,
    special_tokens: Sequence[str] = (),
    *,
    workers: int = 1,
    mp_context: BaseContext | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary on ``text``.

    Ids 0-255 are the single bytes, then one id per merge in merge order, then the
    special tokens in the order given. Any pair occurring at least once may be merged;
    ties on frequency go to the lexicographically larger pair of byte strings. Special
    token text never takes part in pair counting. Training stops early when no pairs
    are left. Raises ``ValueError`` if ``vocab_size < 256 + len(special_tokens)``.

    Regular tokens (bytes and merges) have distinct bytes. Special tokens are a separate
    namespace, as in tiktoken: a single-byte special token such as ``"a"`` shares its
    bytes with a base token and both ids are kept. Since every occurrence of that byte
    is cut out of the text as the special token, the byte never takes part in merges;
    ``train_bpe`` warns (``UserWarning``) about it and trains as usual.

    ``workers > 1`` (opt-in) pre-tokenizes in that many processes started with
    ``mp_context`` (default forkserver, else spawn), one batch of text between special
    tokens at a time; text without special tokens cannot be split this way. Scripts that
    use it need an ``if __name__ == "__main__":`` guard.
    """
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError(f"duplicate special tokens: {list(special_tokens)}")
    n_merges = vocab_size - 256 - len(special_tokens)
    if n_merges < 0:
        raise ValueError(
            f"vocab_size must be at least {256 + len(special_tokens)} "
            f"(256 bytes + {len(special_tokens)} special tokens), got {vocab_size}"
        )

    for token in special_tokens:
        if len(token.encode()) == 1:
            warnings.warn(
                f"special token {token!r} is a single byte: that byte is cut out of the "
                "training text and will not take part in merges",
                UserWarning,
                stacklevel=2,
            )

    specials = set(special_tokens)
    chunks = [c for c in split_on_special_tokens(text, special_tokens) if c not in specials]
    piece_counts = count_pretokens(chunks, workers=workers, mp_context=mp_context)
    merges = _train_merges(piece_counts, n_merges)

    vocab = {i: bytes([i]) for i in range(256)}

    for left, right in merges:
        vocab[len(vocab)] = left + right
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    return vocab, merges
