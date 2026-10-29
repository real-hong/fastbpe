"""Hit rates of encode-cache eviction policies, simulated on the benchmark inputs.

    uv run python scripts/measure_cache.py

Each ~1 MB input is cut into pieces as ``Tokenizer.encode`` cuts it, and the piece
sequence is replayed through models of the eviction policies (keys only, no BPE), so
policy x capacity combinations take seconds. Reported per input: pieces, distinct
pieces, and for each policy the hit rate of a cold pass and of a second ("warm") pass of
the same text, plus how often the policy evicted.

Re-encoding the same text is a cyclic scan over its distinct pieces, the worst case for
LRU and insertion-order FIFO once the distinct pieces exceed the capacity. The
``stream`` case (``adversarial_inputs.stream``, 4 MB) is closer to a long-running
encoder: corpus documents in random order, twice over, between never-repeating base64
and random words; its hit rate counts only the corpus pieces, i.e. whether the long
tail flushes the hot pieces out.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator
from itertools import islice

from adversarial_inputs import INPUTS, many_specials, stream
from corpus import SEPARATOR

from fastbpe.pretokenize import pretokenize, split_on_special_tokens

MB = 1_000_000
CAPACITIES = [100_000, 1 << 17, 1 << 18]


def pieces(text: str, specials: list[str]) -> list[str]:
    allowed = set(specials)
    return [
        p
        for chunk in split_on_special_tokens(text, specials)
        if chunk and chunk not in allowed
        for p in pretokenize(chunk)
    ]


# A policy replays a piece sequence and yields hit (True) or miss per piece; it counts
# its evictions (whole clears or batch evictions) in stats["evictions"].
Policy = Callable[[list[str], int, dict[str, int]], Iterator[bool]]


def clear_all(seq: list[str], cap: int, stats: dict[str, int]) -> Iterator[bool]:
    cache: dict[str, None] = {}
    for p in seq:
        if p in cache:
            yield True
            continue
        if len(cache) >= cap:
            cache.clear()
            stats["evictions"] += 1
        cache[p] = None
        yield False


def evict_oldest(fraction: float) -> Policy:
    def policy(seq: list[str], cap: int, stats: dict[str, int]) -> Iterator[bool]:
        cache: dict[str, None] = {}
        k = max(1, int(cap * fraction))
        for p in seq:
            if p in cache:
                yield True
                continue
            if len(cache) >= cap:
                for key in list(islice(cache, k)):
                    del cache[key]
                stats["evictions"] += 1
            cache[p] = None
            yield False

    return policy


def thin(seq: list[str], cap: int, stats: dict[str, int]) -> Iterator[bool]:
    # Evict every other entry in insertion order: half of the old and half of the new.
    cache: dict[str, None] = {}
    for p in seq:
        if p in cache:
            yield True
            continue
        if len(cache) >= cap:
            for key in list(islice(cache, 0, None, 2)):
                del cache[key]
            stats["evictions"] += 1
        cache[p] = None
        yield False


def lru(seq: list[str], cap: int, stats: dict[str, int]) -> Iterator[bool]:
    cache: OrderedDict[str, None] = OrderedDict()
    for p in seq:
        if p in cache:
            cache.move_to_end(p)
            yield True
            continue
        if len(cache) >= cap:
            cache.popitem(last=False)
            stats["evictions"] += 1
        cache[p] = None
        yield False


def freeze(seq: list[str], cap: int, stats: dict[str, int]) -> Iterator[bool]:
    # Reference only: once full, new pieces are not cached at all.
    cache: set[str] = set()
    for p in seq:
        if p in cache:
            yield True
        else:
            if len(cache) < cap:
                cache.add(p)
            else:
                stats["evictions"] += 1
            yield False


POLICIES: dict[str, Policy] = {
    "clear": clear_all,
    "oldest1/2": evict_oldest(0.5),
    "oldest1/4": evict_oldest(0.25),
    "thin": thin,
    "lru": lru,
    "freeze": freeze,
}


def cold_warm(policy: Policy, seq: list[str], cap: int) -> tuple[float, float, int]:
    stats = {"evictions": 0}
    hits = list(policy(seq + seq, cap, stats))
    n = len(seq)
    return sum(hits[:n]) / n, sum(hits[n:]) / n, stats["evictions"]


def stream_pieces() -> tuple[list[str], list[bool]]:
    """``adversarial_inputs.stream``, the corpus documents' pieces marked as hot."""
    seq: list[str] = []
    hot: list[bool] = []
    for i, doc in enumerate(stream(4 * MB)):
        doc_pieces = pieces(doc, [])
        seq += doc_pieces
        hot += [i % 2 == 0] * len(doc_pieces)
    return seq, hot


def main() -> None:
    cases: dict[str, list[str]] = {}
    for name, make in INPUTS.items():
        text = make(MB)
        cases[name] = pieces(text, [SEPARATOR] if name == "regular" else [])
    text, specials = many_specials(MB)
    cases["specials"] = pieces(text, specials)

    print("hit rate cold / warm (second pass of the same text), evictions over both passes")
    for cap in CAPACITIES:
        print(f"\ncapacity {cap:,}")
        print(
            f"{'input':12} {'pieces':>8} {'distinct':>8} " + " ".join(f"{p:>18}" for p in POLICIES)
        )
        for name, seq in cases.items():
            cells = []
            for policy in POLICIES.values():
                cold, warm, ev = cold_warm(policy, seq, cap)
                cells.append(f"{cold:5.1%} {warm:6.1%} {ev:5}")
            print(
                f"{name:12} {len(seq):8} {len(set(seq)):8} " + " ".join(f"{c:>18}" for c in cells)
            )

    seq, hot = stream_pieces()
    n_hot = sum(hot)
    print(f"\nstream: {len(seq):,} pieces, {len(set(seq)):,} distinct, {n_hot:,} corpus pieces")
    print("hit rate on corpus pieces (and overall)")
    for cap in CAPACITIES:
        cells = []
        for name, policy in POLICIES.items():
            stats = {"evictions": 0}
            hits = list(policy(seq, cap, stats))
            hot_hits = sum(h for h, is_hot in zip(hits, hot, strict=True) if is_hot)
            cells.append(f"{name} {hot_hits / n_hot:5.1%} ({sum(hits) / len(seq):5.1%})")
        print(f"capacity {cap:>7,}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
