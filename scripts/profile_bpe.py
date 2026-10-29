"""Where ``bpe_parts`` spends its time on the distinct pieces of a cold encode.

    uv run python scripts/profile_bpe.py

For each input (~1 MB, GPT-2 ranks), on its distinct pieces, i.e. what a cold encode
runs BPE on:

- time per piece-length bucket, with the real ``bpe_parts`` (best of ``REPEAT``);
- per-piece counts from an instrumented copy: initial pairs and how many are ranked,
  heap pops, stale pops, merges, pushes after merges and how many are ranked, parts;
- phase times from a second copy timed with perf_counter between phases. The timer
  calls add overhead, so the copy's total is printed next to the real function's.
"""

import heapq
import time
from collections import Counter
from collections.abc import Callable

import tiktoken
from adversarial_inputs import INPUTS

from fastbpe.encode import bpe_parts
from fastbpe.pretokenize import pretokenize

REPEAT = 3
MB = 1_000_000
CASES = ["regular", "base64", "minified", "cjk"]
BUCKETS = [2, 4, 8, 16, 32, 64, 1 << 62]


def bucket(n: int) -> int:
    return next(b for b in BUCKETS if n < b)


def label(b: int) -> str:
    i = BUCKETS.index(b)
    lo = 0 if i == 0 else BUCKETS[i - 1]
    return f"{lo}+" if b == BUCKETS[-1] else f"{lo}-{b - 1}"


def best(f: Callable[[], object]) -> float:
    times = []
    for _ in range(REPEAT):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def counted(piece: bytes, ranks: dict[bytes, int], c: Counter[str]) -> None:
    """``bpe_parts`` with counters; same algorithm, results discarded."""
    n = len(piece)
    end_of = list(range(1, n + 1))
    start_before = list(range(-1, n - 1))
    heap: list[tuple[int, int, int, int]] = []

    def push(start: int, mid: int, end: int, kind: str) -> None:
        c[kind] += 1
        rank = ranks.get(piece[start:end])
        if rank is not None:
            c[kind + "_ranked"] += 1
            heapq.heappush(heap, (rank, start, mid, end))

    for i in range(n - 1):
        push(i, i + 1, i + 2, "init")
    while heap:
        c["pop"] += 1
        _, start, mid, end = heapq.heappop(heap)
        if end_of[start] != mid or end_of[mid] != end:
            c["stale"] += 1
            continue
        c["merge"] += 1
        end_of[start] = end
        end_of[mid] = -1
        if start > 0:
            push(start_before[start], start, end, "repush")
        if end < n:
            start_before[end] = start
            push(start, end, end_of[end], "repush")


def phased(piece: bytes, ranks: dict[bytes, int], t: list[float]) -> None:
    """``bpe_parts`` with perf_counter between its four phases, added to ``t``."""
    clock = time.perf_counter
    t0 = clock()
    n = len(piece)
    end_of = list(range(1, n + 1))
    start_before = list(range(-1, n - 1))
    heap: list[tuple[int, int, int, int]] = []

    def push(start: int, mid: int, end: int) -> None:
        rank = ranks.get(piece[start:end])
        if rank is not None:
            heapq.heappush(heap, (rank, start, mid, end))

    t1 = clock()
    for i in range(n - 1):
        push(i, i + 1, i + 2)
    t2 = clock()
    while heap:
        _, start, mid, end = heapq.heappop(heap)
        if end_of[start] != mid or end_of[mid] != end:
            continue
        end_of[start] = end
        end_of[mid] = -1
        if start > 0:
            push(start_before[start], start, end)
        if end < n:
            start_before[end] = start
            push(start, end, end_of[end])
    t3 = clock()
    parts = []
    i = 0
    while i < n:
        parts.append(piece[i : end_of[i]])
        i = end_of[i]
    t4 = clock()
    t[0] += t1 - t0
    t[1] += t2 - t1
    t[2] += t3 - t2
    t[3] += t4 - t3


def main() -> None:
    ranks = dict(tiktoken.get_encoding("gpt2")._mergeable_ranks)
    for name in CASES:
        pieces = list(dict.fromkeys(p.encode() for p in pretokenize(INPUTS[name](MB))))
        total = best(lambda pieces=pieces: [bpe_parts(p, ranks) for p in pieces])
        print(f"\n## {name}: {len(pieces):,} distinct pieces, bpe_parts {total * 1e3:.1f} ms")

        groups: dict[int, list[bytes]] = {}
        for p in pieces:
            groups.setdefault(bucket(len(p)), []).append(p)
        print(
            "| bytes | pieces | ms | % | us/piece | init ranked % | pops | stale | merges | parts |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for b in sorted(groups):
            g = groups[b]
            ms = best(lambda g=g: [bpe_parts(p, ranks) for p in g]) * 1e3
            c: Counter[str] = Counter()
            for p in g:
                counted(p, ranks, c)
            parts = sum(len(bpe_parts(p, ranks)) for p in g)
            k = len(g)
            init_pct = 100 * c["init_ranked"] / c["init"] if c["init"] else 0.0
            print(
                f"| {label(b)} | {k:,} | {ms:.1f} | {100 * ms / (total * 1e3):.0f} "
                f"| {1e3 * ms / k:.2f} | {init_pct:.0f} | {c['pop'] / k:.1f} "
                f"| {c['stale'] / k:.1f} | {c['merge'] / k:.1f} | {parts / k:.1f} |"
            )

        t = [0.0] * 4
        for p in pieces:
            phased(p, ranks, t)
        s = sum(t)
        names = ["setup", "initial pairs", "merge loop", "collect parts"]
        phases = ", ".join(
            f"{nm} {x * 1e3:.1f} ms ({100 * x / s:.0f}%)" for nm, x in zip(names, t, strict=True)
        )
        print(f"phases (instrumented {s * 1e3:.1f} ms, real {total * 1e3:.1f} ms): {phases}")


if __name__ == "__main__":
    main()
