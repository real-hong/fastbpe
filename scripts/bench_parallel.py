"""Serial vs parallel train_bpe across corpus sizes and start methods.

    uv run python scripts/bench_parallel.py [sizes in MB, default 0.1 1 10 100]

Prints a Markdown table of best-of-N wall times and speedups over workers=1. Corpora
above the stdlib size (~11 MB) repeat it: pre-tokenization cost stays linear in the
text, but the number of unique words stops growing.
"""

import multiprocessing
import sys
import time

from corpus import SEPARATOR, load_corpus

from fastbpe import train_bpe

VOCAB_SIZE = 1000
METHODS = ("spawn", "forkserver")
WORKERS = (2, 4, 8)


def corpus_of(megabytes: float) -> str:
    size = int(megabytes * 1_000_000)
    base = load_corpus(size)
    text = base
    while len(text.encode()) < size:
        text += SEPARATOR + base
    return text.encode()[:size].decode(errors="ignore")


def best_time(text: str, rounds: int, **kwargs: object) -> float:
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        train_bpe(text, VOCAB_SIZE, [SEPARATOR], **kwargs)  # type: ignore[arg-type]
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    sizes = [float(arg) for arg in sys.argv[1:]] or [0.1, 1, 10, 100]
    header = " | ".join(f"{m} w={w}" for m in METHODS for w in WORKERS)
    print(f"| MB | serial | {header} |")
    print("|---" * (2 + len(METHODS) * len(WORKERS)) + "|")
    for megabytes in sizes:
        text = corpus_of(megabytes)
        rounds = 1 if megabytes >= 50 else 3
        serial = best_time(text, rounds)
        cells = []
        for method in METHODS:
            context = multiprocessing.get_context(method)
            for workers in WORKERS:
                t = best_time(text, rounds, workers=workers, mp_context=context)
                cells.append(f"{t:.2f}s ({serial / t:.2f}x)")
        print(f"| {megabytes:g} | {serial:.2f}s | {' | '.join(cells)} |", flush=True)


if __name__ == "__main__":
    main()
