"""Training on adversarial inputs (long words without spaces) and on the regular corpus.

Besides the timed rounds, each case makes one untimed call that records into
``extra_info``: the tracemalloc peak and, for the incremental trainer, heap statistics
(pushes, heapify calls, dead-entry ratio). ``scripts/bench_table.py`` prints them.
"""

import base64
import heapq
import random
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest
from naive_bpe import naive_train_bpe
from pytest_benchmark.fixture import BenchmarkFixture

import fastbpe.train as train_module
from fastbpe import train_bpe

TrainFn = Callable[..., tuple[dict[int, bytes], list[tuple[bytes, bytes]]]]

ADVERSARIAL_VOCAB_SIZE = 256 + 200
LENGTHS = (10_000, 40_000, 160_000)


def _dna(n: int) -> str:
    # Letters only: the whole string is one pre-token.
    return "".join(random.Random(0).choices("ACGT", k=n))


def _letters(n: int) -> str:
    # One pre-token over ten letters, as in scripts/review/longword.py.
    return "".join(random.Random(0).choices("abcdefghij", k=n))


def _run_a(n: int) -> str:
    # One pre-token of a single repeated byte: every pair overlaps its neighbours.
    return "a" * n


def _base64(n: int) -> str:
    # The GPT-2 pattern splits letters, digits and "+/=" apart, so this is many short
    # (mean ~3 bytes), mostly distinct pieces: short words, low repetition.
    return base64.b64encode(random.Random(0).randbytes(n * 3 // 4)).decode()


INPUTS: dict[str, Callable[[int], str]] = {
    "dna": _dna,
    "letters": _letters,
    "run_a": _run_a,
    "base64": _base64,
}
IMPLS: dict[str, TrainFn] = {"current": train_bpe, "naive": naive_train_bpe}


class HeapProbe:
    """Stands in for the ``heapq`` module inside ``fastbpe.train`` during the untimed call.

    It only sees calls made as ``heapq.<name>(...)``; any other heapq function raises
    AttributeError, and a trainer that stops calling through the module records zero
    pushes, which the benchmark treats as a failure rather than reporting it.
    """

    def __init__(self) -> None:
        self.pushes = 0
        self.heapify_calls = 0
        self.heapify_entries = 0
        self.max_dead_ratio = 0.0
        self.stats: train_module._PairStats | None = None

    def _sample(self, heap: list[train_module._Candidate]) -> None:
        if heap and self.stats is not None:
            dead = len(heap) - len(self.stats.counts)
            self.max_dead_ratio = max(self.max_dead_ratio, dead / len(heap))

    def heappush(self, heap: list[train_module._Candidate], item: train_module._Candidate) -> None:
        self.pushes += 1
        heapq.heappush(heap, item)

    def heappop(self, heap: list[train_module._Candidate]) -> train_module._Candidate:
        self._sample(heap)
        return heapq.heappop(heap)

    def heapify(self, heap: list[train_module._Candidate]) -> None:
        self.heapify_calls += 1
        self.heapify_entries += len(heap)
        heapq.heapify(heap)


def _record(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    fn: TrainFn,
    args: tuple[Any, ...],
    *,
    probe_heap: bool,
) -> None:
    """One untimed call: tracemalloc peak, plus heap statistics when ``probe_heap``."""
    probe = HeapProbe()
    if probe_heap:
        stats_cls = train_module._PairStats
        original_init = stats_cls.__init__

        def init(self: train_module._PairStats) -> None:
            original_init(self)
            probe.stats = self

        monkeypatch.setattr(stats_cls, "__init__", init)
        monkeypatch.setattr(train_module, "heapq", probe)

    tracemalloc.start()
    try:
        fn(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        monkeypatch.undo()
    benchmark.extra_info["peak_mib"] = round(peak / 2**20, 2)

    if probe_heap:
        assert probe.stats is not None, "_PairStats was not instantiated"
        assert probe.pushes > 0, "no heappush seen: the heapq probe no longer applies"
        heap, live = len(probe.stats._heap), len(probe.stats.counts)
        benchmark.extra_info.update(
            heap_pushes=probe.pushes,
            heapify_calls=probe.heapify_calls,
            heapify_entries=probe.heapify_entries,
            max_dead_ratio=round(probe.max_dead_ratio, 4),
            final_heap=heap,
            final_live=live,
            final_dead_ratio=round((heap - live) / heap, 4) if heap else 0.0,
        )


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("kind", list(INPUTS))
def test_train_adversarial(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    length: int,
    impl: str,
) -> None:
    benchmark.group = f"train-{kind}"
    fn = IMPLS[impl]
    args = (INPUTS[kind](length), ADVERSARIAL_VOCAB_SIZE)
    _record(benchmark, monkeypatch, fn, args, probe_heap=impl == "current")
    benchmark.pedantic(fn, args=args, rounds=1 if length > 40_000 else 3)


@pytest.mark.parametrize("vocab_size", [1000, 5000])
def test_train_regular(
    benchmark: BenchmarkFixture,
    monkeypatch: pytest.MonkeyPatch,
    train_corpus: str,
    separator: str,
    vocab_size: int,
) -> None:
    benchmark.group = "train-regular"
    args = (train_corpus, vocab_size, [separator])
    _record(benchmark, monkeypatch, train_bpe, args, probe_heap=True)
    benchmark.pedantic(train_bpe, args=args, rounds=3)
