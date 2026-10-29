"""Cold and warm ``Tokenizer.encode`` per encode-cache variant, one process per measurement.

    uv run python scripts/time_variants.py [--reps N] [--inputs a,b] [--variants a,b]

Each (input, variant) pair runs in a fresh process, and the order of the variants turns
by one every repetition, so neither the heap a previous measurement leaves behind nor
drift over the run favours one variant. Measuring several variants in one process was
misleading: a cache that had been thinned (every other entry freed) and was still alive
left holes that the next tokenizer's cache filled, spreading it over ~45% more memory
pages and making its warm encode ~24% slower (TODO.md, encode cache).

A worker (``--worker INPUT VARIANT``) prints one JSON line. For the 1 MB inputs: cold is
the first encode with a fresh tokenizer, warm the best of ``WARM`` encodes after it.
``stream`` encodes ``adversarial_inputs.stream`` document by document twice, timing the
corpus documents apart from the never-repeating filler. ``regular_after_unicode_all``
is the corpus, cold and warm, while a tokenizer that encoded unicode_all (and so
cleared or thinned its cache) is still alive. Both variants cache the corpus alike, so
any difference is the locality cost a thinned cache imposes on what is allocated next
to it.

Inputs are cut with GPT-2 ranks, GC off (``--gc`` leaves it on), as in
``scripts/measure_share.py``.
"""

import argparse
import gc
import heapq
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from itertools import islice
from typing import Literal

import tiktoken
from adversarial_inputs import INPUTS, many_specials, stream

import fastbpe.tokenizer as tokenizer_module
from fastbpe import Tokenizer
from fastbpe.encode import encode_bytes
from fastbpe.pretokenize import pretokenize

MB = 1_000_000
WARM = 3
GC_ON_ENV = "TIME_VARIANTS_GC"
INPUT_NAMES = [*INPUTS, "specials", "stream", "regular_after_unicode_all"]


class PreInlineTokenizer(Tokenizer):
    """Master before ``encode`` looked cache hits up itself: a method call per piece."""

    def encode(
        self, text: str, *, allowed_special: Literal["all"] | AbstractSet[str] = "all"
    ) -> list[int]:
        if allowed_special == "all":
            allowed, pattern = self._all_split or self._compile_all_split()
        elif isinstance(allowed_special, AbstractSet):
            allowed, pattern = self._special_split(allowed_special)
        else:
            raise TypeError(
                f'allowed_special must be "all" or a set of strings, got {allowed_special!r}'
            )

        ids: list[int] = []
        for chunk in pattern.split(text) if pattern is not None else [text]:
            if not chunk:
                continue
            if chunk in allowed:
                ids.append(self._special_tokens[chunk])
                continue
            for piece in pretokenize(chunk):
                ids.extend(self._encode_piece(piece))
        return ids

    def _encode_piece(self, piece: str) -> list[int]:
        ids = self._cache.get(piece)
        if ids is None:
            data = piece.encode()
            token_id = self._ranks.get(data)
            ids = [token_id] if token_id is not None else encode_bytes(data, self._ranks)
            if self._cache_size:
                if len(self._cache) >= self._cache_size:
                    self._cache.clear()
                self._cache[piece] = ids
        return ids


class ThinTokenizer(PreInlineTokenizer):
    """Thinning instead of clearing: a full cache is replaced by every other entry.

    Inserting and replacing hold one lock, so a thread that inserts while another
    replaces the dict cannot lose its entry into the dropped one; lookups take no lock.
    Built on ``PreInlineTokenizer``, the code it was measured against: master's
    ``encode`` binds the cache's ``get`` once per call, so it cannot replace the dict.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        super().__init__(vocab, merges, special_tokens)
        self._cache_lock = threading.Lock()

    def _encode_piece(self, piece: str) -> list[int]:
        ids = self._cache.get(piece)
        if ids is None:
            data = piece.encode()
            token_id = self._ranks.get(data)
            ids = [token_id] if token_id is not None else encode_bytes(data, self._ranks)
            with self._cache_lock:
                if len(self._cache) >= tokenizer_module.CACHE_SIZE:
                    self._cache = dict(islice(self._cache.items(), 1, None, 2))
                self._cache[piece] = ids
        return ids


class MasterCopyTokenizer(Tokenizer):
    """Control: ``Tokenizer.encode`` and ``_encode_miss`` copied unchanged.

    The other subclasses differ from master only in what they change, so they are
    compared against this one; its ratio to ``clear`` shows what being a subclass
    defined in a script costs by itself, and its spread is the noise. Keep it in step
    with master: a variant overriding a method master no longer calls measures nothing.
    """

    def encode(
        self, text: str, *, allowed_special: Literal["all"] | AbstractSet[str] = "all"
    ) -> list[int]:
        if allowed_special == "all":
            allowed, pattern = self._all_split or self._compile_all_split()
        elif isinstance(allowed_special, AbstractSet):
            allowed, pattern = self._special_split(allowed_special)
        else:
            raise TypeError(
                f'allowed_special must be "all" or a set of strings, got {allowed_special!r}'
            )

        get = self._cache.get
        ids: list[int] = []
        for chunk in pattern.split(text) if pattern is not None else [text]:
            if not chunk:
                continue
            if chunk in allowed:
                ids.append(self._special_tokens[chunk])
                continue
            for piece in pretokenize(chunk):
                hit = get(piece)
                ids.extend(hit if hit is not None else self._encode_miss(piece))
        return ids

    def _encode_miss(self, piece: str) -> list[int]:
        data = piece.encode()
        token_id = self._ranks.get(data)
        ids = [token_id] if token_id is not None else encode_bytes(data, self._ranks)
        if self._cache_size:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[piece] = ids
        return ids


def bpe_parts_pre_heapify(piece: bytes, ranks: dict[bytes, int]) -> list[bytes]:
    """``bpe_parts`` before 880d05f: initial pairs pushed one by one through ``push``."""
    n = len(piece)
    end_of = list(range(1, n + 1))
    start_before = list(range(-1, n - 1))
    heap: list[tuple[int, int, int, int]] = []

    def push(start: int, mid: int, end: int) -> None:
        rank = ranks.get(piece[start:end])
        if rank is not None:
            heapq.heappush(heap, (rank, start, mid, end))

    for i in range(n - 1):
        push(i, i + 1, i + 2)

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

    parts = []
    i = 0
    while i < n:
        parts.append(piece[i : end_of[i]])
        i = end_of[i]
    return parts


def bpe_parts_heapify(piece: bytes, ranks: dict[bytes, int]) -> list[bytes]:
    """``bpe_parts`` at 880d05f: one heapify, re-pushes still through ``push``."""
    n = len(piece)
    end_of = list(range(1, n + 1))
    start_before = list(range(-1, n - 1))
    heap = [
        (rank, i, i + 1, i + 2)
        for i in range(n - 1)
        if (rank := ranks.get(piece[i : i + 2])) is not None
    ]
    heapq.heapify(heap)

    def push(start: int, mid: int, end: int) -> None:
        rank = ranks.get(piece[start:end])
        if rank is not None:
            heapq.heappush(heap, (rank, start, mid, end))

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

    parts = []
    i = 0
    while i < n:
        parts.append(piece[i : end_of[i]])
        i = end_of[i]
    return parts


def encode_bytes_pre(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    return [ranks[part] for part in bpe_parts_pre_heapify(piece, ranks)]


def encode_bytes_heapify(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    return [ranks[part] for part in bpe_parts_heapify(piece, ranks)]


class BpePreTokenizer(MasterCopyTokenizer):
    """Master with ``bpe_parts`` as before 880d05f and a5b2521 (finding #8)."""

    def _encode_miss(self, piece: str) -> list[int]:
        data = piece.encode()
        token_id = self._ranks.get(data)
        ids = [token_id] if token_id is not None else encode_bytes_pre(data, self._ranks)
        if self._cache_size:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[piece] = ids
        return ids


class BpeHeapifyTokenizer(MasterCopyTokenizer):
    """Master with ``bpe_parts`` as at 880d05f, before a5b2521 inlined the re-pushes."""

    def _encode_miss(self, piece: str) -> list[int]:
        data = piece.encode()
        token_id = self._ranks.get(data)
        ids = [token_id] if token_id is not None else encode_bytes_heapify(data, self._ranks)
        if self._cache_size:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[piece] = ids
        return ids


class NoWholeTokenizer(MasterCopyTokenizer):
    """Master without the whole-piece lookup of 19495bc: every miss runs BPE.

    Same output for GPT-2 ranks (tests/test_whole_token.py); checked per run.
    """

    def _encode_miss(self, piece: str) -> list[int]:
        ids = encode_bytes(piece.encode(), self._ranks)
        if self._cache_size:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[piece] = ids
        return ids


VARIANTS: dict[str, type[Tokenizer]] = {
    "clear": Tokenizer,
    "thin": ThinTokenizer,
    "master_copy": MasterCopyTokenizer,
    "pre_inline": PreInlineTokenizer,
    "bpe_pre": BpePreTokenizer,
    "bpe_heapify": BpeHeapifyTokenizer,
    "no_whole": NoWholeTokenizer,
}
DEFAULT_VARIANTS = "clear,thin"


def gpt2_parts() -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], dict[str, int]]:
    enc = tiktoken.get_encoding("gpt2")
    gpt2 = Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)
    return gpt2.vocab, gpt2.merges, gpt2.special_tokens


def timed(f: Callable[[str], object], arg: str) -> float:
    start = time.perf_counter()
    f(arg)
    return time.perf_counter() - start


def cold_warm(make: Callable[[], Tokenizer], text: str) -> dict[str, float]:
    tok = make()
    cold = timed(tok.encode, text)
    return {"cold": cold, "warm": min(timed(tok.encode, text) for _ in range(WARM))}


def encode_stream(tok: Tokenizer, docs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in (1, 2):
        corpus = filler = 0.0
        for i, doc in enumerate(docs):
            t = timed(tok.encode, doc)
            if i % 2:
                filler += t
            else:
                corpus += t
        out |= {f"pass{p}": corpus + filler, f"corpus{p}": corpus, f"filler{p}": filler}
    return out


def disable_gc() -> None:
    if not os.environ.get(GC_ON_ENV):
        gc.disable()


def worker(name: str, variant: str) -> dict[str, float]:
    vocab, merges, specials = gpt2_parts()
    cls = VARIANTS[variant]
    text = ""
    docs: list[str] = []
    if name == "specials":
        text, extra = many_specials(MB)
        specials = specials | {s: len(vocab) + i for i, s in enumerate(extra)}
    elif name == "regular_after_unicode_all":
        text = INPUTS["regular"](MB)
    elif name != "stream":
        text = INPUTS[name](MB)

    def make() -> Tokenizer:
        return cls(vocab, merges, specials)

    if name == "stream":
        docs = stream(4 * MB)
        disable_gc()
        result = encode_stream(make(), docs)
    elif name == "regular_after_unicode_all":
        other = INPUTS["unicode_all"](MB)
        disable_gc()
        alive = make()
        for _ in range(2):
            alive.encode(other)
        result = cold_warm(make, text)
    else:
        disable_gc()
        result = cold_warm(make, text)
    # After timing: a variant must encode exactly like master.
    if cls is not Tokenizer:
        tok, ref = make(), Tokenizer(vocab, merges, specials)
        for t in docs if name == "stream" else [text]:
            if tok.encode(t) != ref.encode(t):
                raise SystemExit(f"{variant} encodes {name} differently from master")
    return result


def run(name: str, variant: str) -> dict[str, float]:
    out = subprocess.run(
        [sys.executable, __file__, "--worker", name, variant],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, float] = json.loads(out.stdout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--inputs", default=",".join(INPUT_NAMES))
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--gc", action="store_true", help="keep GC on while timing")
    parser.add_argument("--json", help="also write every process's values to this file")
    parser.add_argument("--worker", nargs=2, metavar=("INPUT", "VARIANT"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return

    if args.gc:
        os.environ[GC_ON_ENV] = "1"  # inherited by the workers
    names = args.inputs.split(",")
    variants = args.variants.split(",")
    results: dict[tuple[str, str], list[dict[str, float]]] = {}
    for rep in range(args.reps):
        for name in names:
            k = rep % len(variants)
            for variant in variants[k:] + variants[:k]:
                results.setdefault((name, variant), []).append(run(name, variant))
        print(f"rep {rep + 1}/{args.reps} done", file=sys.stderr, flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([[n, v, r] for (n, v), r in results.items()], f)
    base = variants[0]
    print(
        f"ms, min (median) over {args.reps} processes; "
        f"ratio = {base} min / variant min ({base} median / variant median)"
    )
    for name in names:
        for key in results[name, base][0]:
            cells = []
            for variant in variants:
                values = [r[key] * 1e3 for r in results[name, variant]]
                base_values = [r[key] * 1e3 for r in results[name, base]]
                lo, mid = min(values), statistics.median(values)
                ratio = min(base_values) / lo
                mid_ratio = statistics.median(base_values) / mid
                cells.append(f"{variant} {lo:8.1f} ({mid:8.1f}) {ratio:5.2f}x ({mid_ratio:4.2f}x)")
            print(f"{name:18} {key:8} " + "  ".join(cells))


if __name__ == "__main__":
    main()
