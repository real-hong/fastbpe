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

Inputs are cut with GPT-2 ranks, GC off, as in ``scripts/measure_share.py``.
"""

import argparse
import gc
import json
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from itertools import islice

import tiktoken
from adversarial_inputs import INPUTS, many_specials, stream

import fastbpe.tokenizer as tokenizer_module
from fastbpe import Tokenizer
from fastbpe.encode import encode_bytes

MB = 1_000_000
WARM = 3
INPUT_NAMES = [*INPUTS, "specials", "stream", "regular_after_unicode_all"]


class ThinTokenizer(Tokenizer):
    """Thinning instead of clearing: a full cache is replaced by every other entry.

    Inserting and replacing hold one lock, so a thread that inserts while another
    replaces the dict cannot lose its entry into the dropped one; lookups take no lock.
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


VARIANTS: dict[str, type[Tokenizer]] = {"clear": Tokenizer, "thin": ThinTokenizer}


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


def worker(name: str, variant: str) -> dict[str, float]:
    vocab, merges, specials = gpt2_parts()
    cls = VARIANTS[variant]
    text = ""
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
        gc.disable()
        return encode_stream(make(), docs)
    if name == "regular_after_unicode_all":
        other = INPUTS["unicode_all"](MB)
        gc.disable()
        alive = make()
        for _ in range(2):
            alive.encode(other)
        return cold_warm(make, text)
    gc.disable()
    return cold_warm(make, text)


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
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--worker", nargs=2, metavar=("INPUT", "VARIANT"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return

    names = args.inputs.split(",")
    variants = args.variants.split(",")
    results: dict[tuple[str, str], list[dict[str, float]]] = {}
    for rep in range(args.reps):
        for name in names:
            k = rep % len(variants)
            for variant in variants[k:] + variants[:k]:
                results.setdefault((name, variant), []).append(run(name, variant))
        print(f"rep {rep + 1}/{args.reps} done", file=sys.stderr, flush=True)

    base = variants[0]
    print(f"ms, min (median) over {args.reps} processes; ratio = {base} min / variant min")
    for name in names:
        for key in results[name, base][0]:
            cells = []
            for variant in variants:
                values = [r[key] * 1e3 for r in results[name, variant]]
                ratio = min(r[key] for r in results[name, base]) * 1e3 / min(values)
                cells.append(
                    f"{variant} {min(values):8.1f} ({statistics.median(values):8.1f}) {ratio:5.2f}x"
                )
            print(f"{name:18} {key:8} " + "  ".join(cells))


if __name__ == "__main__":
    main()
