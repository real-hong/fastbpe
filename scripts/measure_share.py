"""Share of pre-tokenization in training and encoding, measured with perf_counter.

    uv run python scripts/measure_share.py

Train: split on special tokens, ``count_pretokens`` (pre-tokenization plus UTF-8
encoding and counting) and ``_train_merges``, timed separately in one process.
Encode: ``Tokenizer.encode`` with GPT-2 ranks, cold and warm cache, against
``pretokenize`` alone over the same chunks. Timings are per chunk list, not per piece, so
the measurement adds no per-call overhead (unlike cProfile). Best of ``REPEAT``.
"""

import time
from collections.abc import Callable

import tiktoken
from adversarial_inputs import INPUTS, many_specials
from corpus import SEPARATOR

from fastbpe import Tokenizer
from fastbpe.pretokenize import count_pretokens, pretokenize, split_on_special_tokens
from fastbpe.train import _train_merges

REPEAT = 3
MB = 1_000_000

# (name, text, special tokens, vocab size)
TRAIN_CASES: list[tuple[str, str, list[str], int]] = [
    ("regular 1MB v1000", INPUTS["regular"](MB), [SEPARATOR], 1000),
    ("dna 160k v456", INPUTS["dna"](160_000), [], 456),
    ("base64 1MB v456", INPUTS["base64"](MB), [], 456),
    ("minified 1MB v1000", INPUTS["minified"](MB), [SEPARATOR], 1000),
    ("cjk 1MB v1000", INPUTS["cjk"](MB), [], 1000),
    ("50 specials 1MB v1050", *many_specials(MB), 1050),
]
ENCODE_CASES = ["regular", "base64", "minified", "cjk", "dna"]


def best(f: Callable[..., object], *args: object) -> float:
    times = []
    for _ in range(REPEAT):
        start = time.perf_counter()
        f(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def pretokenize_all(chunks: list[str]) -> None:
    for chunk in chunks:
        pretokenize(chunk)


def encode_cold(gpt2: Tokenizer, text: str) -> None:
    Tokenizer(gpt2.vocab, gpt2.merges, gpt2.special_tokens).encode(text)


def measure_train() -> None:
    print(f"TRAIN (workers=1), best of {REPEAT}, seconds")
    print(
        f"{'case':24} {'split':>7} {'count':>7} {'(pretok)':>8} {'merges':>7} {'total':>7} "
        f"{'pretok%':>7} {'count%':>7}"
    )
    for name, text, specials, vocab_size in TRAIN_CASES:
        special_set = set(specials)
        chunks = [c for c in split_on_special_tokens(text, specials) if c not in special_set]
        t_split = best(split_on_special_tokens, text, specials)
        t_count = best(count_pretokens, chunks)
        t_pre = best(pretokenize_all, chunks)
        counts = count_pretokens(chunks)
        n_merges = vocab_size - 256 - len(specials)
        t_merge = best(_train_merges, counts, n_merges)
        total = t_split + t_count + t_merge
        print(
            f"{name:24} {t_split:7.3f} {t_count:7.3f} {t_pre:8.3f} {t_merge:7.3f} {total:7.3f} "
            f"{t_pre / total:7.1%} {t_count / total:7.1%}"
        )


def measure_encode() -> None:
    enc = tiktoken.get_encoding("gpt2")
    gpt2 = Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)
    print(f"\nENCODE (gpt2 ranks), best of {REPEAT}, seconds")
    print(f"{'case':24} {'cold':>7} {'warm':>7} {'pretok':>7} {'pre/cold':>8} {'pre/warm':>8}")
    allowed = set(gpt2.special_tokens)
    for name in ENCODE_CASES:
        text = INPUTS[name](160_000 if name == "dna" else MB)

        t_cold = best(encode_cold, gpt2, text)
        warm = Tokenizer(gpt2.vocab, gpt2.merges, gpt2.special_tokens)
        warm.encode(text)
        t_warm = best(warm.encode, text)
        chunks = [c for c in split_on_special_tokens(text, allowed) if c not in allowed]
        t_pre = best(pretokenize_all, chunks)
        print(
            f"{name:24} {t_cold:7.3f} {t_warm:7.3f} {t_pre:7.3f} "
            f"{t_pre / t_cold:8.1%} {t_pre / t_warm:8.1%}"
        )


if __name__ == "__main__":
    measure_train()
    measure_encode()
