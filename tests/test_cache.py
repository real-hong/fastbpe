"""The encode cache: its size limit, ``cache_size`` validation, and concurrent use."""

import random
import sys
import sysconfig
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import fastbpe.tokenizer as tokenizer_module
from fastbpe import Tokenizer

# Many distinct pieces: words, numbers and CJK, so a small cache fills up repeatedly.
WORDS = " ".join(f"w{i} {i * 7} 字{chr(0x4E00 + i)}" for i in range(400))


def fresh(tok: Tokenizer, **kwargs: int) -> Tokenizer:
    return Tokenizer(tok.vocab, tok.merges, tok.special_tokens, **kwargs)


def test_default_cache_size(small_tokenizer: Tokenizer) -> None:
    assert fresh(small_tokenizer)._cache_size == tokenizer_module.CACHE_SIZE


@pytest.mark.parametrize("size", [1, 5, 64])
def test_cache_stays_within_size(small_tokenizer: Tokenizer, size: int) -> None:
    expected = fresh(small_tokenizer).encode(WORDS)
    tok = fresh(small_tokenizer, cache_size=size)
    for _ in range(2):
        for word in WORDS.split():
            tok.encode(word)
            assert len(tok._cache) <= size
        assert tok.encode(WORDS) == expected


def test_cache_size_zero_disables_the_cache(small_tokenizer: Tokenizer) -> None:
    expected = fresh(small_tokenizer).encode(WORDS)
    tok = fresh(small_tokenizer, cache_size=0)
    for _ in range(2):
        assert tok.encode(WORDS) == expected
        assert not tok._cache


@pytest.mark.parametrize("size", [-1, -100])
def test_negative_cache_size_is_rejected(small_tokenizer: Tokenizer, size: int) -> None:
    with pytest.raises(ValueError, match="cache_size"):
        fresh(small_tokenizer, cache_size=size)


@pytest.mark.parametrize("size", [True, False, 1.0, "10", None])
def test_cache_size_must_be_int(small_tokenizer: Tokenizer, size: object) -> None:
    with pytest.raises(TypeError, match="cache_size"):
        Tokenizer(small_tokenizer.vocab, small_tokenizer.merges, cache_size=size)  # type: ignore[arg-type]


def test_constructors_pass_cache_size_on(
    small_corpus: str, small_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tok.json"
    small_tokenizer.save(path)
    ranks = {token: token_id for token_id, token in small_tokenizer.vocab.items()}
    ranks = {t: i for t, i in ranks.items() if i not in small_tokenizer.special_tokens.values()}
    built = [
        Tokenizer.train(small_corpus, 300, ["<|endoftext|>"], cache_size=7),
        Tokenizer.load(path, cache_size=7),
        Tokenizer.from_ranks(ranks, small_tokenizer.special_tokens, cache_size=7),
    ]
    for tok in built:
        assert tok._cache_size == 7
        assert tok.encode(WORDS) == small_tokenizer.encode(WORDS)


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs a free-threaded build"
)
def test_free_threaded_build_runs_without_the_gil() -> None:
    # Importing an extension module that does not declare free-threading support turns
    # the GIL back on; then the thread tests below would not run in parallel at all.
    assert not sys._is_gil_enabled()


@pytest.fixture
def short_switch_interval() -> Iterator[None]:
    # On GIL builds, switch threads far more often than every 5 ms, so that they
    # interleave inside encode; free-threaded builds run them in parallel anyway.
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(old)


@pytest.mark.usefixtures("short_switch_interval")
@pytest.mark.parametrize("size", [0, 1, 50, 100_000])
def test_threads_sharing_a_tokenizer(small_tokenizer: Tokenizer, size: int) -> None:
    # Threads encode the same pieces in different orders through one small cache, so it
    # is cleared while others read and insert. Every result must match a tokenizer
    # without cache, and the cache may overshoot its size by at most one entry per
    # thread (each may insert after another's check-and-clear).
    words = WORDS.split()
    uncached = fresh(small_tokenizer, cache_size=0)
    expected = {w: uncached.encode(w) for w in words}
    tok = fresh(small_tokenizer, cache_size=size)
    n_threads = 8
    errors: list[str] = []
    oversize: list[int] = []
    start = threading.Barrier(n_threads)

    def work(seed: int) -> None:
        order = random.Random(seed).sample(words, len(words))
        start.wait()
        for _ in range(5):
            for w in order:
                if tok.encode(w) != expected[w]:
                    errors.append(w)
                if len(tok._cache) > size + n_threads:
                    oversize.append(len(tok._cache))

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert not oversize
