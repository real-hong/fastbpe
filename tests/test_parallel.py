"""Parallel pre-tokenization under different process start methods.

Start methods and daemon processes need a real ``__main__`` and fresh processes, so
these tests run small scripts in a subprocess.
"""

import multiprocessing
import subprocess
import sys
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from fastbpe.pretokenize import count_pretokens, default_mp_context

START_METHODS = ["spawn", "forkserver"]

PRELUDE = """\
import multiprocessing as mp
import warnings

from fastbpe import train_bpe

TEXT = "<|e|>".join(f"doc {i}: the quick brown fox jumps over the lazy dog" for i in range(60))
"""


def run_script(tmp_path: Path, body: str) -> str:
    script = tmp_path / "script.py"
    script.write_text(PRELUDE + textwrap.dedent(body))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("method", START_METHODS)
def test_parallel_training_under_global_start_method(tmp_path: Path, method: str) -> None:
    out = run_script(
        tmp_path,
        f"""
        if __name__ == "__main__":
            mp.set_start_method({method!r}, force=True)
            serial = train_bpe(TEXT, 300, ["<|e|>"])
            print(train_bpe(TEXT, 300, ["<|e|>"], workers=2) == serial)
        """,
    )
    assert out == "True"


@pytest.mark.parametrize("method", START_METHODS)
def test_parallel_training_with_explicit_mp_context(tmp_path: Path, method: str) -> None:
    out = run_script(
        tmp_path,
        f"""
        if __name__ == "__main__":
            serial = train_bpe(TEXT, 300, ["<|e|>"])
            context = mp.get_context({method!r})
            print(train_bpe(TEXT, 300, ["<|e|>"], workers=2, mp_context=context) == serial)
        """,
    )
    assert out == "True"


def test_default_mp_context_prefers_forkserver() -> None:
    expected = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
    assert default_mp_context().get_start_method() == expected


def test_executor_and_mp_context_are_exclusive() -> None:
    with ThreadPoolExecutor(2) as executor, pytest.raises(ValueError, match="mp_context"):
        count_pretokens(
            ["a b", "c d"],
            workers=2,
            executor=executor,
            mp_context=multiprocessing.get_context("spawn"),
        )


def test_injected_executor_is_used_and_left_open() -> None:
    texts = [f"line {i}: the quick brown fox" for i in range(100)]
    with ThreadPoolExecutor(2) as executor:
        assert count_pretokens(texts, workers=2, executor=executor) == count_pretokens(texts)
        assert executor.submit(len, "still open").result() == 10


@pytest.mark.parametrize("method", START_METHODS)
def test_daemon_process_falls_back_to_serial_with_warning(tmp_path: Path, method: str) -> None:
    # Pool workers are daemonic and may not start children of their own.
    out = run_script(
        tmp_path,
        f"""
        def job():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                parallel = train_bpe(TEXT, 300, ["<|e|>"], workers=2)
            same = parallel == train_bpe(TEXT, 300, ["<|e|>"])
            return same, sorted({{w.category.__name__ for w in caught}})

        if __name__ == "__main__":
            with mp.get_context({method!r}).Pool(1) as pool:
                print(pool.apply(job))
        """,
    )
    assert out == "(True, ['RuntimeWarning'])"


def test_no_fork_while_other_threads_run() -> None:
    # fork() with other threads alive can deadlock the child; Python warns about it.
    texts = [f"line {i}: the quick brown fox" for i in range(200)]
    stop = threading.Event()
    other = threading.Thread(target=stop.wait)
    other.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parallel = count_pretokens(texts, workers=2)
    finally:
        stop.set()
        other.join()
    assert parallel == count_pretokens(texts)
    assert not [w for w in caught if "fork" in str(w.message)]
