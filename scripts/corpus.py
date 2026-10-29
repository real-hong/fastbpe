"""Benchmark corpus: the running Python's standard library sources.

They ship with the uv-managed Python pinned in .python-version, so every machine sees
the same text: code plus a lot of English prose (docstrings, pydoc_data/topics.py).
"""

import sysconfig
from pathlib import Path

SEPARATOR = "<|endoftext|>"


def load_corpus(max_bytes: int) -> str:
    """Concatenate stdlib ``.py`` files in path order, separated by ``SEPARATOR``."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    texts: list[str] = []
    size = 0
    for path in sorted(stdlib.rglob("*.py")):
        if "site-packages" in path.parts or "test" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        texts.append(text)
        size += len(text.encode())
        if size >= max_bytes:
            break
    return SEPARATOR.join(texts).encode()[:max_bytes].decode(errors="ignore")
