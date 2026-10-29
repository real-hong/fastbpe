"""Train on the benchmark corpus; run under cProfile.

uv run python -m cProfile -s cumtime scripts/profile_train.py
"""

from corpus import SEPARATOR, load_corpus

from fastbpe import train_bpe

train_bpe(load_corpus(1_000_000), 1000, [SEPARATOR])
