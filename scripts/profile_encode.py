"""Encode the benchmark corpus with GPT-2 ranks; run under cProfile.

uv run python -m cProfile -s tottime scripts/profile_encode.py
"""

import tiktoken
from corpus import load_corpus

from fastbpe import Tokenizer

enc = tiktoken.get_encoding("gpt2")
tokenizer = Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)
tokenizer.encode(load_corpus(1_000_000))
