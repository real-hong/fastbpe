"""Public tokenizer API: encoding, decoding, special tokens and serialization."""

import json
import re
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from multiprocessing.context import BaseContext
from os import PathLike
from pathlib import Path
from typing import Literal, Self

from fastbpe.encode import bpe_parts, encode_bytes
from fastbpe.pretokenize import pretokenize, special_token_pattern
from fastbpe.train import train_bpe

# Encoded pieces kept per tokenizer; the cache is simply emptied when it fills up.
CACHE_SIZE = 100_000
# Special-token patterns kept per tokenizer for subsets of its special tokens passed as
# allowed_special; "all" has its own, compiled at construction.
SPECIAL_CACHE_SIZE = 16

# The allowed special tokens and the pattern that splits text around them.
_SpecialSplit = tuple[frozenset[str], re.Pattern[str] | None]


def _recover_merges(ranks: dict[bytes, int]) -> list[tuple[bytes, bytes]]:
    """Rebuild the merge list behind tiktoken-style ranks.

    Each multi-byte token is the last merge BPE makes when it encodes that token with
    only the tokens ranked below it.
    """
    merges: list[tuple[bytes, bytes]] = []
    known: dict[bytes, int] = {}
    for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if len(token) > 1:
            parts = bpe_parts(token, known)
            if len(parts) != 2:
                raise ValueError(f"token {token!r} (rank {rank}) is not a merge of two lower ranks")
            merges.append((parts[0], parts[1]))
        known[token] = rank
    return merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self._special_tokens = dict(special_tokens or {})
        if "" in self._special_tokens:
            raise ValueError("special tokens must not be empty")
        if len(set(self._special_tokens.values())) != len(self._special_tokens):
            raise ValueError(f"duplicate special token ids: {self._special_tokens}")
        self._vocab = dict(vocab)
        for special, token_id in self._special_tokens.items():
            # A trained vocab already holds the special token at its id; anything else
            # there would be silently overwritten.
            existing = self._vocab.get(token_id)
            if existing is not None and existing != special.encode():
                raise ValueError(
                    f"special token {special!r} id {token_id} is already used by {existing!r}"
                )
            self._vocab[token_id] = special.encode()
        self._merges = list(merges)

        # Training can build the same bytes through two different merges; encoding
        # always uses the lowest id.
        special_ids = set(self._special_tokens.values())
        self._ranks: dict[bytes, int] = {}
        for token_id, token in sorted(self._vocab.items()):
            if token_id not in special_ids:
                self._ranks.setdefault(token, token_id)
        self._cache: dict[str, list[int]] = {}

        # Building the pattern escapes and joins every special token, which with many of
        # them costs far more than encoding a short text; so it is done once per subset.
        all_special = frozenset(self._special_tokens)
        self._all_special: _SpecialSplit = (all_special, special_token_pattern(all_special))
        self._special_cache: dict[frozenset[str], _SpecialSplit] = {}

    @classmethod
    def from_ranks(cls, ranks: dict[bytes, int], special_tokens: dict[str, int]) -> Self:
        """Build from tiktoken-style mergeable ranks (token bytes -> id)."""
        vocab = {token_id: token for token, token_id in ranks.items()}
        return cls(vocab, _recover_merges(ranks), special_tokens)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        *,
        workers: int = 1,
        mp_context: BaseContext | None = None,
    ) -> Self:
        """See ``train_bpe``; parallel pre-tokenization is opt-in via ``workers``."""
        vocab, merges = train_bpe(
            text, vocab_size, special_tokens, workers=workers, mp_context=mp_context
        )
        first_special = len(vocab) - len(special_tokens)
        specials = {token: first_special + k for k, token in enumerate(special_tokens)}
        return cls(vocab, merges, specials)

    @property
    def vocab(self) -> dict[int, bytes]:
        return dict(self._vocab)

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        return list(self._merges)

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self._special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(
        self, text: str, *, allowed_special: Literal["all"] | AbstractSet[str] = "all"
    ) -> list[int]:
        """Encode ``text``; special tokens outside ``allowed_special`` are plain text."""
        if allowed_special == "all":
            allowed, pattern = self._all_special
        elif isinstance(allowed_special, AbstractSet):
            allowed, pattern = self._special_split(allowed_special)
        else:
            # Like tiktoken: any other string (or a list) is an error, not "all".
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

    def _special_split(self, allowed_special: AbstractSet[str]) -> _SpecialSplit:
        # The caller may change its set between calls, so the key is rebuilt every time:
        # O(min(len(allowed_special), number of special tokens)) for a set or frozenset,
        # which frozenset.intersection iterates from the smaller side. Unknown strings
        # drop out, so subsets that differ only in those share an entry.
        allowed = self._all_special[0].intersection(allowed_special)
        split = self._special_cache.get(allowed)
        if split is None:
            split = (allowed, special_token_pattern(allowed))
            if len(self._special_cache) >= SPECIAL_CACHE_SIZE:
                # Drop the oldest entry; pop, not del: another thread may drop it first.
                self._special_cache.pop(next(iter(self._special_cache)), None)
            self._special_cache[allowed] = split
        return split

    def _encode_piece(self, piece: str) -> list[int]:
        ids = self._cache.get(piece)
        if ids is None:
            ids = encode_bytes(piece.encode(), self._ranks)
            if len(self._cache) >= CACHE_SIZE:
                self._cache.clear()
            self._cache[piece] = ids
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode to text, replacing invalid UTF-8 sequences."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self._vocab[token_id] for token_id in ids)

    def save(self, path: str | PathLike[str]) -> None:
        data = {
            "vocab": {str(token_id): token.hex() for token_id, token in self._vocab.items()},
            "merges": [[left.hex(), right.hex()] for left, right in self._merges],
            "special_tokens": self._special_tokens,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {int(token_id): bytes.fromhex(token) for token_id, token in data["vocab"].items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in data["merges"]]
        return cls(vocab, merges, data["special_tokens"])
