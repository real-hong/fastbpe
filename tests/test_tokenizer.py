from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Literal

import pytest

import fastbpe.tokenizer as tokenizer_module
from fastbpe import Tokenizer

EOT = "<|endoftext|>"
SAMPLES = ["hello world", f"before{EOT}after", "你好 👍🏽", ""]


def test_vocab_size(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.vocab_size == 300
    assert len(small_tokenizer.vocab) == 300


def test_constructor_matches_trained(small_tokenizer: Tokenizer) -> None:
    rebuilt = Tokenizer(
        small_tokenizer.vocab, small_tokenizer.merges, small_tokenizer.special_tokens
    )
    for text in SAMPLES:
        assert rebuilt.encode(text) == small_tokenizer.encode(text)


def test_special_token_alone(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.encode(EOT) == [small_tokenizer.special_tokens[EOT]]


def test_special_token_inside_text(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.encode(f"hello{EOT} world") == [
        *small_tokenizer.encode("hello"),
        eot,
        *small_tokenizer.encode(" world"),
    ]


def test_overlapping_special_tokens_use_longest_match(small_corpus: str) -> None:
    tok = Tokenizer.train(small_corpus, vocab_size=300, special_tokens=["<|a|>", "<|a|><|b|>"])
    long_id = tok.special_tokens["<|a|><|b|>"]
    short_id = tok.special_tokens["<|a|>"]
    assert tok.encode("<|a|><|b|>") == [long_id]
    assert tok.encode("<|a|><|a|>") == [short_id, short_id]


def test_disallowed_special_is_plain_text(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    ids = small_tokenizer.encode(EOT, allowed_special=set())
    assert eot not in ids
    assert len(ids) > 1
    assert small_tokenizer.decode(ids) == EOT


def test_allowed_special_subset(small_corpus: str) -> None:
    tok = Tokenizer.train(small_corpus, vocab_size=300, special_tokens=[EOT, "<|pad|>"])
    ids = tok.encode(f"{EOT}<|pad|>", allowed_special={"<|pad|>"})
    assert tok.special_tokens[EOT] not in ids
    assert ids[-1] == tok.special_tokens["<|pad|>"]


@pytest.mark.parametrize("bad", ["none", EOT, "", "ALL", [EOT], (EOT,), None])
def test_allowed_special_rejects_anything_but_all_or_a_set(
    small_tokenizer: Tokenizer, bad: object
) -> None:
    # As in tiktoken: a string other than "all" is not silently treated as "all".
    with pytest.raises(TypeError):
        small_tokenizer.encode(f"hi{EOT}", allowed_special=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("allowed", [frozenset({EOT}), {EOT: 0}.keys()])
def test_allowed_special_accepts_any_abstract_set(
    small_tokenizer: Tokenizer, allowed: AbstractSet[str]
) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.encode(EOT, allowed_special=allowed) == [eot]


def test_empty_special_token_is_rejected_at_construction() -> None:
    vocab = {i: bytes([i]) for i in range(256)}
    with pytest.raises(ValueError, match="empty"):
        Tokenizer(vocab, [], {"": 256})
    with pytest.raises(ValueError, match="empty"):
        Tokenizer.train("ab ab ab cd", vocab_size=300, special_tokens=[""])


BYTE_VOCAB = {i: bytes([i]) for i in range(256)}
BYTE_VOCAB_RANKS = {token: i for i, token in BYTE_VOCAB.items()}


def test_duplicate_special_tokens_are_rejected_in_training() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Tokenizer.train("abc <x> abc", vocab_size=300, special_tokens=["<x>", "<x>"])


def test_special_tokens_sharing_an_id_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Tokenizer(BYTE_VOCAB, [], {"<a>": 256, "<b>": 256})


def test_special_token_id_taken_by_a_regular_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="already used"):
        Tokenizer.from_ranks(BYTE_VOCAB_RANKS | {b"ab": 256}, {"<s>": 256})


def test_special_token_id_may_already_hold_its_own_bytes() -> None:
    tok = Tokenizer(BYTE_VOCAB | {256: b"<s>"}, [], {"<s>": 256})
    assert tok.encode("<s>") == [256]


def test_decode_special_token(small_tokenizer: Tokenizer) -> None:
    eot = small_tokenizer.special_tokens[EOT]
    assert small_tokenizer.decode([eot]) == EOT
    assert small_tokenizer.decode_bytes([eot]) == EOT.encode()


def test_decode_replaces_invalid_utf8(small_tokenizer: Tokenizer) -> None:
    assert small_tokenizer.decode([0xFF]) == "�"


def test_save_load_roundtrip(small_tokenizer: Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    small_tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocab == small_tokenizer.vocab
    assert loaded.merges == small_tokenizer.merges
    assert loaded.special_tokens == small_tokenizer.special_tokens
    for text in SAMPLES:
        assert loaded.encode(text) == small_tokenizer.encode(text)


def test_save_load_accepts_str_path(small_tokenizer: Tokenizer, tmp_path: Path) -> None:
    path = str(tmp_path / "tok.json")
    small_tokenizer.save(path)
    assert Tokenizer.load(path).vocab == small_tokenizer.vocab


@pytest.mark.oracle
def test_from_ranks_gpt2(gpt2_tokenizer: Tokenizer) -> None:
    assert gpt2_tokenizer.vocab_size == 50257
    assert gpt2_tokenizer.special_tokens == {EOT: 50256}


# Special-token patterns are compiled once per allowed subset and cached; none of that
# may change what encode returns.
SPECIALS = [EOT, "<|a|>", "<|b|>", "<|a|><|b|>"]
MIXED = f"x<|a|>y<|b|> z<|a|><|b|>{EOT}w"
SUBSETS: list[Literal["all"] | AbstractSet[str]] = [
    "all",
    set(),
    {"<|a|>"},
    {"<|b|>"},
    {"<|a|>", "<|b|>"},
    {"<|a|><|b|>", EOT},
    {"<|a|>", "not a special token"},
]


@pytest.fixture(scope="module")
def specials_tokenizer(small_corpus: str) -> Tokenizer:
    return Tokenizer.train(small_corpus, vocab_size=300, special_tokens=SPECIALS)


def fresh(tokenizer: Tokenizer) -> Tokenizer:
    return Tokenizer(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)


@pytest.mark.parametrize("allowed", SUBSETS)
def test_repeated_encodes_agree(
    specials_tokenizer: Tokenizer, allowed: Literal["all"] | AbstractSet[str]
) -> None:
    tok = fresh(specials_tokenizer)
    first = tok.encode(MIXED, allowed_special=allowed)
    for _ in range(3):
        assert tok.encode(MIXED, allowed_special=allowed) == first


def test_allowed_subsets_do_not_interfere(specials_tokenizer: Tokenizer) -> None:
    # Each subset on a tokenizer that has seen all the others, in two orders, agrees
    # with a tokenizer that has seen none.
    expected = [fresh(specials_tokenizer).encode(MIXED, allowed_special=a) for a in SUBSETS]
    tok = fresh(specials_tokenizer)
    for order in (range(len(SUBSETS)), reversed(range(len(SUBSETS)))):
        for i in order:
            assert tok.encode(MIXED, allowed_special=SUBSETS[i]) == expected[i]


def test_allowed_subset_ids(specials_tokenizer: Tokenizer) -> None:
    ids = specials_tokenizer.special_tokens
    tok = fresh(specials_tokenizer)
    assert tok.encode("<|a|><|b|>", allowed_special={"<|a|>", "<|b|>"}) == [
        ids["<|a|>"],
        ids["<|b|>"],
    ]
    assert tok.encode("<|a|><|b|>") == [ids["<|a|><|b|>"]]
    plain = tok.encode("<|a|><|b|>", allowed_special={EOT})
    assert not set(plain) & set(ids.values())
    assert tok.decode(plain) == "<|a|><|b|>"


def test_unknown_strings_in_allowed_special_are_ignored(specials_tokenizer: Tokenizer) -> None:
    tok = fresh(specials_tokenizer)
    assert tok.encode(MIXED, allowed_special={"<|a|>", "<|zzz|>"}) == tok.encode(
        MIXED, allowed_special={"<|a|>"}
    )


def test_changes_to_the_callers_set_are_seen(specials_tokenizer: Tokenizer) -> None:
    tok = fresh(specials_tokenizer)
    allowed = {"<|a|>"}
    only_a = tok.encode(MIXED, allowed_special=allowed)
    allowed.add("<|b|>")
    assert tok.encode(MIXED, allowed_special=allowed) == fresh(tok).encode(
        MIXED, allowed_special={"<|a|>", "<|b|>"}
    )
    allowed.clear()
    assert tok.encode(MIXED, allowed_special=allowed) == fresh(tok).encode(
        MIXED, allowed_special=set()
    )
    assert tok.encode(MIXED, allowed_special={"<|a|>"}) == only_a


def test_special_pattern_cache_is_bounded(small_corpus: str) -> None:
    size = tokenizer_module.SPECIAL_CACHE_SIZE
    specials = [f"<|s{i}|>" for i in range(size + 5)]
    tok = Tokenizer.train(small_corpus, vocab_size=300, special_tokens=specials)
    text = "".join(specials)
    for special in specials:
        ids = tok.encode(text, allowed_special={special})
        assert ids.count(tok.special_tokens[special]) == 1
        assert len(tok._special_cache) <= size
    # The first subset was evicted long ago and is compiled again, still correctly.
    first = specials[0]
    assert tok.encode(first, allowed_special={first}) == [tok.special_tokens[first]]


def test_special_patterns_are_compiled_once_on_first_use(
    specials_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled: list[frozenset[str]] = []
    original = tokenizer_module.special_token_pattern

    def counting(tokens: frozenset[str]) -> object:
        compiled.append(tokens)
        return original(tokens)

    monkeypatch.setattr(tokenizer_module, "special_token_pattern", counting)
    tok = fresh(specials_tokenizer)
    assert compiled == []  # nothing at construction
    expected = fresh(specials_tokenizer)
    for _ in range(3):
        for allowed in ("all", {"<|a|>"}):
            assert tok.encode(MIXED, allowed_special=allowed) == expected.encode(
                MIXED, allowed_special=allowed
            )
    # One compile for each of tok and expected, per subset.
    assert sorted(map(sorted, compiled)) == sorted(
        [sorted(SPECIALS), sorted(SPECIALS), ["<|a|>"], ["<|a|>"]]
    )
