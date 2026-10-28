import string
from concurrent.futures import ThreadPoolExecutor

import pytest
import regex
from hypothesis import given
from hypothesis import strategies as st

from fastbpe import Tokenizer, train_bpe
from fastbpe.pretokenize import GPT2_PATTERN, count_pretokens, pretokenize

EOT = "<|endoftext|>"

# Characters assigned in this Python's Unicode database. Code points assigned only in newer
# Unicode versions are letters/numbers to ``regex`` and tiktoken but unknown to the stdlib.
assigned_text = st.text(st.characters(exclude_categories=("Cs", "Cn")))

# Words of distinct letters: every pair occurs exactly once at every step, so a word of
# length L allows exactly L - 1 merges.
distinct_words = st.lists(
    st.sampled_from(string.ascii_lowercase), min_size=2, max_size=26, unique=True
).map("".join)


@given(st.text())
def test_roundtrip_trained(small_tokenizer: Tokenizer, text: str) -> None:
    assert small_tokenizer.decode(small_tokenizer.encode(text)) == text


@pytest.mark.oracle
@given(st.text())
def test_roundtrip_gpt2(gpt2_tokenizer: Tokenizer, text: str) -> None:
    assert gpt2_tokenizer.decode(gpt2_tokenizer.encode(text)) == text


@given(st.text())
def test_roundtrip_bytes(small_tokenizer: Tokenizer, text: str) -> None:
    assert small_tokenizer.decode_bytes(small_tokenizer.encode(text)) == text.encode()


@given(word=distinct_words, with_special=st.booleans(), data=st.data())
def test_vocab_size_is_exact(word: str, with_special: bool, data: st.DataObject) -> None:
    specials = [EOT] if with_special else []
    n_merges = data.draw(st.integers(min_value=0, max_value=len(word) - 1))
    target = 256 + n_merges + len(specials)
    vocab, merges = train_bpe(word, vocab_size=target, special_tokens=specials)
    assert len(vocab) == target
    assert len(merges) == n_merges


@given(word=distinct_words, extra=st.integers(min_value=1, max_value=50))
def test_training_stops_when_pairs_run_out(word: str, extra: int) -> None:
    vocab, merges = train_bpe(word, vocab_size=256 + len(word) - 1 + extra)
    assert len(merges) == len(word) - 1
    assert len(vocab) == 256 + len(word) - 1


@given(text=st.text(max_size=300), n_merges=st.integers(min_value=0, max_value=100))
def test_merges_are_unique_and_well_formed(text: str, n_merges: int) -> None:
    vocab, merges = train_bpe(text, vocab_size=256 + n_merges)
    assert len(set(merges)) == len(merges)
    known = {bytes([i]) for i in range(256)}
    for i, (left, right) in enumerate(merges):
        assert left in known
        assert right in known
        assert vocab[256 + i] == left + right
        known.add(left + right)


@given(assigned_text)
def test_pretokenize_matches_reference_regex(text: str) -> None:
    assert pretokenize(text) == regex.findall(GPT2_PATTERN, text)


@given(st.text())
def test_pretokenize_is_lossless(text: str) -> None:
    assert "".join(pretokenize(text)) == text


# Batching and merging of the parallel path, run on in-process threads; starting real
# worker processes is covered by tests/test_parallel.py.
@given(st.lists(st.text(max_size=100), max_size=50))
def test_parallel_count_matches_serial(texts: list[str]) -> None:
    with ThreadPoolExecutor(3) as executor:
        parallel = count_pretokens(texts, workers=3, executor=executor)
    assert parallel == count_pretokens(texts, workers=1)
