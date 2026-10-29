# fastbpe

A byte-level BPE tokenizer (training, encoding, decoding) in pure Python, standard
library only. With the same merges, encoding matches tiktoken; `tests/test_oracle.py`
checks this against tiktoken's `gpt2` encoding.

## Known differences from tiktoken

- **Special tokens that are prefixes of one another.** If one special token is a
  prefix of another, both can match at the same position. fastbpe always takes the
  longest match. tiktoken (0.14.0) takes whichever token its internal hash map lists
  first. The result is the same on every run and does not depend on the ids or the
  order the tokens are given in, but it follows no rule such as longest match. With
  a byte vocabulary, `"xyz"` encodes to the special token `"xyz"` when `"xy"` is also
  special, as in fastbpe, but `"ab"` encodes to the special token `"a"` followed by
  the byte `b` when `"a"` is also special. The same happens with `"<|a|>"` and
  `"<|a|><|b|>"`. The two agree whenever no special token is a prefix of another.
  Pinned by `tests/test_tiktoken_differences.py` (seven pairs, every id assignment
  and order) and `test_overlapping_special_tokens_use_longest_match`.
- **Code points unassigned in Python's Unicode database.** Pre-tokenization classifies
  characters with the standard library's `unicodedata`: Unicode 15.0 on Python 3.12,
  15.1 on 3.13, 16.0 on 3.14. tiktoken's tables know newer code points. On 3.12 and
  3.13, a code point assigned in Unicode 16.0 (for example U+13460, Egyptian
  Hieroglyphs Extended-A) is a letter to tiktoken but category `Cn` (unassigned) here,
  so text containing it can split into different pieces, and with some vocabularies
  into different ids. On 3.14 that code point matches; code points newer than the
  running Python's Unicode version would differ the same way. All code points the
  running Python knows match (`tests/test_oracle.py`, which excludes `Cn`). Pinned by
  `tests/test_tiktoken_differences.py`, which expects the difference below Unicode 16.0
  and none from 16.0 on.

Not a difference: a special token may have the same bytes as a regular token, for
example the single-byte special token `"a"`. As in tiktoken, both ids are kept, and
`allowed_special` decides which one the text encodes to (`tests/test_tiktoken_differences.py`,
`tests/test_vocab_invariants.py`). `train_bpe` warns about single-byte special tokens
because that byte never takes part in merges (`tests/test_train.py`).
