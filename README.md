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
- **Code points unassigned in Python's Unicode database.** Pre-tokenization classifies
  characters with the standard library's `unicodedata` (Unicode 15.1 on Python 3.13).
  tiktoken uses the `regex` crate's newer tables. A code point assigned after 15.1
  (for example U+13460, Egyptian Hieroglyphs Extended-A, Unicode 16.0) is a letter to
  tiktoken but category `Cn` (unassigned) here, so text containing such code points
  can split into different pieces. All assigned code points match; the oracle tests
  exclude `Cn`.

Not a difference: a special token may have the same bytes as a regular token, for
example the single-byte special token `"a"`. As in tiktoken, both ids are kept, and
`allowed_special` decides which one the text encodes to. `train_bpe` warns about
single-byte special tokens because that byte never takes part in merges.
