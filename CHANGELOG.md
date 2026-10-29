# Changelog

## Unreleased

### Training

- Merges update only the pairs next to each occurrence (words are linked token lists
  with a pair → position index), so a round costs O(k log k) for k occurrences instead
  of rewriting every word that holds the pair. Only pairs whose count changed are
  pushed, and the heap is rebuilt when dead entries outnumber live pairs 2 to 1.
  Long words without spaces (DNA, 160k chars): 8.47 s → 0.219 s (38.7x), peak memory
  157.8 → 19.6 MiB; 1 MB corpus: 1.36x (vocab 1000) and 1.46x (vocab 5000) faster.
  Output is unchanged (hypothesis oracle against the naive trainer).
- Known trade-off: a single run of one repeated byte (`"a" * n`) trains ~1.6x slower
  (160k chars: 0.124 s → 0.204 s) and long single words take more peak memory than
  before (`"a" * 160k`: 6.7 → 22.0 MiB), since every byte is a list node. Such input
  needs only ~log2(n) merges, each touching most nodes. Run-length compression of
  repeated bytes would address it; not done.

- Trained regular tokens (the 256 bytes and one per merge) have distinct bytes: a
  token forms in every word through the same merge, so no merge can rebuild it and no
  merged pair can come back. The comments that said otherwise (from the naive
  baseline) are corrected, and the dead "pair merged again, no new vocab entry" branch
  is replaced by a `RuntimeError` if a merge ever rebuilds an existing token.
- Special tokens are a namespace of their own, as in tiktoken: a single-byte special
  token (e.g. `"a"`) shares its bytes with a base token and both ids stay valid.
  `train_bpe` now warns (`UserWarning`) for such a token, since that byte is cut out
  of the training text and never takes part in merges; output is unchanged.

### Pre-tokenization

- `pretokenize` maps each character to an ASCII representative of its class (letter,
  number, whitespace, other; `'`, `sdmtlvre` and U+0020 stay themselves) with
  `str.translate` and runs an ASCII form of the GPT-2 pattern with `re`, instead of
  classifying characters one by one in Python. The map is filled lazily from
  `unicodedata` (building it for all code points at import would take ~0.9 s). Output
  is unchanged: differential tests against the old implementation (kept in
  `tests/reference_pretokenize.py`) on random Unicode and on every code point.
  1 MB corpus: pre-tokenization 4.9x, training (vocab 1000) 2.1x, encoding 2.6x cold
  and 3.6x warm. Adversarial inputs: 2.7x-4.8x for pre-tokenization alone; least with
  many distinct code points (all-plane Unicode 2.7x, 1.4x on the first call while the
  map fills) and with many tiny chunks (special token every ~16 chars: 2.8x).
- The fast path is valid only for the GPT-2 pattern: the map erases case, which
  letter or digit, and newline vs. other whitespace. Other patterns are rejected.

### Encoding

- `bpe_parts` builds its initial heap with one `heapify` and inlines the re-pushes
  after each merge. Cold encode of 1 MB, one process per measurement: cjk 1.14x,
  unicode_all 1.14x, base64 1.11x, minified 1.09x, regular 1.03x, one 160 kB DNA
  piece 1.03x; warm encode is unchanged except where the cache is too small to stay
  warm. (First reported from pytest-benchmark as regular 1.07x, cjk 1.16x,
  unicode_all 1.18x; re-measured, TODO.md.) Short-piece scan paths,
  C-level iterator chains and local bindings were measured and not kept (TODO.md).

- A pre-tokenized piece whose bytes are a token encodes to that token without running
  BPE, as tiktoken does. This is tiktoken's semantics, not only a shortcut: with a
  vocabulary holding a token that no merge reaches (e.g. `b"abc"` without `ab`/`bc`),
  `encode("abc")` now gives that token instead of three bytes. For GPT-2 ranks and for
  trained vocabularies the output is unchanged (BPE on a token's bytes ends in that
  token; `tests/test_whole_token.py`). Cold encode of 1 MB, one process per
  measurement: regular 1.17x, 50 special tokens 1.10x; low-repetition inputs with few
  whole-token pieces pay one extra lookup per distinct piece (unicode_all 0.97x,
  low_rep 0.99x, with GC off and on). Corrected: unicode_all was given as 0.98x with
  GC off and 0.93-0.95x with GC on (TODO.md); both were measured in one process,
  variants one after the other, which biased them, and re-measure as 0.97x.

- `encode` looks cached pieces up itself instead of calling a method per piece; only
  a miss calls `Tokenizer._encode_miss` (formerly `_encode_piece`). Warm encode of
  1 MB, one process per measurement: regular 1.08x, high_rep 1.11x, minified 1.09x,
  emoji and 50 special tokens 1.04x; cold regular 1.05x; inputs that nearly always
  miss the cache (base64, low_rep, unicode_all, dna) unchanged (1.00-1.01x).

- `Tokenizer(..., cache_size=...)` (also on `train`, `from_ranks` and `load`) sets how
  many encoded pieces a tokenizer keeps; the default stays 100,000 (~15-22 MB at
  ~150-220 bytes per entry). `0` disables the cache; a negative size raises
  `ValueError`, anything but an `int` (including `bool`) `TypeError`. The cache is
  still emptied when full. Keeping every other entry instead was measured and rejected:
  on a 4 MB stream of documents between never-repeating filler (`encode_stream`
  benchmark) it was 5-6% slower than emptying, with a hit rate only ~0.5 points
  higher; it won only on re-encoding the same 1 MB text (base64 1.38x, low_rep 1.43x),
  a cyclic scan that favours whatever keeps the most entries. Encode speed with the
  default size is unchanged (regular, high_rep, low_rep and the stream at 1.00x, one
  process per measurement).

### Parallel pre-tokenization

- Parallel pre-tokenization is **opt-in**: `train_bpe`, `Tokenizer.train` and
  `count_pretokens` default to `workers=1` and never start processes unless asked, so
  scripts without an `if __name__ == "__main__":` guard keep working.
- With `workers > 1`, workers are started with `mp_context`, defaulting to
  `forkserver` where the platform has it and `spawn` otherwise; never `fork`, which can
  deadlock when the calling process has other threads. Both need the `__main__` guard.
- Inside a daemonic process (e.g. a `multiprocessing.Pool` worker), `workers > 1` warns
  (`RuntimeWarning`) and counts in the current process.
- `count_pretokens(executor=...)` runs the batches on an executor you own (left open);
  it cannot be combined with `mp_context`.
- No automatic "small input → serial" threshold. Measured with
  `scripts/bench_parallel.py` (train_bpe, vocab 1000, 32-core Linux, best of 3;
  100 MB repeats the 11 MB stdlib corpus):

  | MB | serial | forkserver w=2 / 4 / 8 | spawn w=2 / 4 / 8 |
  |---|---|---|---|
  | 0.1 | 0.08 s | 1.08x / 1.11x / 1.09x | 0.78x / 0.77x / 0.76x |
  | 1 | 0.40 s | 1.32x / 1.59x / 1.65x | 1.20x / 1.42x / 1.42x |
  | 10 | 3.09 s | 1.56x / 2.11x / 2.41x | 1.52x / 2.05x / 2.33x |
  | 100 | 22.7 s | 1.96x / 3.54x / 5.83x | 1.94x / 3.50x / 5.88x |

  A cold first call (fresh process, 4 workers) is within ±10 ms of serial up to
  0.5 MB and 1.4x faster from 1 MB. The default (forkserver) is never slower, and
  spawn loses at most ~30 ms on tiny inputs; a size threshold would instead make the
  `__main__`-guard requirement appear only once inputs grow.

### Development

- Commit dcfc7d6 (on master, merged in PR #4) fails one test:
  `tests/test_vocab_invariants.py::test_special_tokens_sharing_bytes_with_regular_tokens_follow_tiktoken`
  fails on overlapping special tokens (`["ab", "a"]`) under both hypothesis profiles.
  The library code is the same as in the next commit, 1e11470, which only fixes the
  test (and TODO.md); every other test passes at dcfc7d6. History is left as is;
  bisecting across it should skip that test.
- A pre-push hook runs the fast test suite (`.pre-commit-config.yaml`; install with
  `uv run pre-commit install`), and CLAUDE.md forbids piping the output of
  pass/fail commands without `set -o pipefail`.
- Corrected: TODO.md and a test comment from PR #4 said tiktoken's choice among
  overlapping special tokens depends on their ids. It depends only on the token
  strings (README, "Known differences from tiktoken").
- The known differences from tiktoken in the README are pinned by
  `tests/test_tiktoken_differences.py`. Writing it showed that the `Cn` difference is
  gone on Python 3.14, whose `unicodedata` is Unicode 16.0; the README now says so.
- A Claude Code PreToolUse hook (`.claude/settings.json`) blocks Bash pipelines
  without `pipefail`; CLAUDE.md requires claims about external libraries to be
  backed by committed, repeatable tests or scripts.
- `scripts/time_variants.py` times variants with one process per measurement, in
  rotating order: in one process, a thinned cache that was still alive made the next
  tokenizer's warm encode up to ~24% slower (TODO.md). pyperf was considered and not
  used: it runs all processes of one benchmark before the next, so variants are not
  interleaved. `scripts/measure_cache.py` simulates eviction policies' hit rates.
  It has a `master_copy` control variant (master's code copied into the script),
  which CLAUDE.md now requires in every comparison, plus `--gc` and `--json`.
- CI also runs the tests on free-threaded Python 3.14 (`3.14t`); `tests/test_cache.py`
  shares one tokenizer between threads and checks that the GIL stays off.

### Fixes

- `encode(allowed_special=...)` accepts only `"all"` or a set; any other value
  (including other strings, which used to mean "all") raises `TypeError`.
- Empty special tokens raise `ValueError` (they used to split text into characters).
- Duplicate special tokens, two special tokens sharing an id, and a special token id
  already used by a regular token raise `ValueError`.
