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

### Fixes

- `encode(allowed_special=...)` accepts only `"all"` or a set; any other value
  (including other strings, which used to mean "all") raises `TypeError`.
- Empty special tokens raise `ValueError` (they used to split text into characters).
- Duplicate special tokens, two special tokens sharing an id, and a special token id
  already used by a regular token raise `ValueError`.
