# TODO

Review findings #7 (category-map pre-tokenizer, 8a3d987) and #6 (special-token patterns
compiled once, 25fdd06) are done. Measure each item below against the current master.

## Hotspots after #6 and #7 (2026-10-29)

`scripts/measure_share.py` and benchmarks on ~1 MB inputs, with an estimate of what each
change could actually remove (see the lesson under the rejected item below):

- **Cold encode is BPE on distinct pieces** (`bpe_parts`): cold minus warm is ~180 ms
  of 210 ms on CJK, ~290 ms of 343 ms on base64 (warm there is cold, see the cache
  item), 209 ms on one 160 kB DNA piece. Review finding #8 (`scripts/review/bpe_micro.py`:
  local bindings, inlining `push`) measured ~2x on unique pieces, so roughly half of
  that is removable. Largest removable time in encoding.
- **Training on adversarial input is the merges**: 87% (base64), 91% (minified), 98%
  (CJK, 1.1 s at vocab 1000), ~100% (DNA). Not profiled yet; nothing is known about
  what is removable. Profile CJK first.
- **Many short chunks**: fixed cost per chunk, see its own item below.
- Warm encode on the corpus is 68% `pretokenize` (38 of 55 ms); the rest is cache
  lookups and `list.extend`, little to remove.

## Many short chunks: fixed cost per chunk in `count_pretokens`

Text with many special tokens is cut into many short chunks, and each chunk pays fixed
costs. 50 special tokens, one every ~16 chars (1 MB, 62,581 chunks): `count_pretokens`
117 ms vs 53 ms on the plain 1 MB corpus. Of that:

- `pretokenize` per call (translate, findall, two length checks): 74 ms over the chunks
  vs 35 ms on the plain corpus, so ~39 ms of per-call overhead.
- `_count` calls `Counter.update` with a new generator per chunk: ~22 ms (one generator
  over all chunks: 96 ms instead of 117 ms).

Removable: about the 22 ms in `_count`, plus part of the ~39 ms, e.g. by pre-tokenizing
several chunks per call. Check that the gain holds on the plain corpus and the
adversarial inputs too (few long chunks must not get slower), and keep
`tests/test_count.py` passing.

## Rejected: count `str` pieces, encode only the distinct keys

Tried counting the `str` pieces and encoding each distinct key once afterwards (keys
encoded in C: `dict.update(Counter(), zip(map(str.encode, c), c.values()))`). Counts
were identical (`tests/test_count.py`), but the change was not kept.
`count_pretokens` on ~1 MB inputs, interleaved, best of 7 (old -> new ms):

| input | pieces | distinct | old | new | |
|---|---|---|---|---|---|
| regular | 240,160 | 10,851 | 52.4 | 46.2 | 1.13x |
| high_rep | 291,667 | 8 | 54.5 | 45.7 | 1.19x |
| 50 specials | 286,187 | 19,594 | 116.1 | 103.6 | 1.12x |
| minified | 185,109 | 19,896 | 50.0 | 46.8 | 1.07x |
| base64 | 313,259 | 103,599 | 59.3 | 60.3 | 0.98x |
| cjk | 55,096 | 30,207 | 23.2 | 23.9 | 0.97x |
| unicode_all | 145,133 | 135,180 | 50.4 | 57.6 | 0.88x |
| low_rep | 153,843 | 118,253 | 33.3 | 38.2 | 0.87x |

On low-repetition input almost every piece is distinct, so no `.encode()` is saved and
the extra `str` dict (hashing non-ASCII `str` costs more than `bytes`) is pure overhead.
End to end, training on the 1 MB corpus gained only ~2%: pre-tokenization is most of
`count_pretokens` (~35 of ~53 ms).

Lesson: `measure_share.py` put encoding plus counting at ~30% of the count step, but
deduplication can only remove the encodes of *repeated* pieces, and on low-repetition
input it removes none while adding a second hash per piece. A large share is not a
large saving. Before an optimization, estimate the part it can actually remove, on the
adversarial inputs too, not the share of the step it touches.

## Encode cache: clearing everything at `CACHE_SIZE` is a cliff

`Tokenizer._encode_piece` clears the whole cache once it holds `CACHE_SIZE` (100k)
pieces. On low-repetition input whose distinct pieces exceed that, the cache is emptied
again and again and a "warm" encode is as slow as a cold one. 1 MB of base64 has
103,599 distinct pieces: cold 0.494 s, warm 0.485 s (`scripts/measure_share.py`).
`benchmarks/test_pretokenize.py` shows the same for low_rep (118,253 distinct, cold
503 ms / warm 495 ms) and unicode_all (135,180 distinct, 356 / 354 ms).
Options to compare by measurement: evicting part of the cache (e.g. the oldest half of
the insertion-ordered dict) or evicting by access order (LRU).
