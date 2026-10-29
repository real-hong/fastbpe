# TODO

Optimizations deferred until review finding #7 (category-map pre-tokenizer) is done.
Each one is measured against the baseline in place after #7, not the one before it.

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
