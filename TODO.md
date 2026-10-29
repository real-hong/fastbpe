# TODO

Optimizations deferred until review finding #7 (category-map pre-tokenizer) is done.
Each one is measured against the baseline in place after #7, not the one before it.

## Count `str` pieces, encode only the distinct keys

`count_pretokens` (`_count`) encodes every piece to UTF-8 before counting. Counting the
`str` pieces first and encoding only the distinct keys afterwards skips most of those
`.encode()` calls (1 MB corpus: ~240k pieces, far fewer distinct ones).

## Encode cache: clearing everything at `CACHE_SIZE` is a cliff

`Tokenizer._encode_piece` clears the whole cache once it holds `CACHE_SIZE` (100k)
pieces. On low-repetition input whose distinct pieces exceed that, the cache is emptied
again and again and a "warm" encode is as slow as a cold one. 1 MB of base64 has
103,599 distinct pieces: cold 0.494 s, warm 0.485 s (`scripts/measure_share.py`).
`benchmarks/test_pretokenize.py` shows the same for low_rep (118,253 distinct, cold
503 ms / warm 495 ms) and unicode_all (135,180 distinct, 356 / 354 ms).
Options to compare by measurement: evicting part of the cache (e.g. the oldest half of
the insertion-ordered dict) or evicting by access order (LRU).
