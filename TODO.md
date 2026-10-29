# TODO

Review findings #7 (category-map pre-tokenizer, 8a3d987) and #6 (special-token patterns
compiled once, 25fdd06) are done. Measure each item below against the current master.

## Hotspots after #6 and #7 (2026-10-29)

`scripts/measure_share.py` and benchmarks on ~1 MB inputs, with an estimate of what each
change could actually remove (see the lesson under the rejected item below):

- **Cold encode is BPE on distinct pieces** (`bpe_parts`): finding #8 is done, see
  its section below; it gave 1.12-1.18x on adversarial cold encode, not the ~2x the
  review measured. What is left is mostly per-byte slicing and dict lookups in Python.
- **Training on adversarial input is the merges**: 87% (base64), 91% (minified), 98%
  (CJK, 1.1 s at vocab 1000), ~100% (DNA). Not profiled yet; nothing is known about
  what is removable. Profile CJK first.
- **Many short chunks**: fixed cost per chunk, see its own item below.
- Warm encode on the corpus is 68% `pretokenize` (38 of 55 ms); the rest is cache
  lookups and `list.extend`, little to remove.

## Finding #8: `bpe_parts` (2026-10-29)

`scripts/profile_bpe.py` on the distinct pieces of 1 MB inputs (GPT-2 ranks), before:

| input | distinct | bpe_parts | setup | initial pairs | merge loop | collect parts |
|---|---|---|---|---|---|---|
| regular | 10,856 | 35 ms | 9% | 26% | 61% | 5% |
| base64 | 103,599 | 221 ms | 14% | 39% | 32% | 15% |
| minified | 19,896 | 158 ms | 4% | 25% | 67% | 4% |
| cjk | 30,207 | 151 ms | 8% | 51% | 18% | 23% |

base64: 91% of pieces under 16 bytes, 43% of initial pairs ranked, ~3 merges per piece.
cjk: average ~30 bytes, only ~10% of initial pairs ranked, so the time is the n-1
slices and lookups and collecting ~40 mostly single-byte parts, not the heap.
regular/minified: about half the heap pops are stale.

Kept (each its own commit with numbers): initial heap by one `heapify` (880d05f),
inlined re-pushes (a5b2521). Cold `Tokenizer.encode`, pytest-benchmark min, ms:

| input | cold before | cold after | | warm before | warm after | |
|---|---|---|---|---|---|---|
| regular | 94.1 | 87.6 | 1.07x | 54.2 | 55.3 | 0.98x |
| high_rep | 54.1 | 53.8 | 1.01x | 53.0 | 52.3 | 1.02x |
| specials | 188.9 | 178.4 | 1.06x | 116.0 | 116.4 | 1.00x |
| minified | 208.2 | 186.4 | 1.12x | 39.5 | 39.0 | 1.01x |
| base64 | 332.6 | 295.7 | 1.12x | 331.9 | 297.3 | 1.12x |
| cjk | 202.3 | 173.7 | 1.16x | 27.2 | 27.6 | 0.99x |
| dna (one 160 kB piece) | 2032.3 | 1959.4 | 1.04x | 2.7 | 2.7 | 0.98x |
| low_rep | 407.1 | 363.5 | 1.12x | 406.9 | 363.7 | 1.12x |
| unicode_all | 284.6 | 240.7 | 1.18x | 286.3 | 241.4 | 1.19x |
| emoji | 74.3 | 66.3 | 1.12x | 24.9 | 25.2 | 0.99x |
| whitespace | 263.9 | 230.0 | 1.15x | 23.9 | 23.8 | 1.01x |

Warm changes only where the cache cliff makes warm cold (base64, low_rep,
unicode_all); the other warm differences are noise (one cjk warm run read 31.9 ms).

Rejected, measured on `bpe_parts` over the same pieces:

- Linear-scan path for short pieces (tiktoken's `byte_pair_merge`: rank list,
  `min()` + `index()`, `del` on merge): 1.05x on base64, 0.87x on cjk, and slower than
  the heap from n ~ 64 (4x at 512 bytes). Short pieces have few merges, so the heap is
  not what costs.
- Moving per-byte work into C with iterator chains (`map(ranks.get,
  map(piece.__getitem__, slices))`, `compress`, `map(bytes, zip(...))` for pairs or
  parts): 10-40% slower. Slot-wrapper calls and iterator setup cost more than the
  specialized `piece[i:i+2]` bytecode on pieces this short.
- 65536-entry table of two-byte pair ranks: 1.01-1.06x and an extra argument on
  `encode_bytes`.
- Local bindings of `ranks.get`/`heappush`/`heappop`, reading `end_of[i]` once,
  collecting parts with a comprehension, returning early when no pair is ranked:
  each 0.93-1.01x.

Done afterwards: a piece that is itself a token is looked up first, as tiktoken does
(`Tokenizer._encode_piece`, not `bpe_parts`). Checked before implementing: every GPT-2
token that is a whole piece encodes like tiktoken, and hypothesis plus ~5.9M tokens of
~14k random trained vocabularies found no token whose own bytes BPE to anything else
(no trained vocabulary had two tokens with the same bytes either). Cold encode,
interleaved, best of 7, GC off: regular 1.18x, unicode_all 0.98x, low_rep 0.99x. With
GC on, unicode_all varied 0.93-0.95x: the lookup itself is ~4 ms of it, the rest is
when full collections run.

## Overlapping special tokens: tiktoken is no oracle

When one special token is a prefix of another, both match at one position. This
tokenizer takes the longest (`test_overlapping_special_tokens_use_longest_match`);
tiktoken takes whichever its Rust hash map lists first, so the result depends on the
tokens and their ids (byte vocab: `{"xy": 256, "xyz": 257}` gives `[257]` on "xyz",
`{"xyz": 256, "xy": 257}` gives `[256]`; `"a"`/`"ab"` gives `a` either way). The
tiktoken comparisons in `tests/test_vocab_invariants.py` skip such sets. No action
needed unless exact tiktoken parity is wanted there, which would mean copying an
order that is arbitrary.

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
