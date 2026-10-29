# TODO

Review findings #7 (category-map pre-tokenizer, 8a3d987) and #6 (special-token patterns
compiled once, 25fdd06) are done. Measure each item below against the current master.

## Hotspots after #6 and #7 (2026-10-29)

`scripts/measure_share.py` and benchmarks on ~1 MB inputs, with an estimate of what each
change could actually remove (see the lesson under the rejected item below):

- **Cold encode is BPE on distinct pieces** (`bpe_parts`): finding #8 is done, see
  its section below; it gave 1.09-1.14x on adversarial cold encode (re-measured one
  process per measurement), not the ~2x the review measured. What is left is mostly per-byte slicing and dict lookups in Python.
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

Re-measured on master (2026-10-29) with `scripts/time_variants.py --reps 7 --variants
clear,master_copy,bpe_pre,bpe_heapify,no_whole`: one process per measurement, GC off,
master's `bpe_parts` against copies of the code before 880d05f (`bpe_pre`) and before
a5b2521 (`bpe_heapify`). The control `master_copy` (master's `_encode_piece` copied
into the script) was 0.99-1.01x of `clear` on every input. Cold encode, master faster
by (min; medians agree within 0.01):

| input | before | re-measured | heapify | inlined re-pushes |
|---|---|---|---|---|
| base64 | 1.12x | 1.11x | 1.03x | 1.08x |
| minified | 1.12x | 1.09x | 1.04x | 1.05x |
| cjk | 1.16x | 1.14x | 1.08x | 1.05x |
| unicode_all | 1.18x | 1.14x | 1.05x | 1.09x |
| whitespace | 1.15x | 1.13x | 1.07x | 1.06x |
| emoji | 1.12x | 1.09x | 1.05x | 1.04x |
| low_rep | 1.12x | 1.12x | 1.04x | 1.08x |
| dna | 1.04x | 1.03x | | |
| regular | 1.07x | 1.03x | 1.00x | 1.03x |
| specials | 1.06x | 1.03x | | |
| high_rep | 1.01x | 1.00x | | |
| stream, pass 1 (4 MB) | | 1.10x | | |

Warm where the cache cliff makes it cold: base64 1.10x, low_rep 1.10x, unicode_all
1.16x. The gain holds, 0-4 points below the pytest-benchmark numbers. regular shrank
to under half of its gain, specials to half. Unverified guess: the old numbers predate the
whole-piece lookup, which now skips BPE for many regular pieces, leaving less for
`bpe_parts` to save there.

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
interleaved in one process, best of 7, GC off: regular 1.18x, unicode_all 0.98x,
low_rep 0.99x; with GC on, unicode_all 0.93-0.95x.

Re-measured (2026-10-29, same run as finding #8 above, variant `no_whole`; one
process per measurement): cold regular 1.17x, specials 1.10x, corpus documents of the
stream 1.17-1.20x (whole stream 1.03x). Costs one lookup per distinct piece: cold
unicode_all 0.97x, base64, cjk, emoji, whitespace, dna and low_rep 0.99x; warm
unchanged (0.99-1.01x) except on the cache-cliff inputs (unicode_all 0.97x, low_rep
and base64 0.99x). With GC on (`--gc`, regular, unicode_all, low_rep, specials):
unicode_all 0.97x cold and warm, the rest as with GC off. The earlier 0.93-0.95x was
measured in one process, variants one after the other, and does not reproduce.

## Overlapping special tokens: tiktoken is no oracle

When one special token is a prefix of another, both match at one position. This
tokenizer takes the longest (`test_overlapping_special_tokens_use_longest_match`);
tiktoken 0.14.0 takes whichever its Rust hash map lists first. That depends only on
the token strings, not on their ids or order (`"xy"`/`"xyz"`, `"q"`/`"qq"`: longest;
`"a"`/`"ab"`, `"<|a|>"`/`"<|a|><|b|>"`: shortest), so it follows no rule. The
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

## Done: encode cache, keep emptying it; `cache_size` argument (2026-10-29)

`Tokenizer._encode_piece` clears the whole cache once it holds `CACHE_SIZE` (100k)
pieces. On low-repetition input whose distinct pieces exceed that, the cache is emptied
again and again and a "warm" encode is as slow as a cold one. 1 MB of base64 has
103,599 distinct pieces: cold 0.494 s, warm 0.485 s (`scripts/measure_share.py`).
`benchmarks/test_pretokenize.py` shows the same for low_rep (118,253 distinct, cold
503 ms / warm 495 ms) and unicode_all (135,180 distinct, 356 / 354 ms).
Options to compare by measurement: evicting part of the cache (e.g. the oldest half of
the insertion-ordered dict) or evicting by access order (LRU).

Outcome: emptying stays; `cache_size` (default 100k, 0 = off) lets callers size the
cache. The warm "cliff" on 1 MB inputs is a cyclic scan (the same text again), the worst
case for LRU and FIFO and the best for whatever keeps the most entries; it says little
about a long-running encoder. Raising the default to 2^17 was rejected: it only fits
the three 1 MB inputs (103-135k distinct), helps nothing at 4 MB, and costs ~10 MB more
per tokenizer and process.

Simulated hit rates (`scripts/measure_cache.py`, capacity 100k), second pass of the same
1 MB text / 4 MB stream, corpus pieces:

| policy | base64 | low_rep | unicode_all | stream |
|---|---|---|---|---|
| empty when full | 67% | 23% | 6% | 95.2% |
| evict oldest half | 67% | 23% | 7% | 95.4% |
| keep every other | 83% | 55% | 33% | 95.8% |
| LRU | 68% | 26% | 9% | 96.7% |
| stop inserting (reference) | 99% | 88% | 75% | 97.1% |

Keeping every other entry ("thin", the only one that helped the 1 MB inputs) was timed
against emptying with `scripts/time_variants.py`, 7 processes each, min ms,
ratio = empty / thin:

| case | empty | thin | |
|---|---|---|---|
| stream pass 1 | 802.7 | 848.2 | 0.95x |
| stream pass 2 | 791.7 | 846.5 | 0.94x |
| stream pass 2, corpus docs | 150.1 | 167.2 | 0.90x |
| base64 warm | 291.3 | 211.8 | 1.38x |
| low_rep warm | 356.0 | 248.8 | 1.43x |
| unicode_all cold / warm | 269.9 / 232.5 | 292.0 / 217.9 | 0.92x / 1.07x |
| regular warm | 53.5 | 53.4 | 1.00x |
| regular next to a thinned tokenizer, warm | 54.7 | 54.9 | 1.00x |

Rejected: slower on the stream, the case that matters, for ~0.5 points of hit rate.
Where the cost comes from (a lock on every insert, copying half the dict on each thin,
the thinned cache's own locality) was not broken down. Cold, thin was slower where it
evicts (base64 0.95x, low_rep 0.96x); the other 1 MB inputs were 0.97-1.00x cold and
1.00-1.01x warm.

### Measuring in one process misleads (2026-10-29)

Variants timed one after another in one process were off by up to ~25% on low_rep warm
encode, with the same code path for both (no eviction at that capacity). Cause: the
previous input's tokenizer was still alive, its cache thinned (every other entry
freed). The next cache filled those holes and spread over more pages. Separate process,
low_rep cache of 118,253 entries, after a unicode_all run whose thinned tokenizer is
kept alive or not (scratch script, numbers from one run each):

| variant before | kept alive | key pages | value pages | warm |
|---|---|---|---|---|
| clear | no | 2,053 | 2,700 | 43.3 ms |
| clear | yes | 2,053 | 2,757 | 43.7 ms |
| thin | no | 2,053 | 2,699 | 43.4 ms |
| thin | yes | 3,035 | 3,879 | 53.7 ms |

Same with `PYTHONMALLOC=malloc`, so not specific to pymalloc. `scripts/time_variants.py`
now runs every (input, variant) pair in its own process, rotating the variant order
each repetition; pyperf was not used because it runs all processes of one benchmark
before the next, so variants are not interleaved and drift enters the comparison.

Re-checked (2026-10-29): finding #8 and the whole-piece lookup, both measured in one
process before, hold when measured one process per measurement (numbers in their
sections above). Noise on this machine, from the `master_copy` control: 0.99-1.01x
with GC off, up to 1.03x on one input with GC on.

## Done: inline the cache lookup in `encode` (2026-10-29)

Every piece cost a Python method call to `_encode_piece` even on a hit. `encode` now
binds the cache's `get` once per call and calls `_encode_miss` only on a miss (which
looks the cache up no second time). Safe because a full cache is emptied in place,
never replaced (`test_full_cache_is_emptied_in_place`).

`scripts/time_variants.py`, 7 processes per (input, variant), GC off, two runs:

- Before the change, variant `inline` vs `master_copy`: regular warm 1.087x
  [median 1.084x]. That run was noisy: `master_copy` vs `clear` 0.987-1.015 (medians
  0.981-1.014), per-variant spread up to 27-42% on cjk. Cause not found.
- After the change, `clear` and `master_copy` (new master) vs `pre_inline` (the code
  before), machine idle (load at most 1.00 on 32 cores, sampled every 5 s; CPU
  frequency governor powersave, EPP `power`): `master_copy` vs `clear` 0.986-1.006
  (medians 0.984-1.013), spreads 1-4% except high_rep warm 8% and dna warm (3.8 ms) 7%.

New vs old, second run (min [median]):

| input | cold | warm |
|---|---|---|
| regular | 1.05x [1.05x] | 1.08x [1.09x] |
| regular next to a tokenizer that encoded unicode_all | 1.05x [1.05x] | 1.09x [1.08x] |
| high_rep | 1.10x [1.10x] | 1.11x [1.10x] |
| minified | 1.01x | 1.09x [1.08x] |
| emoji | 1.01x | 1.04x [1.04x] |
| specials | 1.02x | 1.04x [1.03x] |
| cjk | 1.00x | 1.03x [1.03x] |
| stream corpus documents, pass 1 / 2 | 1.05x / 1.05x | |
| base64, whitespace, low_rep, unicode_all, dna, stream filler | 1.00-1.01x | 1.00-1.01x |

The gain is on cache hits; where nearly every piece misses, it is within the noise.
A per-instance `functools.lru_cache` (a C call) had pointed at the method call
(1.04-1.05x, one process); it was not pursued further.

## Release preparation (not started)

- Add `src/fastbpe/py.typed` (PEP 561) and check that the built wheel contains it.
  Without it, mypy run on code outside `src` (e.g. `uv run mypy --strict
  scripts/time_variants.py`) skips `fastbpe` as untyped (`import-untyped`), so
  subclasses of `Tokenizer` are checked against `Any`.
- Decide whether CI also runs mypy on `scripts/` and `tests/` (not necessarily
  `--strict`); today only `src` is checked.
