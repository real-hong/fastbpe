# Review scripts (archive)

Reproductions and prototypes from the performance/correctness review of `src/` done
after commit 37339b7. They are kept as evidence, as written by the reviewer: only the
hard-coded paths were made relative. They are not linted (excluded in pyproject.toml).

Run from the repo root, e.g. `uv run python scripts/review/edges.py`.

| Script | Review finding |
|---|---|
| `edges.py` | #1 empty special token, #2 duplicate special tokens, #3 `allowed_special` strings |
| `noguard.py <spawn\|forkserver>` | #4 pool start method without a `__main__` guard |
| `daemon.py` | #4 `workers>1` inside a daemon process |
| `workers.py` | #4 fork while other threads run; parallel determinism |
| `longword.py`, `heapstats.py`, `noop.py`, `phase.py` | #5 whole-word pair updates, heap growth |
| `longword_naive.py <path>` | #5 same, on the naive baseline (see below) |
| `scaling.py`, `specials2.py` | #6 many special tokens; pre-tokenizer stays linear |
| `proto_translate.py` | #7 category-map + ASCII-regex pre-tokenizer prototype (~4.7x) |
| `proto_re.py` | #7 rejected alternative: stdlib `re` with huge classes (no gain) |
| `bpe_micro.py` | #8 `bpe_parts` local bindings / inlining (~2x on unique pieces) |
| `fuzz_pre.py`, `fuzz_cn.py` | pre-tokenizer vs `regex` (only Unicode 15.1 `Cn` differ) |
| `threads.py` | encode cache under 8 threads (no mismatches with the GIL) |
| `prof_train.txt`, `prof_encode.txt` | cProfile output (inflates fine-grained calls; see CLAUDE.md) |

`longword_naive.py` needs the baseline sources on `sys.path`:

    git worktree add /tmp/fastbpe-naive 762305e
    uv run python scripts/review/longword_naive.py /tmp/fastbpe-naive/src
