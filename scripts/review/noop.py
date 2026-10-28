import sys, time, heapq
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus, SEPARATOR
import fastbpe.train as T
last={}; stat={'push':0,'noop':0}
orig=T._PairStats.pop_best
def pop_best(self):
    for p in self._changed:
        if p in self.counts:
            stat['push']+=1
            if last.get(p)==self.counts[p]: stat['noop']+=1
            last[p]=self.counts[p]
    return orig(self)
T._PairStats.pop_best=pop_best
text=load_corpus(1_000_000)
t=time.perf_counter(); T.train_bpe(text, 5000, [SEPARATOR]); print(stat, f"{time.perf_counter()-t:.2f}")
import cProfile, pstats
T._PairStats.pop_best=orig
from fastbpe.pretokenize import count_pretokens, split_on_special_tokens
t=time.perf_counter(); chunks=[c for c in split_on_special_tokens(text,[SEPARATOR]) if c!=SEPARATOR]; pc=count_pretokens(chunks); print(f"pretok+count {time.perf_counter()-t:.3f}s")
for vs in (1000, 5000):
    t=time.perf_counter(); T.train_bpe(text, vs, [SEPARATOR]); print(vs, f"train total {time.perf_counter()-t:.3f}s")
