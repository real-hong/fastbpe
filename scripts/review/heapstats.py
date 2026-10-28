import sys, time, heapq, random
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus, SEPARATOR
import fastbpe.train as T
pushes = 0; noop = 0
orig_add, orig_remove, orig_pop = T._PairStats.add, T._PairStats.remove, T._PairStats.pop_best
def pop_best(self):
    global pushes
    pushes += sum(1 for p in self._changed if p in self.counts)
    return orig_pop(self)
T._PairStats.pop_best = pop_best
def run(text, vs, specials):
    global pushes; pushes = 0
    holder = {}
    orig_init = T._PairStats.__init__
    def init(self):
        orig_init(self); holder['s'] = self
    T._PairStats.__init__ = init
    t=time.perf_counter(); v,m = T.train_bpe(text, vs, specials); dt=time.perf_counter()-t
    T._PairStats.__init__ = orig_init
    s = holder['s']
    print(f"vocab={vs} time={dt:.2f}s merges={len(m)} pushes={pushes} final_heap={len(s._heap)} live_pairs={len(s.counts)}")
text = load_corpus(int(sys.argv[1]))
for vs in map(int, sys.argv[2:]):
    run(text, vs, [SEPARATOR])
