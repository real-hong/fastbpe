import sys, time
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus, SEPARATOR
import fastbpe.train as T
acc={'pop':0.0,'add':0.0,'rem':0.0,'merge':0.0}
def wrap(name, f):
    def g(*a):
        t=time.perf_counter(); r=f(*a); acc[name]+=time.perf_counter()-t; return r
    return g
T._PairStats.pop_best=wrap('pop',T._PairStats.pop_best)
T._PairStats.add=wrap('add',T._PairStats.add)
T._PairStats.remove=wrap('rem',T._PairStats.remove)
T._merge_word=wrap('merge',T._merge_word)
text=load_corpus(1_000_000)
t=time.perf_counter(); T.train_bpe(text, int(sys.argv[1]), [SEPARATOR]); tot=time.perf_counter()-t
print(f"total {tot:.3f}s", {k:round(v,3) for k,v in acc.items()})
