import sys, time, random
import fastbpe.train as T
holder={}
oi=T._PairStats.__init__
def init(self): oi(self); holder['s']=self
T._PairStats.__init__=init
random.seed(1)
for L in (10_000, 20_000, 40_000):
    text = "".join(random.choice("abcdefghij") for _ in range(L))  # one giant letter piece
    t=time.perf_counter(); v,m=T.train_bpe(text, 256+200); dt=time.perf_counter()-t
    print(f"one word L={L}: {dt:.2f}s heap={len(holder['s']._heap)} live={len(holder['s'].counts)}")
for L in (10_000, 20_000, 40_000):
    text = " ".join("".join(random.choice("abcdefghij") for _ in range(5)) for _ in range(L//6))
    t=time.perf_counter(); v,m=T.train_bpe(text, 256+200); dt=time.perf_counter()-t
    print(f"many words total={L}: {dt:.2f}s heap={len(holder['s']._heap)}")
