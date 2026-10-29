import sys, time, random
sys.path.insert(0, sys.argv[1])
import fastbpe.train as T
print(T.__file__)
random.seed(1)
for L in (10_000, 40_000):
    text = "".join(random.choice("abcdefghij") for _ in range(L))
    t=time.perf_counter(); v,m=T.train_bpe(text, 256+200); dt=time.perf_counter()-t
    print(f"one word L={L}: {dt:.2f}s")
