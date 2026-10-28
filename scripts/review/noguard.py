import multiprocessing as mp, sys
mp.set_start_method(sys.argv[1], force=True)
from fastbpe import train_bpe
print("top-level running in", mp.current_process().name)
v,m = train_bpe("hello world<|e|>foo bar"*50, 300, ["<|e|>"], workers=2)
print("ok", len(m))
