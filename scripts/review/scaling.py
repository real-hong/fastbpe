import time, random, re
from fastbpe.pretokenize import pretokenize, split_on_special_tokens
def tm(f,*a):
    t=time.perf_counter(); f(*a); return time.perf_counter()-t
for n in (100_000, 400_000):
    print("spaces", n, f"{tm(pretokenize, ' '*n+'x'):.3f}", "letters", f"{tm(pretokenize, 'a'*n):.3f}", "mixed ws", f"{tm(pretokenize, ' \n'*(n//2)+'x'):.3f}")
# many special tokens, html-like text
random.seed(0)
text = "<div class='x'>hello</div> " * 20000
for k in (1, 10, 100, 1000):
    toks = [f"<|special_{i}|>" for i in range(k)]
    print("specials", k, f"{tm(split_on_special_tokens, text, toks):.3f}s len={len(text)}")
toks = [f"<|special_{i}|>" for i in range(1000)]
t=time.perf_counter(); 
for _ in range(1000): split_on_special_tokens("short text", toks)
print("1000 calls short text w/ 1000 specials:", f"{time.perf_counter()-t:.3f}s")
t=time.perf_counter(); 
for _ in range(1000): split_on_special_tokens("short text", ["<|endoftext|>"])
print("1000 calls short text w/ 1 special:", f"{time.perf_counter()-t:.3f}s", "re cache size", re._MAXCACHE)
