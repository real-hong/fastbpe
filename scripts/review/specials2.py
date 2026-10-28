import time, random, string
from fastbpe.pretokenize import split_on_special_tokens
random.seed(0)
text = "".join(random.choice(string.ascii_lowercase+" ") for _ in range(200_000))
for k in (1, 10, 100, 1000):
    toks = ["".join(random.choice(string.ascii_lowercase) for _ in range(12)) for _ in range(k)]
    t=time.perf_counter(); split_on_special_tokens(text, toks); print("random-prefix specials", k, f"{time.perf_counter()-t:.3f}s")
for k in (1, 10, 100, 1000):
    toks = ["<|"+"".join(random.choice(string.ascii_lowercase) for _ in range(10))+"|>" for _ in range(k)]
    t2 = text.replace(" ", "<|")
    t=time.perf_counter(); split_on_special_tokens(t2, toks); print("'<|'-prefix specials on text full of '<|'", k, f"{time.perf_counter()-t:.3f}s")
