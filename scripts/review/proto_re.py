import sys, time, re, unicodedata, regex
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus, SEPARATOR
from fastbpe.pretokenize import pretokenize, split_on_special_tokens, _is_space, GPT2_PATTERN
from fastbpe import train_bpe, Tokenizer
def ranges(pred):
    out=[]; start=None
    for cp in range(0x110000+1):
        ok = cp < 0x110000 and not (0xD800<=cp<=0xDFFF) and pred(chr(cp))
        if ok and start is None: start=cp
        if not ok and start is not None: out.append((start,cp-1)); start=None
    return "".join(re.escape(chr(a)) if a==b else re.escape(chr(a))+"-"+re.escape(chr(b)) for a,b in out)
t0=time.perf_counter()
L = ranges(lambda c: unicodedata.category(c)[0]=="L")
N = ranges(lambda c: unicodedata.category(c)[0]=="N")
SP = ranges(_is_space)
pat = re.compile(rf"'(?:[sdmt]|ll|ve|re)| ?[{L}]++| ?[{N}]++| ?[^{SP}{L}{N}]++|[{SP}]++\Z|[{SP}]+(?![^{SP}])|[{SP}]")
print(f"build {time.perf_counter()-t0:.2f}s")
text = load_corpus(1_000_000)
chunks = [c for c in split_on_special_tokens(text,[SEPARATOR]) if c!=SEPARATOR]
t=time.perf_counter(); a=[pretokenize(c) for c in chunks]; t1=time.perf_counter()-t
t=time.perf_counter(); b=[pat.findall(c) for c in chunks]; t2=time.perf_counter()-t
print(f"pretokenize current {t1:.3f}s  stdlib-re prototype {t2:.3f}s equal={a==b}")
t=time.perf_counter(); train_bpe(text, 1000, [SEPARATOR]); print(f"train_bpe total {time.perf_counter()-t:.3f}s")
import tiktoken
enc=tiktoken.get_encoding("gpt2")
tok=Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)
t=time.perf_counter(); ids=tok.encode(text); print(f"encode total (cold cache) {time.perf_counter()-t:.3f}s")
t=time.perf_counter(); ids2=tok.encode(text); print(f"encode total (warm cache) {time.perf_counter()-t:.3f}s")
print("matches tiktoken:", ids==enc.encode(text, allowed_special="all"))
# fuzz equality
import random
alpha = list(" \t\n\r\x0b\x0c\x1c\x85\xa0　'sdmtlvreA1é٣_!.")
random.seed(0)
print("fuzz eq", all(pat.findall(s)==pretokenize(s) for s in ("".join(random.choice(alpha) for _ in range(random.randint(0,12))) for _ in range(100000))))
