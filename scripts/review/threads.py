import sys, threading, random, tiktoken
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus
import fastbpe.tokenizer as TK
TK.CACHE_SIZE = 50
enc=tiktoken.get_encoding("gpt2")
tok=TK.Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)
text=load_corpus(300_000); lines=text.splitlines(keepends=True)
expected=[enc.encode(l, allowed_special="all") for l in lines]
errs=[]
def work(seed):
    r=random.Random(seed); idx=list(range(len(lines))); r.shuffle(idx)
    for i in idx:
        if tok.encode(lines[i])!=expected[i]: errs.append(i)
ts=[threading.Thread(target=work,args=(s,)) for s in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
print("gil enabled:", getattr(sys, "_is_gil_enabled", lambda: True)(), "mismatches:", len(errs))
