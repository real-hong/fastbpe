import sys, time, re, unicodedata, random
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus, SEPARATOR
from fastbpe.pretokenize import pretokenize, split_on_special_tokens, _is_space
class Table(dict):
    def __missing__(self, cp):
        c = chr(cp)
        if _is_space(c): v = "\t"
        else:
            cat = unicodedata.category(c)[0]
            v = "A" if cat == "L" else "1" if cat == "N" else "!"
        self[cp] = v; return v
table = Table()
for cp in range(128):
    c=chr(cp)
    if "\x1c"<=c<="\x1f": table[cp]="!"
    elif c.isspace() or c.isalnum() or c=="'": table[cp]=c
    else: table[cp]="!" if not c.isalpha() else c
pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]++| ?[0-9]++| ?[^\sA-Za-z0-9]++|\s++\Z|\s+(?!\S)|\s", re.ASCII)
def pt2(text):
    cls = text.translate(table)
    return [text[m.start():m.end()] for m in pat.finditer(cls)]
text = load_corpus(1_000_000)
chunks = [c for c in split_on_special_tokens(text,[SEPARATOR]) if c!=SEPARATOR]
for _ in range(2):
    t=time.perf_counter(); a=[pretokenize(c) for c in chunks]; t1=time.perf_counter()-t
    t=time.perf_counter(); b=[pt2(c) for c in chunks]; t2=time.perf_counter()-t
    print(f"current {t1:.3f}s translate+ASCII-re {t2:.3f}s equal={a==b}")
alpha = list(" \t\n\r\x0b\x0c\x1c\x85\xa0　'sdmtlvreA1é٣_!.²́")
random.seed(0)
print("fuzz eq", all(pt2(s)==pretokenize(s) for s in ("".join(random.choice(alpha) for _ in range(random.randint(0,12))) for _ in range(200000))))
