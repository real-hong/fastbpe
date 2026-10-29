import sys, time, heapq
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus
import tiktoken
from fastbpe.encode import bpe_parts
from fastbpe.pretokenize import pretokenize
enc=tiktoken.get_encoding("gpt2"); ranks=dict(enc._mergeable_ranks)
text=load_corpus(1_000_000)
pieces=list(dict.fromkeys(p.encode() for p in pretokenize(text)))
def bpe2(piece, ranks):
    n=len(piece)
    if n < 2: return [piece]
    get=ranks.get; heappush=heapq.heappush; heappop=heapq.heappop
    if (r := get(piece)) is not None: return [piece]
    end_of=list(range(1,n+1)); start_before=list(range(-1,n-1))
    heap=[(r,i,i+1,i+2) for i in range(n-1) if (r:=get(piece[i:i+2])) is not None]
    heapq.heapify(heap)
    while heap:
        _, start, mid, end = heappop(heap)
        if end_of[start]!=mid or end_of[mid]!=end: continue
        end_of[start]=end; end_of[mid]=-1
        if start>0:
            s=start_before[start]; r=get(piece[s:end])
            if r is not None: heappush(heap,(r,s,start,end))
        if end<n:
            start_before[end]=start; e=end_of[end]; r=get(piece[start:e])
            if r is not None: heappush(heap,(r,start,end,e))
    parts=[]; i=0
    while i<n:
        j=end_of[i]; parts.append(piece[i:j]); i=j
    return parts
for f in (bpe_parts,bpe2,bpe_parts,bpe2):
    t=time.perf_counter(); out=[f(p,ranks) for p in pieces]; print(f.__name__, f"{time.perf_counter()-t:.3f}s", len(pieces))
assert [bpe_parts(p,ranks) for p in pieces]==[bpe2(p,ranks) for p in pieces]
# whole-piece-in-vocab fraction
print("fraction unique pieces that are a single token:", sum(p in ranks for p in pieces)/len(pieces))
