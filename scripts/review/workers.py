import sys, time, warnings, threading, multiprocessing as mp
sys.path.insert(0, __file__.rsplit("/review/", 1)[0])  # scripts/, for corpus.py
from corpus import load_corpus, SEPARATOR
from fastbpe import train_bpe
from fastbpe.pretokenize import count_pretokens, split_on_special_tokens
if __name__ == "__main__":
    text=load_corpus(1_000_000)
    chunks=[c for c in split_on_special_tokens(text,[SEPARATOR]) if c!=SEPARATOR]
    print("chunks", len(chunks), "default start method", mp.get_start_method())
    for w in (1,2,4,8):
        t=time.perf_counter(); r=train_bpe(text, 1000, [SEPARATOR], workers=w); dt=time.perf_counter()-t
        t=time.perf_counter(); c=count_pretokens(chunks, workers=w); dc=time.perf_counter()-t
        if w==1: base=r; basec=c
        print(f"workers={w} train {dt:.3f}s count_pretokens {dc:.3f}s same_merges={r==base} same_counter_order={list(c.items())==list(basec.items())}")
    # call from a thread
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        th=threading.Thread(target=lambda: time.sleep(2)); th.start()
        count_pretokens(chunks[:20], workers=2); th.join()
        print("warnings while another thread alive:", [str(w.message)[:90] for w in ws])
