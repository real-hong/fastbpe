import multiprocessing as mp
from fastbpe import train_bpe
def job():
    try:
        train_bpe("hello world<|e|>foo bar"*50, 300, ["<|e|>"], workers=2); print("daemon ok")
    except BaseException as e: print("daemon child error:", type(e).__name__, e)
if __name__ == "__main__":
    with mp.Pool(1) as p: p.apply(job)
