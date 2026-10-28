import tiktoken, os
from fastbpe import Tokenizer, train_bpe
from fastbpe.pretokenize import split_on_special_tokens
enc=tiktoken.get_encoding("gpt2")
tok=Tokenizer.from_ranks(enc._mergeable_ranks, enc._special_tokens)
s="hi<|endoftext|>there"
print("allowed_special='none' ->", tok.encode(s, allowed_special="none"))  # type: ignore
print("allowed_special=set() ->", tok.encode(s, allowed_special=set()))
print("tiktoken disallowed=() allowed=set():", enc.encode(s, disallowed_special=()))
# duplicate special tokens in training
t = Tokenizer.train("abc <x> abc", 256+3+2, ["<x>", "<x>"])
print("dup specials: special_tokens", t.special_tokens, "vocab tail", {k:v for k,v in t.vocab.items() if k>=256})
print("  ranks has b'<x>'?", t._ranks.get(b"<x>"), " encode('<x>', allowed=set()) ->", t.encode("<x>", allowed_special=set()))
# empty special token
print("split with '' special:", split_on_special_tokens("ab cd", [""]))
t2 = Tokenizer.train("ab cd ab cd", 256+2, [""])
print("empty special encode 'ab cd':", t2.encode("ab cd"), "vs without allowing:", t2.encode("ab cd", allowed_special=set()))
# overlapping special tokens, leftmost vs longest
print("overlap:", split_on_special_tokens("abcd", ["ab","bcd"]))
# save/load roundtrip w/ special token
import tempfile
p=os.path.join(tempfile.mkdtemp(), "tok.json"); tok.save(p); t3=Tokenizer.load(p)
print("roundtrip eq", t3._ranks==tok._ranks and t3.special_tokens==tok.special_tokens and t3.merges==tok.merges)
# from_ranks where special id collides with a mergeable rank
tc = Tokenizer.from_ranks({bytes([i]):i for i in range(256)}|{b"ab":256}, {"<s>":256})
print("special id collides with rank 256: ranks has b'ab'?", b"ab" in tc._ranks, "encode('ab') ->", tc.encode("ab"))
