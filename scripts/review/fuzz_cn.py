import regex, unicodedata
from fastbpe.pretokenize import pretokenize, GPT2_PATTERN
pat = regex.compile(GPT2_PATTERN)
d=[cp for cp in range(0x110000) for t in [" "+chr(cp)+"a"] if pretokenize(t)!=pat.findall(t)]
print(len(d), sum(unicodedata.category(chr(c))!="Cn" for c in d))
