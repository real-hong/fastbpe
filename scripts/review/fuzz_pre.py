import random, regex, sys
from fastbpe.pretokenize import pretokenize, GPT2_PATTERN
pat = regex.compile(GPT2_PATTERN)
alpha = list(" \t\n\r\x0b\x0c\x1c\x1d\x85\xa0 　᠎​'sdmtlvreA1é٣_!.,-") + ["\U0001F600", "́", "Ⅻ", "²", "﻿"]
random.seed(0)
bad = 0
for _ in range(200000):
    t = "".join(random.choice(alpha) for _ in range(random.randint(0, 12)))
    if pretokenize(t) != pat.findall(t):
        bad += 1
        if bad < 6: print(repr(t), pretokenize(t), pat.findall(t))
# all code points as single chars and with surrounding
diff=[]
for cp in range(0x110000):
    c = chr(cp)
    for t in (c, " "+c+"a", "a"+c+" "):
        try:
            if pretokenize(t) != pat.findall(t):
                diff.append(cp); break
        except Exception as e:
            diff.append(cp); break
print("random mismatches", bad, "codepoint mismatches", len(diff), [hex(x) for x in diff[:20]])
import unicodedata; print(unicodedata.unidata_version, regex.__version__)
