"""Hypothesis strategies for text at the edges of the pre-tokenizer's classes."""

from hypothesis import strategies as st

# Characters at the edges of the pattern's classes, written as escapes to keep them
# readable. Hypothesis rarely draws most of them from the full range on its own.
WHITESPACE = [
    *"\t\n\x0b\x0c\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000",
    *map(chr, range(0x2000, 0x200B)),  # EN QUAD .. HAIR SPACE
]
# Not White_Space, though str.isspace() or their names suggest otherwise.
NOT_WHITESPACE = [*"\x1c\x1d\x1e\x1f\u180e\u200b\u200c\u200d\u2060\ufeff"]
CONTRACTION = [*"'sdmtlvreSDMTLVRE", "\u2019"]  # and RIGHT SINGLE QUOTATION MARK
LETTERS = [
    *"aZ\xe9\xdf\u01c5\u02b0\xaa\u4e2d\u3042\ud55c\u05d0\u0627",  # Lu Ll Lt Lm Lo, CJK
    "\U0001d400",  # MATHEMATICAL BOLD CAPITAL A, outside the BMP
    "\U00020000",  # CJK EXTENSION B
]
MARKS = [*"\u0301\u0903\u20dd\u0e31"]  # Mn Mc Me, Thai vowel sign
NUMBERS = [*"09\u0663\u216b\xb2\xbd\u3007", "\U0001d7ce"]  # Nd Nl No, astral digit
OTHER = [
    *"!_-.,$^\xac\xa9\uff0c\u3002\x00\x7f",
    "\ue000",  # private use (Co)
    "\U000f0000",  # supplementary private use
    "\u0378",  # unassigned (Cn)
    "\U0010ffff",  # last code point, a noncharacter
    "\ud800",  # lone surrogates (Cs)
    "\udfff",
]
EMOJI = ["\U0001f44d", "\U0001f3fd", "\u200d", "\u2764", "\ufe0f", "\U0001f1e8", "\U0001f1f3"]
INTERESTING = WHITESPACE + NOT_WHITESPACE + CONTRACTION + LETTERS + MARKS + NUMBERS + OTHER + EMOJI

# Multi-character pieces around the contraction and whitespace alternatives.
FRAGMENTS = [
    *("'s", "'d", "'m", "'t", "'ll", "'ve", "'re", "'S", "'LL", "'l", "'v", "'r"),
    *(" ", "  ", "   ", "\n", "\n\n", " \n", "\n ", "\r\n", " \t ", "\u3000 "),
    *("ab", " ab", "12", " 12", "!!", " !!", "e\u0301", "\U0001f468\u200d\U0001f469"),
]

texts = st.one_of(
    st.text(st.characters(exclude_categories=())),  # any code point, surrogates included
    st.text(st.sampled_from(INTERESTING)),
    st.lists(st.sampled_from(FRAGMENTS + INTERESTING)).map("".join),
)
