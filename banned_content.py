import re
import unicodedata

# P2.7: Add common plurals/variants to BANNED_DISPLAY_WORDS.
# \b prevents "scams" matching "scam". Rather than changing the regex
# (which risks false positives), add explicit variants.
BANNED_DISPLAY_WORDS: frozenset[str] = frozenset({
    "fraud",
    "scam", "scams",
    "illegal",
    "banned",
    "weapon", "weapons",
    "drugs", "drug",
    "abuse",
    "porn", "pornography",
    "escort", "escorts", "escorting",
    "gambling", "gamble",
    "casino", "casinos",
})

# FIX-13: Pre-compiled banned-word pattern — longest-first ordering prevents
# shorter words from shadowing longer compounds.
# FIX-13: Uses (?<!\w)/(?!\w) instead of \b for word boundary matching.
# \b fails when keywords start/end with non-word characters.
_ordered_banned = sorted(BANNED_DISPLAY_WORDS, key=len, reverse=True)
BANNED_PATTERN = re.compile(
    r'(?<!\w)(?:' + '|'.join(map(re.escape, _ordered_banned)) + r')(?!\w)',
    re.IGNORECASE,
)

# FIX-12: Explicit confusables table for Cyrillic lookalikes.
# NFKC normalization does NOT convert these — they are distinct Unicode
# codepoints that happen to be visually identical to Latin characters.
_CONFUSABLES = str.maketrans({
    "\u0455": "s",  # Cyrillic small dze  → Latin s
    "\u0441": "c",  # Cyrillic small es   → Latin c
    "\u0430": "a",  # Cyrillic small a    → Latin a
    "\u0435": "e",  # Cyrillic small ie   → Latin e
    "\u043E": "o",  # Cyrillic small o    → Latin o
    "\u0440": "p",  # Cyrillic small er   → Latin p
    "\u0445": "x",  # Cyrillic small ha   → Latin x
    "\u0443": "y",  # Cyrillic small u    → Latin y
    "\u0405": "S",  # Cyrillic capital DZE
    "\u0421": "C",  # Cyrillic capital ES
    "\u0410": "A",  # Cyrillic capital A
    "\u0415": "E",  # Cyrillic capital IE
    "\u041E": "O",  # Cyrillic capital O
    "\u0420": "P",  # Cyrillic capital ER
    "\u0425": "X",  # Cyrillic capital HA
    "\u0423": "Y",  # Cyrillic capital U
})


# FIX 25: Exact supported threat model documentation.
# Coverage: Common Latin variants, explicit Cyrillic lookalikes (_CONFUSABLES),
# and zero-width character stripping.
# NOT covered: Greek/Armenian homoglyphs, advanced bidirectional overrides.
# Adding broad regex obfuscation causes false positives, so we only add terms
# based on observed real-world misses.
def _normalize_display_text(text: str) -> str:
    """NFKC + Cyrillic confusable mapping for banned-content detection."""
    text = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES).lower()
    # F2: Strip zero-width characters
    text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)
    return text

# Blocker 15: Safe terms for compact substring matching
_COMPACT_SAFE_TERMS = frozenset({
    "fraud", "scam", "scams", "porn", "pornography",
    "escort", "escorts", "escorting", "casino", "casinos",
})

# FIX-10: Narrowed separator to whitespace, punctuation, and zero-width chars to prevent
# matching across words (e.g. "spoke person" matching "porn").
_SEP = r"[\s._\-\u200B\u200C\u200D\uFEFF]*"

def _obfuscated_word_pattern(word: str) -> str:
    chars = [re.escape(c) for c in word]
    return r"(?<![a-z])" + _SEP.join(chars) + r"(?![a-z])"

_OBFUSCATED_SAFE_PATTERN = re.compile(
    "|".join(_obfuscated_word_pattern(term) for term in sorted(_COMPACT_SAFE_TERMS, key=len, reverse=True)),
    re.IGNORECASE,
)

# FIX-13: Public API — replaces the private _contains_banned_content that was
# previously defined in passion_insight_generator.py.
def contains_banned_content(text: str) -> bool:
    """Check if text contains any banned display words after normalization."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    normalized = _normalize_display_text(text)
    if BANNED_PATTERN.search(normalized):
        return True

    return bool(_OBFUSCATED_SAFE_PATTERN.search(normalized))


__all__ = ["BANNED_DISPLAY_WORDS", "BANNED_PATTERN", "contains_banned_content"]
