from __future__ import annotations

import re


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
TATWEEL = re.compile(r"\u0640")
MULTI_SPACE = re.compile(r"\s+")
REPEATED_CHAR = re.compile(r"(.)\1{2,}")


def preprocess_text(text: object, normalize_arabic: bool = True) -> str:
    value = "" if text is None else str(text)
    if not normalize_arabic:
        return value.strip()

    value = re.sub(ARABIC_DIACRITICS, "", value)
    value = re.sub(TATWEEL, "", value)
    value = re.sub(REPEATED_CHAR, r"\1\1", value)
    value = re.sub(MULTI_SPACE, " ", value).strip()
    return value
