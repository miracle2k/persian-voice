from __future__ import annotations

from .schema import TEXT_KIND, Word


def persian_to_arabic_chars(text: str) -> str:
    # Persian Yeh / Kaf -> Arabic variants; keeps other characters as-is.
    return (
        text.replace("ی", "ي")
        .replace("ک", "ك")
        .replace("ۀ", "ة")
        .replace("هٔ", "ة")
    )


def text_for_kind(word: Word, kind: TEXT_KIND) -> str:
    if kind == "fa":
        return word.fa
    if kind == "ar":
        return word.ar
    if kind == "fa_diac":
        return word.fa_diac or word.fa
    if kind == "latn":
        return word.latn
    if kind == "fa_latn":
        return f"{word.fa} ({word.latn})"
    raise ValueError(f"Unknown text kind: {kind}")

