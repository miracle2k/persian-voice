from __future__ import annotations

from .schema import Word
from .text import persian_to_arabic_chars


def default_words() -> list[Word]:
    # Baseline 10 words (mix common/rare). Use `generate_words.py` to regenerate.
    words: list[Word] = [
        Word(id="salam", fa="سلام", ar="", fa_diac="سَلام", latn="salām", gloss_en="hello", rarity="common"),
        Word(id="ketab", fa="کتاب", ar="", fa_diac="کِتاب", latn="ketāb", gloss_en="book", rarity="common"),
        Word(id="derakht", fa="درخت", ar="", fa_diac="دِرَخت", latn="derakht", gloss_en="tree", rarity="common"),
        Word(id="ayene", fa="آینه", ar="", fa_diac="آیِنه", latn="āyene", gloss_en="mirror", rarity="common"),
        Word(id="kuh", fa="کوه", ar="", fa_diac="کُوه", latn="kuh", gloss_en="mountain", rarity="common"),
        Word(
            id="shegeftangiz",
            fa="شگفت‌انگیز",
            ar="",
            fa_diac="شِگِفت‌اَنگیز",
            latn="shegeft-angīz",
            gloss_en="amazing",
            rarity="rare",
        ),
        Word(id="zharfa", fa="ژرفا", ar="", fa_diac="ژَرفا", latn="zharfā", gloss_en="depth", rarity="rare"),
        Word(id="nilufar", fa="نیلوفر", ar="", fa_diac="نیلوفَر", latn="nilufar", gloss_en="water lily", rarity="common"),
        Word(id="qoqnus", fa="ققنوس", ar="", fa_diac="قُقنوس", latn="qoqnus", gloss_en="phoenix", rarity="rare"),
        Word(id="parcham", fa="پرچم", ar="", fa_diac="پَرچَم", latn="parcham", gloss_en="flag", rarity="common"),
    ]

    # Fill Arabic variants deterministically.
    return [
        Word(
            id=w.id,
            fa=w.fa,
            ar=w.ar or persian_to_arabic_chars(w.fa),
            fa_diac=w.fa_diac,
            latn=w.latn,
            gloss_en=w.gloss_en,
            rarity=w.rarity,
        )
        for w in words
    ]
