from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from persian_voice.storage import public_dir, write_json_atomic
from persian_voice.text import persian_to_arabic_chars
from persian_voice.schema import Word
from persian_voice.wordlist import default_words


def _slugify_id(value: str) -> str:
    value = value.strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _dedupe_ids(words: list[Word]) -> list[Word]:
    seen: set[str] = set()
    out: list[Word] = []
    for w in words:
        base = _slugify_id(w.latn) or _slugify_id(w.fa) or "word"
        wid = base
        i = 2
        while wid in seen:
            wid = f"{base}_{i}"
            i += 1
        seen.add(wid)
        out.append(
            Word(
                id=wid,
                fa=w.fa,
                ar=w.ar,
                fa_diac=w.fa_diac,
                latn=w.latn,
                gloss_en=w.gloss_en,
                rarity=w.rarity,
            )
        )
    return out


def _generate_words_llm(*, count: int, model: str) -> list[Word]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (required for --llm).")

    client = OpenAI()
    prompt = f"""
Generate exactly {count} Persian (فارسی) words for a TTS comparison dataset.

Requirements:
- Mix common and rare words (include a \"rarity\" field: \"common\" or \"rare\").
- Return ONLY valid JSON (no markdown).
- Output format: {{ "words": [ ... ] }}.
- Each item must have:
  - fa: Persian script (use Persian forms like ی and ک when applicable)
  - fa_diac: same word but with vowel diacritics (harakat) where appropriate
  - latn: Latin transliteration (use a readable scholarly style, e.g. ā ī ū where needed)
  - gloss_en: short English gloss
  - rarity: "common" | "rare"
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content.")
    data = json.loads(content)
    raw_words = data.get("words")
    if not isinstance(raw_words, list):
        raise RuntimeError("Expected JSON object with key 'words' as a list.")

    words: list[Word] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        fa = str(item.get("fa", "")).strip()
        if not fa:
            continue
        fa_diac = str(item.get("fa_diac", "")).strip() or fa
        latn = str(item.get("latn", "")).strip()
        gloss_en = str(item.get("gloss_en", "")).strip() or None
        rarity = str(item.get("rarity", "")).strip().lower() or None
        words.append(
            Word(
                id="",
                fa=fa,
                ar=persian_to_arabic_chars(fa),
                fa_diac=fa_diac,
                latn=latn,
                gloss_en=gloss_en,
                rarity=rarity,
            )
        )

    if len(words) != count:
        raise RuntimeError(f"Expected {count} words, got {len(words)}.")
    return _dedupe_ids(words)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the baseline Persian word list (words.json).")
    parser.add_argument(
        "--out",
        type=str,
        default="data/words.json",
        help="Path relative to public dir (default: data/words.json).",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of words (used with --llm).")
    parser.add_argument("--llm", action="store_true", help="Use OpenAI to generate the list instead of the baked-in set.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="OpenAI model for --llm.")
    args = parser.parse_args()

    words = _generate_words_llm(count=args.count, model=args.llm_model) if args.llm else default_words()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "words": [
            {
                "id": w.id,
                "fa": w.fa,
                "ar": w.ar,
                "fa_diac": w.fa_diac,
                "latn": w.latn,
                "gloss_en": w.gloss_en,
                "rarity": w.rarity,
            }
            for w in words
        ],
    }

    out_path = public_dir() / args.out
    write_json_atomic(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
