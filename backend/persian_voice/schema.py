from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


SchemaVersion = Literal[1]
TEXT_KIND = Literal["fa", "fa_diac", "latn", "ar"]


@dataclass(frozen=True, slots=True)
class Word:
    id: str
    fa: str
    ar: str
    fa_diac: str
    latn: str
    gloss_en: str | None = None
    rarity: str | None = None


@dataclass(frozen=True, slots=True)
class ModelVariant:
    id: str
    provider_id: str
    provider_label: str
    engine_id: str
    voice_id: str | None
    input_kind: TEXT_KIND
    label: str
    group: str
    audio_format: Literal["mp3", "wav", "flac"] = "mp3"
    available: bool = True
    unavailable_reason: str | None = None


class ClipMeta(TypedDict, total=False):
    provider_id: str
    engine_id: str
    voice_id: str | None
    input_kind: TEXT_KIND


@dataclass(frozen=True, slots=True)
class Clip:
    word_id: str
    model_id: str
    text: str
    audio_path: str
    created_at: str
    meta: dict[str, Any]
