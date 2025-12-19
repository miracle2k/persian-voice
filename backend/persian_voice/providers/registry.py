from __future__ import annotations

import os
from typing import Iterable

from .base import Provider
from .openai_tts import OpenAITTSProvider


def _known_provider_factories() -> dict[str, type[Provider]]:
    return {
        "openai": OpenAITTSProvider,
    }


def load_providers(provider_ids: str | None = None) -> list[Provider]:
    factories = _known_provider_factories()

    raw = provider_ids or os.environ.get("PERSIAN_VOICE_PROVIDERS") or "openai"
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    if not ids:
        ids = ["openai"]

    providers: list[Provider] = []
    for pid in ids:
        if pid == "all":
            providers.extend(factory() for factory in factories.values())
            continue
        factory = factories.get(pid)
        if not factory:
            raise ValueError(f"Unknown provider id: {pid}. Known: {', '.join(sorted(factories.keys()))}")
        providers.append(factory())

    return providers

