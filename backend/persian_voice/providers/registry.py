from __future__ import annotations

import os
from typing import Iterable

from .base import Provider
from .azure_speech import AzureSpeechTTSProvider
from .cambai_tts import CambAITTSProvider
from .elevenlabs_tts import ElevenLabsTTSProvider
from .google_cloud_tts import GoogleCloudTTSProvider
from .lovo_tts import LovoTTSProvider
from .narakeet_tts import NarakeetTTSProvider
from .openai_tts import OpenAITTSProvider
from .resemble_tts import ResembleTTSProvider
from .speechify_tts import SpeechifyTTSProvider
from .speechgen_tts import SpeechGenTTSProvider
from .wellsaid_tts import WellSaidTTSProvider


def _known_provider_factories() -> dict[str, type[Provider]]:
    return {
        "openai": OpenAITTSProvider,
        "azure_speech": AzureSpeechTTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "lovo": LovoTTSProvider,
        "google_cloud_tts": GoogleCloudTTSProvider,
        "resemble": ResembleTTSProvider,
        "narakeet": NarakeetTTSProvider,
        "speechify": SpeechifyTTSProvider,
        "speechgen": SpeechGenTTSProvider,
        "wellsaid": WellSaidTTSProvider,
        "cambai": CambAITTSProvider,
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
