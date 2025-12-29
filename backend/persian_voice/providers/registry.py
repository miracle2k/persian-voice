from __future__ import annotations

import os
from typing import Iterable

from .base import Provider
from .azure_speech import AzureSpeechTTSProvider
from .aws_polly_tts import AWSPollyTTSProvider
from .cambai_tts import CambAITTSProvider
from .chatterbox_tts import ChatterboxTTSProvider
from .edge_tts import EdgeTTSProvider
from .elevenlabs_tts import ElevenLabsTTSProvider
from .google_cloud_tts import GoogleCloudTTSProvider
from .lovo_tts import LovoTTSProvider
from .manatts_tts import ManaTTSProvider
from .mms_tts import MMSTTSProvider
from .openai_tts import OpenAITTSProvider
from .outetts_tts import OuteTTSTTSProvider
from .piper_tts import PiperTTSProvider
from .resemble_tts import ResembleTTSProvider
from .wellsaid_tts import WellSaidTTSProvider


def _known_provider_factories() -> dict[str, type[Provider]]:
    return {
        "openai": OpenAITTSProvider,
        "azure_speech": AzureSpeechTTSProvider,
        "cambai": CambAITTSProvider,
        "chatterbox": ChatterboxTTSProvider,
        "edge_tts": EdgeTTSProvider,
        "elevenlabs": ElevenLabsTTSProvider,
        "lovo": LovoTTSProvider,
        "google_cloud_tts": GoogleCloudTTSProvider,
        "manatts": ManaTTSProvider,
        "mms": MMSTTSProvider,
        "outetts": OuteTTSTTSProvider,
        "piper": PiperTTSProvider,
        "aws_polly": AWSPollyTTSProvider,
        "resemble": ResembleTTSProvider,
        "wellsaid": WellSaidTTSProvider,
    }


def load_providers(provider_ids: str | None = None) -> list[Provider]:
    factories = _known_provider_factories()

    raw = provider_ids or os.environ.get("PERSIAN_VOICE_PROVIDERS") or "all"
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    if not ids:
        ids = ["all"]

    providers: list[Provider] = []
    for pid in ids:
        if pid == "all":
            providers.extend(factory() for factory in factories.values())
            continue
        factory = factories.get(pid)
        if not factory:
            raise ValueError(
                f"Unknown provider id: {pid}. Known: {', '.join(sorted(factories.keys()))}"
            )
        providers.append(factory())

    return providers
