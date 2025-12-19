from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from ..schema import ModelVariant, TEXT_KIND
from .base import Provider


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class OpenAITTSProvider(Provider):
    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_OPENAI_TIMEOUT", 20.0)
        self._max_retries = _env_int("PERSIAN_VOICE_OPENAI_MAX_RETRIES", 2)
        self._client: OpenAI | None = None

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def provider_label(self) -> str:
        return "OpenAI"

    def is_available(self) -> tuple[bool, str | None]:
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY not set"
        return True, None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(timeout=self._timeout_s, max_retries=self._max_retries)
        return self._client

    def list_model_variants(self) -> Iterable[ModelVariant]:
        engine_id = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip()
        voices_raw = os.environ.get("OPENAI_TTS_VOICES", "alloy")
        voices = [v.strip() for v in voices_raw.split(",") if v.strip()]
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]

        available, reason = self.is_available()
        for voice_id in voices:
            group = f"{self.provider_label} · {engine_id} · {voice_id}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{voice_id}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=voice_id,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="mp3",
                    available=available,
                    unavailable_reason=reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        client = self._get_client()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            response = client.audio.speech.create(
                model=model.engine_id,
                voice=model.voice_id or "alloy",
                input=text,
                response_format=model.audio_format,
                timeout=self._timeout_s,
            )

            if hasattr(response, "stream_to_file"):
                response.stream_to_file(tmp_path)
            elif hasattr(response, "content"):
                tmp_path.write_bytes(response.content)
            else:
                tmp_path.write_bytes(response)  # type: ignore[arg-type]

            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return {
            "provider_id": self.provider_id,
            "engine_id": model.engine_id,
            "voice_id": model.voice_id,
            "input_kind": model.input_kind,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
