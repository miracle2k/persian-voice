from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..schema import ModelVariant, TEXT_KIND
from .base import Provider


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class EdgeTTSProvider(Provider):
    """
    Microsoft Edge TTS provider.

    Uses Microsoft Edge's online text-to-speech service via the edge-tts library.
    No API key required - uses the same TTS backend as Microsoft Edge browser.

    Persian Voices:
      - fa-IR-FaridNeural (Male)
      - fa-IR-DilaraNeural (Female)

    Audio Format:
      - MP3 (24kHz, 48kbps mono)

    Environment Variables:
      - EDGE_TTS_ENABLED: Set to 0 to disable (default: enabled)
      - EDGE_TTS_VOICE_IDS: Comma-separated voice names
            (default: fa-IR-FaridNeural,fa-IR-DilaraNeural)

    Note: This is a cloud service - requires internet connection.
    """

    _DEFAULT_VOICES = ["fa-IR-FaridNeural", "fa-IR-DilaraNeural"]

    @property
    def provider_id(self) -> str:
        return "edge_tts"

    @property
    def provider_label(self) -> str:
        return "Microsoft Edge TTS"

    def _enabled(self) -> bool:
        return _env_bool("EDGE_TTS_ENABLED", True)

    def _get_voice_ids(self) -> list[str]:
        raw = os.environ.get("EDGE_TTS_VOICE_IDS", "").strip()
        if raw:
            return [v.strip() for v in raw.split(",") if v.strip()]
        return self._DEFAULT_VOICES

    def is_available(self) -> tuple[bool, str | None]:
        if not self._enabled():
            return False, "EDGE_TTS_ENABLED not set (set to 1 to enable)"

        if importlib.util.find_spec("edge_tts") is None:
            return False, "`edge-tts` not installed (pip install edge-tts)"

        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        available, reason = self.is_available()
        voice_ids = self._get_voice_ids()

        for voice_id in voice_ids:
            # Extract short name for display (e.g., "FaridNeural" from "fa-IR-FaridNeural")
            short_name = voice_id.split("-")[-1] if "-" in voice_id else voice_id
            group = f"{self.provider_label} · {short_name}"

            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{voice_id}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=voice_id,
                    voice_id=voice_id,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="mp3",
                    available=available,
                    unavailable_reason=reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason or "Edge TTS provider unavailable")

        import edge_tts

        voice_id = model.voice_id or self._DEFAULT_VOICES[0]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            communicate = edge_tts.Communicate(text, voice_id)
            communicate.save_sync(str(tmp_path))

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("Edge TTS produced no audio output.")
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
            "voice_id": voice_id,
            "input_kind": model.input_kind,
            "audio_format": "mp3",
            "sample_rate": 24000,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
