from __future__ import annotations

import importlib.util
import os
import urllib.request
import wave
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


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


class PiperTTSProvider(Provider):
    """
    Piper TTS local provider.

    Piper is a fast, local neural text-to-speech engine that uses ONNX models.
    Models are downloaded from HuggingFace on first use and cached locally.

    Voice:
      - gyro (default): Persian male voice, ~63MB

    Environment Variables:
      - PIPER_ENABLED: Set to 1 to enable (default: 0)
      - PIPER_VOICE_IDS: Comma-separated voice names (default: gyro)
      - PIPER_CACHE_DIR: Model cache directory (default: ~/.cache/piper-voices)
    """

    _HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    _DEFAULT_VOICES = ["gyro"]
    _QUALITY = "medium"  # Persian voices only have medium quality

    def __init__(self) -> None:
        self._voice_cache: dict[str, object] = {}

    @property
    def provider_id(self) -> str:
        return "piper"

    @property
    def provider_label(self) -> str:
        return "Piper TTS"

    def _enabled(self) -> bool:
        return _env_bool("PIPER_ENABLED", False)

    def _get_voice_ids(self) -> list[str]:
        raw = os.environ.get("PIPER_VOICE_IDS", "").strip()
        if raw:
            return [v.strip() for v in raw.split(",") if v.strip()]
        return self._DEFAULT_VOICES

    def _get_cache_dir(self) -> Path:
        cache_dir = _env_str("PIPER_CACHE_DIR", "~/.cache/piper-voices")
        return Path(cache_dir).expanduser()

    def is_available(self) -> tuple[bool, str | None]:
        if not self._enabled():
            return False, "PIPER_ENABLED not set (set to 1 to enable)"

        if importlib.util.find_spec("piper") is None:
            return False, "`piper-tts` not installed (pip install piper-tts)"

        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        available, reason = self.is_available()
        voice_ids = self._get_voice_ids()

        for voice_id in voice_ids:
            engine_id = f"fa_IR-{voice_id}-{self._QUALITY}"
            group = f"{self.provider_label} · {voice_id}"
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
                    audio_format="wav",
                    available=available,
                    unavailable_reason=reason,
                )

    def _download_model(self, voice_id: str) -> Path:
        """Download model files from HuggingFace if not cached."""
        cache_dir = self._get_cache_dir()
        voice_dir = cache_dir / "fa" / "fa_IR" / voice_id / self._QUALITY
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Piper model files are named: fa_IR-{voice}-{quality}.onnx
        model_name = f"fa_IR-{voice_id}-{self._QUALITY}"
        model_path = voice_dir / f"{model_name}.onnx"
        config_path = voice_dir / f"{model_name}.onnx.json"

        # Download model if not exists
        if not model_path.exists():
            model_url = f"{self._HF_BASE_URL}/fa/fa_IR/{voice_id}/{self._QUALITY}/{model_name}.onnx"
            self._download_file(model_url, model_path)

        # Download config if not exists
        if not config_path.exists():
            config_url = f"{self._HF_BASE_URL}/fa/fa_IR/{voice_id}/{self._QUALITY}/{model_name}.onnx.json"
            self._download_file(config_url, config_path)

        return model_path

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file with progress indication."""
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(dest)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _load_voice(self, voice_id: str) -> object:
        """Load or get cached voice model."""
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        from piper.voice import PiperVoice  # type: ignore[import-not-found]

        model_path = self._download_model(voice_id)
        voice = PiperVoice.load(str(model_path))
        self._voice_cache[voice_id] = voice
        return voice

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason or "Piper provider unavailable")

        voice_id = model.voice_id or self._DEFAULT_VOICES[0]
        voice = self._load_voice(voice_id)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            with wave.open(str(tmp_path), "wb") as wf:
                voice.synthesize_wav(text, wf)  # type: ignore[union-attr]

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("Piper produced no audio output.")
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        # Get sample rate from voice config
        sample_rate = getattr(voice.config, "sample_rate", 22050)  # type: ignore[union-attr]

        return {
            "provider_id": self.provider_id,
            "engine_id": model.engine_id,
            "voice_id": voice_id,
            "input_kind": model.input_kind,
            "piper_quality": self._QUALITY,
            "sample_rate": sample_rate,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
