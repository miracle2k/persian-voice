from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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


class GoogleCloudTTSProvider(Provider):
    """
    Google Cloud Text-to-Speech provider (REST API).

    Docs:
      - https://cloud.google.com/text-to-speech/docs/reference/rest
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_GOOGLE_CLOUD_TIMEOUT", 20.0)
        self._max_retries = _env_int("PERSIAN_VOICE_GOOGLE_CLOUD_MAX_RETRIES", 2)
        self._cached_voice_names: list[str] | None = None

    @property
    def provider_id(self) -> str:
        return "google_cloud_tts"

    @property
    def provider_label(self) -> str:
        return "Google Cloud TTS"

    def _api_key(self) -> str | None:
        return os.environ.get("GOOGLE_CLOUD_TTS_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "GOOGLE_CLOUD_TTS_API_KEY (or GOOGLE_API_KEY) not set"
        return True, None

    def _language_code(self) -> str:
        return (os.environ.get("GOOGLE_CLOUD_TTS_LANGUAGE_CODE") or "fa-IR").strip() or "fa-IR"

    def _fetch_voice_names(self) -> list[str]:
        key = self._api_key()
        if not key:
            return []
        lang = self._language_code()
        url = (
            "https://texttospeech.googleapis.com/v1/voices?"
            + urllib.parse.urlencode({"languageCode": lang, "key": key})
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        voices = data.get("voices") if isinstance(data, dict) else None
        if not isinstance(voices, list):
            return []
        names: list[str] = []
        for v in voices:
            if not isinstance(v, dict):
                continue
            name = str(v.get("name") or "").strip()
            if name:
                names.append(name)
        names = sorted(set(names))
        max_voices = _env_int("GOOGLE_CLOUD_TTS_MAX_VOICES", 2)
        if max_voices > 0:
            names = names[:max_voices]
        return names

    def _voice_names(self) -> list[str]:
        if self._cached_voice_names is not None:
            return self._cached_voice_names

        raw = (os.environ.get("GOOGLE_CLOUD_TTS_VOICE_NAMES") or "").strip()
        if raw:
            self._cached_voice_names = [v.strip() for v in raw.split(",") if v.strip()]
            return self._cached_voice_names

        available, _ = self.is_available()
        if available:
            try:
                names = self._fetch_voice_names()
                if names:
                    self._cached_voice_names = names
                    return self._cached_voice_names
            except Exception:
                pass

        self._cached_voice_names = []
        return self._cached_voice_names

    def list_model_variants(self) -> Iterable[ModelVariant]:
        lang = self._language_code()
        engine_id = f"text-to-speech:v1:{lang}"
        input_kinds: list[TEXT_KIND] = ["fa", "fa_latn", "latn"]

        voices = self._voice_names()
        if not voices:
            voices = [""]  # Let Google pick the default for the language.

        available, reason = self.is_available()
        for voice_name in voices:
            voice_label = voice_name or "default"
            group = f"{self.provider_label} · {lang} · {voice_label}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{voice_label}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=voice_name or None,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="mp3",
                    available=available,
                    unavailable_reason=reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("GOOGLE_CLOUD_TTS_API_KEY (or GOOGLE_API_KEY) not set")

        lang = self._language_code()
        url = "https://texttospeech.googleapis.com/v1/text:synthesize?" + urllib.parse.urlencode({"key": key})

        audio_config: dict[str, object] = {"audioEncoding": "MP3"}
        sample_rate_hz_raw = (os.environ.get("GOOGLE_CLOUD_TTS_SAMPLE_RATE_HZ") or "").strip()
        if sample_rate_hz_raw:
            try:
                audio_config["sampleRateHertz"] = int(sample_rate_hz_raw)
            except ValueError:
                pass

        speaking_rate_raw = (os.environ.get("GOOGLE_CLOUD_TTS_SPEAKING_RATE") or "").strip()
        if speaking_rate_raw:
            try:
                audio_config["speakingRate"] = float(speaking_rate_raw)
            except ValueError:
                pass

        pitch_raw = (os.environ.get("GOOGLE_CLOUD_TTS_PITCH") or "").strip()
        if pitch_raw:
            try:
                audio_config["pitch"] = float(pitch_raw)
            except ValueError:
                pass

        voice: dict[str, object] = {"languageCode": lang}
        if model.voice_id:
            voice["name"] = model.voice_id

        payload = {"input": {"text": text}, "voice": voice, "audioConfig": audio_config}
        headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    audio_b64 = data.get("audioContent") if isinstance(data, dict) else None
                    if not isinstance(audio_b64, str) or not audio_b64.strip():
                        raise RuntimeError("Google Cloud TTS returned no audioContent.")
                    audio_bytes = base64.b64decode(audio_b64)
                    if not audio_bytes:
                        raise RuntimeError("Google Cloud TTS returned empty audio.")
                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "language_code": lang,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                except urllib.error.HTTPError as exc:
                    last_exc = exc
                    retryable = exc.code in {408, 429, 500, 502, 503, 504}
                    if attempt < self._max_retries and retryable:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    body = ""
                    try:
                        body = exc.read(800).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    raise RuntimeError(f"Google Cloud TTS HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("Google Cloud TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

