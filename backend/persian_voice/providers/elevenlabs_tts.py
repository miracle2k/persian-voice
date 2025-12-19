from __future__ import annotations

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


class ElevenLabsTTSProvider(Provider):
    """
    ElevenLabs Text-to-Speech provider.

    Docs:
      - https://elevenlabs.io/docs/api-reference/text-to-speech
    """

    _DEFAULT_VOICES: list[tuple[str, str]] = [
        ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
        ("EXAVITQu4vr4xnSDxMaL", "Bella"),
    ]

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_ELEVENLABS_TIMEOUT", 20.0)
        self._max_retries = _env_int("PERSIAN_VOICE_ELEVENLABS_MAX_RETRIES", 2)
        self._cached_voices: list[tuple[str, str]] | None = None

    @property
    def provider_id(self) -> str:
        return "elevenlabs"

    @property
    def provider_label(self) -> str:
        return "ElevenLabs"

    def _api_key(self) -> str | None:
        return os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "ELEVENLABS_API_KEY (or XI_API_KEY) not set"
        return True, None

    def _base_url(self) -> str:
        return (os.environ.get("ELEVENLABS_BASE_URL") or "https://api.elevenlabs.io").rstrip("/")

    def _env_voice_ids(self) -> list[str]:
        raw = (os.environ.get("ELEVENLABS_VOICE_IDS") or "").strip()
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]

    def _fetch_voices(self) -> list[tuple[str, str]]:
        key = self._api_key()
        if not key:
            return []

        max_voices = _env_int("ELEVENLABS_MAX_VOICES", 2)
        url = f"{self._base_url()}/v1/voices"
        req = urllib.request.Request(url, headers={"xi-api-key": key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        voices = data.get("voices") if isinstance(data, dict) else None
        if not isinstance(voices, list):
            return []

        out: list[tuple[str, str]] = []
        for v in voices:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("voice_id") or "").strip()
            if not vid:
                continue
            name = str(v.get("name") or vid).strip() or vid
            out.append((vid, name))

        out.sort(key=lambda t: t[1].lower())
        if max_voices > 0:
            out = out[:max_voices]
        return out

    def _voices(self) -> list[tuple[str, str]]:
        if self._cached_voices is not None:
            return self._cached_voices

        env_ids = self._env_voice_ids()
        if env_ids:
            self._cached_voices = [(vid, vid) for vid in env_ids]
            return self._cached_voices

        available, _ = self.is_available()
        if available:
            try:
                fetched = self._fetch_voices()
                if fetched:
                    self._cached_voices = fetched
                    return self._cached_voices
            except Exception:
                pass

        self._cached_voices = list(self._DEFAULT_VOICES)
        return self._cached_voices

    def list_model_variants(self) -> Iterable[ModelVariant]:
        engine_id = (os.environ.get("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]

        available, reason = self.is_available()
        for voice_id, voice_name in self._voices():
            group = f"{self.provider_label} · {engine_id} · {voice_name}"
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
        key = self._api_key()
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY (or XI_API_KEY) not set")

        voice_id = model.voice_id
        if not voice_id:
            raise RuntimeError("ElevenLabs requires a voice_id")

        output_format = (os.environ.get("ELEVENLABS_OUTPUT_FORMAT") or "mp3_44100_128").strip() or "mp3_44100_128"
        optimize_latency = _env_int("ELEVENLABS_OPTIMIZE_STREAMING_LATENCY", 0)

        query = {"output_format": output_format}
        if optimize_latency:
            query["optimize_streaming_latency"] = str(optimize_latency)
        url = f"{self._base_url()}/v1/text-to-speech/{urllib.parse.quote(voice_id)}?{urllib.parse.urlencode(query)}"

        voice_settings_raw = (os.environ.get("ELEVENLABS_VOICE_SETTINGS_JSON") or "").strip()
        voice_settings = None
        if voice_settings_raw:
            try:
                voice_settings = json.loads(voice_settings_raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError("ELEVENLABS_VOICE_SETTINGS_JSON must be valid JSON") from exc

        payload: dict[str, object] = {"text": text, "model_id": model.engine_id}
        if isinstance(voice_settings, dict):
            payload["voice_settings"] = voice_settings

        headers = {
            "xi-api-key": key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "audio/mpeg",
            "User-Agent": "persian-voice/elevenlabs-tts",
        }

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
                        audio_bytes = resp.read()
                    if not audio_bytes:
                        raise RuntimeError("ElevenLabs returned empty audio.")
                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "elevenlabs_output_format": output_format,
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
                        body = exc.read(600).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    raise RuntimeError(f"ElevenLabs HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("ElevenLabs TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
