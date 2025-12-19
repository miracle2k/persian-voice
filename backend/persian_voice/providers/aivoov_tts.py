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


class AiVOOVTTSProvider(Provider):
    """
    AiVOOV Text-to-Speech provider.

    Base URL: https://aivoov.com/api/v8/
    Docs: https://github.com/AiVOOV/aivoov-api
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_AIVOOV_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_AIVOOV_MAX_RETRIES", 2)
        self._cached_voices: list[tuple[str, str, str | None]] | None = None

    @property
    def provider_id(self) -> str:
        return "aivoov"

    @property
    def provider_label(self) -> str:
        return "AiVOOV"

    def _api_key(self) -> str | None:
        return os.environ.get("AIVOOV_API_KEY")

    def _base_url(self) -> str:
        return (os.environ.get("AIVOOV_BASE_URL") or "https://aivoov.com/api/v8").rstrip("/")

    def _language_code(self) -> str:
        return (os.environ.get("AIVOOV_LANGUAGE_CODE") or "fa-IR").strip() or "fa-IR"

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "AIVOOV_API_KEY not set"
        return True, None

    def _voice_ids_from_env(self) -> list[str]:
        raw = (os.environ.get("AIVOOV_VOICE_IDS") or "").strip()
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]

    def _fetch_voices(self) -> list[tuple[str, str, str | None]]:
        key = self._api_key()
        if not key:
            return []

        params: dict[str, str] = {}
        lang = self._language_code()
        if lang:
            params["language_code"] = lang

        url = f"{self._base_url()}/voices"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(
            url,
            headers={"X-API-KEY": key, "Accept": "application/json", "User-Agent": "persian-voice/aivoov-tts"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not isinstance(data, list):
            return []

        voices: list[tuple[str, str, str | None]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            voice_id = str(item.get("voice_id") or "").strip()
            if not voice_id:
                continue
            name = str(item.get("name") or voice_id).strip() or voice_id
            language = item.get("language")
            lang_str = str(language).strip() if isinstance(language, str) else None
            voices.append((voice_id, name, lang_str))

        voices.sort(key=lambda t: t[1].lower())
        max_voices = _env_int("AIVOOV_MAX_VOICES", 2)
        if max_voices > 0:
            voices = voices[:max_voices]
        return voices

    def _voices(self) -> list[tuple[str, str, str | None]]:
        if self._cached_voices is not None:
            return self._cached_voices

        env_ids = self._voice_ids_from_env()
        if env_ids:
            self._cached_voices = [(vid, vid, None) for vid in env_ids]
            return self._cached_voices

        available, _ = self.is_available()
        if available:
            try:
                voices = self._fetch_voices()
                if voices:
                    self._cached_voices = voices
                    return self._cached_voices
            except Exception:
                pass

        self._cached_voices = []
        return self._cached_voices

    def list_model_variants(self) -> Iterable[ModelVariant]:
        engine_id = "api/v8/create"
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]

        available, reason = self.is_available()
        voices = self._voices()
        if not voices:
            voices = [("", "voice", None)]

        for voice_id, voice_name, lang in voices:
            group = f"{self.provider_label} · {voice_name}" + (f" · {lang}" if lang else "")
            mv_available = available and bool(voice_id)
            mv_reason = reason if not available else ("AIVOOV_VOICE_IDS not set" if not voice_id else None)
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{voice_id or 'voice'}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=voice_id or None,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="mp3",
                    available=mv_available,
                    unavailable_reason=mv_reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("AIVOOV_API_KEY not set")
        if not model.voice_id:
            raise RuntimeError("AiVOOV requires a voice_id (set AIVOOV_VOICE_IDS)")

        url = f"{self._base_url()}/create"

        # These arrays must be aligned; we send just one segment per word.
        pitch = (os.environ.get("AIVOOV_PITCH") or "default").strip() or "default"
        speaking_rate = (os.environ.get("AIVOOV_SPEAKING_RATE") or "default").strip() or "default"
        volume = (os.environ.get("AIVOOV_VOLUME") or "default").strip() or "default"

        fields = {
            "voice_id[]": [model.voice_id],
            "transcribe_text[]": [text],
            "transcribe_ssml_pitch_rate[]": [pitch],
            "transcribe_ssml_spk_rate[]": [speaking_rate],
            "transcribe_ssml_volume[]": [volume],
        }

        body = urllib.parse.urlencode(fields, doseq=True).encode("utf-8")
        headers = {
            "X-API-KEY": key,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "persian-voice/aivoov-tts",
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                    if not isinstance(data, dict):
                        raise RuntimeError("AiVOOV returned invalid JSON")
                    status = data.get("status")
                    if status is not True:
                        msg = str(data.get("message") or data.get("error") or "AiVOOV request failed").strip()
                        raise RuntimeError(msg)
                    audio_b64 = data.get("audio")
                    if not isinstance(audio_b64, str) or not audio_b64.strip():
                        raise RuntimeError("AiVOOV returned no audio.")
                    audio_bytes = base64.b64decode(audio_b64)
                    if not audio_bytes:
                        raise RuntimeError("AiVOOV returned empty audio.")

                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "pitch": pitch,
                        "speaking_rate": speaking_rate,
                        "volume": volume,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                except urllib.error.HTTPError as exc:
                    last_exc = exc
                    retryable = exc.code in {408, 429, 500, 502, 503, 504}
                    if attempt < self._max_retries and retryable:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    body_text = ""
                    try:
                        body_text = exc.read(800).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    raise RuntimeError(f"AiVOOV HTTP {exc.code}: {body_text}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("AiVOOV TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
