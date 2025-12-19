from __future__ import annotations

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


class NarakeetTTSProvider(Provider):
    """
    Narakeet Text-to-Speech provider.

    Docs:
      - https://www.narakeet.com/docs/automating/text-to-speech-api/
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_NARAKEET_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_NARAKEET_MAX_RETRIES", 2)

    @property
    def provider_id(self) -> str:
        return "narakeet"

    @property
    def provider_label(self) -> str:
        return "Narakeet"

    def _api_key(self) -> str | None:
        return os.environ.get("NARAKEET_API_KEY")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "NARAKEET_API_KEY not set"
        return True, None

    def _voices(self) -> list[str | None]:
        raw = (os.environ.get("NARAKEET_VOICES") or os.environ.get("NARAKEET_VOICE") or "").strip()
        if not raw:
            return [None]
        voices = [v.strip() for v in raw.split(",") if v.strip()]
        return voices or [None]

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        engine_id = "text-to-speech"

        available, reason = self.is_available()
        for voice_id in self._voices():
            voice_label = voice_id or "default"
            group = f"{self.provider_label} · {voice_label}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{voice_label}/{input_kind}"
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
            raise RuntimeError("NARAKEET_API_KEY not set")

        base_url = (os.environ.get("NARAKEET_BASE_URL") or "https://api.narakeet.com").rstrip("/")
        endpoint = f"{base_url}/text-to-speech/mp3"

        params: dict[str, str] = {}
        if model.voice_id:
            params["voice"] = model.voice_id

        voice_speed = (os.environ.get("NARAKEET_VOICE_SPEED") or "").strip()
        if voice_speed:
            params["voice-speed"] = voice_speed
        voice_volume = (os.environ.get("NARAKEET_VOICE_VOLUME") or "").strip()
        if voice_volume:
            params["voice-volume"] = voice_volume

        if params:
            endpoint += "?" + urllib.parse.urlencode(params)

        headers = {
            "x-api-key": key,
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/octet-stream",
            "User-Agent": "persian-voice/narakeet-tts",
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    req = urllib.request.Request(endpoint, data=text.encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                        audio_bytes = resp.read()
                    if not audio_bytes:
                        raise RuntimeError("Narakeet returned empty audio.")
                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
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
                    raise RuntimeError(f"Narakeet HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("Narakeet TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
