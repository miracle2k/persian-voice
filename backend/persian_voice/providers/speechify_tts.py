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


class SpeechifyTTSProvider(Provider):
    """
    Speechify AI API provider.

    Docs:
      - https://docs.sws.speechify.com/
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_SPEECHIFY_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_SPEECHIFY_MAX_RETRIES", 2)

    @property
    def provider_id(self) -> str:
        return "speechify"

    @property
    def provider_label(self) -> str:
        return "Speechify"

    def _api_key(self) -> str | None:
        return os.environ.get("SPEECHIFY_API_KEY")

    def _base_url(self) -> str:
        return (os.environ.get("SPEECHIFY_BASE_URL") or "https://api.sws.speechify.com").rstrip("/")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "SPEECHIFY_API_KEY not set"
        return True, None

    def _voice_ids(self) -> list[str]:
        raw = (os.environ.get("SPEECHIFY_VOICE_IDS") or "cliff").strip()
        voices = [v.strip() for v in raw.split(",") if v.strip()]
        return voices or ["cliff"]

    def list_model_variants(self) -> Iterable[ModelVariant]:
        engine_id = "v1/audio/stream"
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]

        available, reason = self.is_available()
        for voice_id in self._voice_ids():
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
                    audio_format="mp3",
                    available=available,
                    unavailable_reason=reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("SPEECHIFY_API_KEY not set")
        if not model.voice_id:
            raise RuntimeError("Speechify requires a voice_id")

        url = f"{self._base_url()}/v1/audio/stream"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "audio/mpeg",
            "User-Agent": "persian-voice/speechify-tts",
        }
        payload = {"input": text, "voice_id": model.voice_id}

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
                        raise RuntimeError("Speechify returned empty audio.")
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
                    raise RuntimeError(f"Speechify HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("Speechify TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
