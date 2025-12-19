from __future__ import annotations

import json
import os
import time
import urllib.error
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


class WellSaidTTSProvider(Provider):
    """
    WellSaid Labs TTS provider.

    Docs:
      - https://docs.wellsaidlabs.com/reference/ttsstream
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_WELLSAID_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_WELLSAID_MAX_RETRIES", 2)

    @property
    def provider_id(self) -> str:
        return "wellsaid"

    @property
    def provider_label(self) -> str:
        return "WellSaid"

    def _api_key(self) -> str | None:
        return os.environ.get("WELLSAID_API_KEY") or os.environ.get("WELLSAID_X_API_KEY")

    def _speaker_ids(self) -> list[int]:
        raw = (os.environ.get("WELLSAID_SPEAKER_IDS") or "").strip()
        if not raw:
            return []
        out: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                continue
        return out

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "WELLSAID_API_KEY not set"
        if not self._speaker_ids():
            return False, "WELLSAID_SPEAKER_IDS not set"
        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_latn", "latn"]
        engine_id = (os.environ.get("WELLSAID_MODEL") or "caruso").strip() or "caruso"

        available, reason = self.is_available()
        for speaker_id in self._speaker_ids() or [0]:
            group = f"{self.provider_label} · speaker {speaker_id}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{speaker_id}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=str(speaker_id),
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
            raise RuntimeError("WELLSAID_API_KEY not set")
        if not model.voice_id:
            raise RuntimeError("WellSaid requires a speaker_id (set WELLSAID_SPEAKER_IDS)")

        try:
            speaker_id = int(model.voice_id)
        except ValueError as exc:
            raise RuntimeError("WellSaid speaker_id must be an integer") from exc

        base_url = (os.environ.get("WELLSAID_BASE_URL") or "https://api.wellsaidlabs.com/v1/tts").rstrip("/")
        url = f"{base_url}/stream"

        sample_rate = _env_int("WELLSAID_SAMPLE_RATE", 44100)
        payload = {
            "text": text,
            "speaker_id": speaker_id,
            "model": model.engine_id,
            "audio_configs": {"file_format": "mp3", "sample_rate": sample_rate},
        }

        headers = {
            "X-API-KEY": key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "audio/mpeg",
            "User-Agent": "persian-voice/wellsaid-tts",
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
                        raise RuntimeError("WellSaid returned empty audio.")
                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "sample_rate": sample_rate,
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
                    raise RuntimeError(f"WellSaid HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("WellSaid TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

