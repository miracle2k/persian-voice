from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

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


class ResembleTTSProvider(Provider):
    """
    Resemble AI TTS provider.

    Docs:
      - https://docs.resemble.ai/reference/synthesize
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_RESEMBLE_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_RESEMBLE_MAX_RETRIES", 2)

    @property
    def provider_id(self) -> str:
        return "resemble"

    @property
    def provider_label(self) -> str:
        return "Resemble AI"

    def _api_key(self) -> str | None:
        return os.environ.get("RESEMBLE_API_KEY")

    def _voice_uuids(self) -> list[str]:
        raw = (os.environ.get("RESEMBLE_VOICE_UUIDS") or "").strip()
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "RESEMBLE_API_KEY not set"
        if not self._voice_uuids():
            return False, "RESEMBLE_VOICE_UUIDS not set"
        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]

        output_format_raw = (os.environ.get("RESEMBLE_OUTPUT_FORMAT") or "wav").strip().lower() or "wav"
        if output_format_raw not in {"wav", "mp3"}:
            output_format_raw = "wav"
        audio_format: Literal["wav", "mp3"] = "mp3" if output_format_raw == "mp3" else "wav"

        engine_id = f"synthesize:{output_format_raw}"
        available, reason = self.is_available()
        for voice_uuid in self._voice_uuids() or [""]:
            group = f"{self.provider_label} · {voice_uuid or 'voice'}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{voice_uuid}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=voice_uuid or None,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format=audio_format,
                    available=available,
                    unavailable_reason=reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("RESEMBLE_API_KEY not set")
        if not model.voice_id:
            raise RuntimeError("Resemble requires a voice_uuid (set RESEMBLE_VOICE_UUIDS)")

        output_format = (os.environ.get("RESEMBLE_OUTPUT_FORMAT") or "wav").strip().lower() or "wav"
        if output_format not in {"wav", "mp3"}:
            output_format = "wav"
        sample_rate = _env_int("RESEMBLE_SAMPLE_RATE", 48000)

        url = (os.environ.get("RESEMBLE_ENDPOINT") or "https://f.cluster.resemble.ai/synthesize").strip()
        headers = {
            "Authorization": key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "persian-voice/resemble-tts",
        }

        payload = {"voice_uuid": model.voice_id, "data": text, "output_format": output_format, "sample_rate": sample_rate}

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

                    audio_b64 = data.get("audio_content") if isinstance(data, dict) else None
                    if not isinstance(audio_b64, str) or not audio_b64.strip():
                        raise RuntimeError("Resemble returned no audio_content.")
                    audio_bytes = base64.b64decode(audio_b64)
                    if not audio_bytes:
                        raise RuntimeError("Resemble returned empty audio.")

                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "output_format": output_format,
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
                    raise RuntimeError(f"Resemble HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("Resemble TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
