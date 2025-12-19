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


class IBMWatsonTTSProvider(Provider):
    """
    IBM Watson Text to Speech provider (REST API).

    Docs:
      - https://cloud.ibm.com/apidocs/text-to-speech
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_IBM_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_IBM_MAX_RETRIES", 2)

    @property
    def provider_id(self) -> str:
        return "ibm_watson"

    @property
    def provider_label(self) -> str:
        return "IBM Watson"

    def _api_key(self) -> str | None:
        return os.environ.get("IBM_WATSON_TTS_API_KEY") or os.environ.get("WATSON_API_KEY")

    def _service_url(self) -> str | None:
        raw = (os.environ.get("IBM_WATSON_TTS_URL") or "").strip()
        if not raw:
            return None
        return raw.rstrip("/")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "IBM_WATSON_TTS_API_KEY not set"
        if not self._service_url():
            return False, "IBM_WATSON_TTS_URL not set"
        return True, None

    def _voices(self) -> list[str]:
        raw = (os.environ.get("IBM_WATSON_TTS_VOICES") or "en-US_AllisonV3Voice").strip()
        return [v.strip() for v in raw.split(",") if v.strip()] or ["en-US_AllisonV3Voice"]

    def _accept_format(self) -> str:
        raw = (os.environ.get("IBM_WATSON_TTS_ACCEPT") or "audio/mp3").strip().lower()
        if raw in {"audio/mp3", "audio/mpeg", "audio/wav"}:
            return raw
        return "audio/mp3"

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        accept = self._accept_format()
        audio_format = "wav" if accept.endswith("wav") else "mp3"
        engine_id = f"watson:{accept}"

        available, reason = self.is_available()
        for voice_id in self._voices():
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
                    audio_format=audio_format,
                    available=available,
                    unavailable_reason=reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        key = self._api_key()
        url_base = self._service_url()
        if not key:
            raise RuntimeError("IBM_WATSON_TTS_API_KEY not set")
        if not url_base:
            raise RuntimeError("IBM_WATSON_TTS_URL not set")
        if not model.voice_id:
            raise RuntimeError("IBM Watson requires a voice id")

        accept = self._accept_format()
        audio_format = "wav" if accept.endswith("wav") else "mp3"

        synth_url = f"{url_base}/v1/synthesize?" + urllib.parse.urlencode({"voice": model.voice_id})

        # HTTP Basic auth with username "apikey"
        basic = base64.b64encode(f"apikey:{key}".encode("utf-8")).decode("ascii")
        headers = {
            "Authorization": f"Basic {basic}",
            "Accept": accept,
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "persian-voice/ibm-watson-tts",
        }

        payload = {"text": text}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    req = urllib.request.Request(
                        synth_url,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                        audio_bytes = resp.read()

                    if not audio_bytes:
                        raise RuntimeError("IBM Watson returned empty audio.")
                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "accept": accept,
                        "audio_format": audio_format,
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
                    raise RuntimeError(f"IBM Watson HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("IBM Watson TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
