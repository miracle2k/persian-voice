from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape as xml_escape

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


class AzureSpeechTTSProvider(Provider):
    """
    Azure AI Speech (Neural TTS) provider.

    Uses the REST endpoint:
      POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_AZURE_TIMEOUT", 20.0)
        self._max_retries = _env_int("PERSIAN_VOICE_AZURE_MAX_RETRIES", 2)

    @property
    def provider_id(self) -> str:
        return "azure_speech"

    @property
    def provider_label(self) -> str:
        return "Azure Speech"

    def _endpoint(self) -> str | None:
        endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")
        if endpoint and endpoint.strip():
            return endpoint.strip()
        region = os.environ.get("AZURE_SPEECH_REGION")
        if not region or not region.strip():
            return None
        return f"https://{region.strip()}.tts.speech.microsoft.com/cognitiveservices/v1"

    def is_available(self) -> tuple[bool, str | None]:
        if not os.environ.get("AZURE_SPEECH_KEY"):
            return False, "AZURE_SPEECH_KEY not set"
        if not self._endpoint():
            return False, "AZURE_SPEECH_REGION or AZURE_SPEECH_ENDPOINT not set"
        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        voices_raw = os.environ.get("AZURE_SPEECH_VOICES", "fa-IR-DilaraNeural,fa-IR-FaridNeural")
        voices = [v.strip() for v in voices_raw.split(",") if v.strip()]
        lang = os.environ.get("AZURE_SPEECH_LANG", "fa-IR").strip() or "fa-IR"

        input_kinds: list[TEXT_KIND] = ["fa", "fa_latn", "latn"]
        available, reason = self.is_available()
        for voice_id in voices:
            engine_id = f"neural:{lang}"
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
        endpoint = self._endpoint()
        if not endpoint:
            raise RuntimeError("AZURE_SPEECH_REGION or AZURE_SPEECH_ENDPOINT not set")
        key = os.environ.get("AZURE_SPEECH_KEY")
        if not key:
            raise RuntimeError("AZURE_SPEECH_KEY not set")

        region = (os.environ.get("AZURE_SPEECH_REGION") or "").strip()
        lang = (os.environ.get("AZURE_SPEECH_LANG") or "fa-IR").strip() or "fa-IR"
        output_format = (os.environ.get("AZURE_SPEECH_OUTPUT_FORMAT") or "audio-24khz-48kbitrate-mono-mp3").strip()

        ssml = (
            f'<speak version="1.0" xml:lang="{xml_escape(lang)}">'
            f'<voice name="{xml_escape(model.voice_id or "")}">{xml_escape(text)}</voice>'
            f"</speak>"
        )

        headers = {
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
            "User-Agent": "persian-voice/azure-speech-tts",
            "Ocp-Apim-Subscription-Key": key,
        }
        if region:
            headers["Ocp-Apim-Subscription-Region"] = region

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        last_exc: Exception | None = None
        try:
            for attempt in range(self._max_retries + 1):
                try:
                    req = urllib.request.Request(endpoint, data=ssml.encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                        audio_bytes = resp.read()

                    if not audio_bytes:
                        raise RuntimeError("Azure Speech TTS returned empty audio.")

                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)

                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "azure_output_format": output_format,
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
                        body = exc.read(400).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    raise RuntimeError(f"Azure Speech TTS HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        raise RuntimeError("Azure Speech TTS failed.") from last_exc

