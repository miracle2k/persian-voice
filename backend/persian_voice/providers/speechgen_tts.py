from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
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


class SpeechGenTTSProvider(Provider):
    """
    SpeechGen Text-to-Speech provider.

    Docs:
      - https://speechgen.io/en/node/api/
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_SPEECHGEN_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_SPEECHGEN_MAX_RETRIES", 2)
        self._poll_interval_s = _env_float("PERSIAN_VOICE_SPEECHGEN_POLL_INTERVAL", 1.0)
        self._total_timeout_s = _env_float("PERSIAN_VOICE_SPEECHGEN_TOTAL_TIMEOUT", 60.0)
        self._cached_voices: list[str] | None = None

    @property
    def provider_id(self) -> str:
        return "speechgen"

    @property
    def provider_label(self) -> str:
        return "SpeechGen"

    def _token(self) -> str | None:
        return os.environ.get("SPEECHGEN_TOKEN")

    def _email(self) -> str | None:
        return os.environ.get("SPEECHGEN_EMAIL")

    def _base_url(self) -> str:
        return (os.environ.get("SPEECHGEN_BASE_URL") or "https://speechgen.io").rstrip("/")

    def _api_url(self, route: str) -> str:
        # route examples: "api/text", "api/result", "api/voices"
        return f"{self._base_url()}/index.php?r={route}"

    def is_available(self) -> tuple[bool, str | None]:
        if not self._token():
            return False, "SPEECHGEN_TOKEN not set"
        if not self._email():
            return False, "SPEECHGEN_EMAIL not set"
        return True, None

    def _voices_from_env(self) -> list[str]:
        raw = (os.environ.get("SPEECHGEN_VOICES") or "").strip()
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]

    def _fetch_persian_voices(self) -> list[str]:
        url = self._api_url("api/voices")
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "persian-voice/speechgen"})
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, dict):
            return []
        persian = data.get("Persian")
        if not isinstance(persian, list):
            return []
        voices: list[str] = []
        for item in persian:
            if not isinstance(item, dict):
                continue
            v = str(item.get("voice") or "").strip()
            if v:
                voices.append(v)
        # Keep order stable, but remove duplicates.
        seen: set[str] = set()
        out: list[str] = []
        for v in voices:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        max_voices = _env_int("SPEECHGEN_MAX_VOICES", 2)
        if max_voices > 0:
            out = out[:max_voices]
        return out

    def _voices(self) -> list[str]:
        if self._cached_voices is not None:
            return self._cached_voices

        env = self._voices_from_env()
        if env:
            self._cached_voices = env
            return self._cached_voices

        try:
            voices = self._fetch_persian_voices()
        except Exception:
            voices = []

        self._cached_voices = voices or ["Dilara", "Farid"]
        return self._cached_voices

    def _output_format(self) -> Literal["mp3", "wav"]:
        raw = (os.environ.get("SPEECHGEN_FORMAT") or "mp3").strip().lower()
        return "wav" if raw == "wav" else "mp3"

    def list_model_variants(self) -> Iterable[ModelVariant]:
        engine_id = "api/text"
        input_kinds: list[TEXT_KIND] = ["fa", "fa_latn", "latn"]
        audio_format = self._output_format()

        available, reason = self.is_available()
        for voice in self._voices():
            group = f"{self.provider_label} · {voice}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{voice}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=voice,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format=audio_format,
                    available=available,
                    unavailable_reason=reason,
                )

    def _post_form(self, *, url: str, fields: dict[str, object]) -> dict:
        encoded = urllib.parse.urlencode({k: str(v) for k, v in fields.items()}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": "persian-voice/speechgen",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _normalize_file_url(self, file_value: str) -> str:
        file_value = file_value.strip()
        if file_value.startswith("http://") or file_value.startswith("https://"):
            return file_value
        return urllib.parse.urljoin(self._base_url() + "/", file_value.lstrip("/"))

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        token = self._token()
        email = self._email()
        if not token:
            raise RuntimeError("SPEECHGEN_TOKEN not set")
        if not email:
            raise RuntimeError("SPEECHGEN_EMAIL not set")
        if not model.voice_id:
            raise RuntimeError("SpeechGen requires a voice name")

        fmt = self._output_format()
        speed = _env_float("SPEECHGEN_SPEED", 1.0)
        pitch = _env_float("SPEECHGEN_PITCH", 0.0)
        emotion = (os.environ.get("SPEECHGEN_EMOTION") or "").strip() or None
        pause_sentence = _env_int("SPEECHGEN_PAUSE_SENTENCE_MS", 300)
        pause_paragraph = _env_int("SPEECHGEN_PAUSE_PARAGRAPH_MS", 400)
        bitrate = _env_int("SPEECHGEN_BITRATE", 48000)

        url = self._api_url("api/text")
        fields: dict[str, object] = {
            "token": token,
            "email": email,
            "voice": model.voice_id,
            "text": text,
            "format": fmt,
            "speed": speed,
            "pitch": pitch,
            "pause_sentence": pause_sentence,
            "pause_paragraph": pause_paragraph,
            "bitrate": bitrate,
        }
        if emotion:
            fields["emotion"] = emotion

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    resp = self._post_form(url=url, fields=fields)
                    if not isinstance(resp, dict):
                        raise RuntimeError("SpeechGen returned invalid JSON")

                    status_raw = resp.get("status")
                    try:
                        status = int(status_raw)
                    except Exception:
                        status = -1

                    if status == 1:
                        file_val = resp.get("file")
                        if not isinstance(file_val, str) or not file_val.strip():
                            raise RuntimeError("SpeechGen returned status=1 but no file")
                        audio_url = self._normalize_file_url(file_val)
                    elif status == 0 and resp.get("id"):
                        # Poll result endpoint until completed.
                        task_id = str(resp.get("id"))
                        start = time.time()
                        audio_url = None
                        while time.time() - start < self._total_timeout_s:
                            result = self._post_form(url=self._api_url("api/result"), fields={"token": token, "email": email, "id": task_id})
                            if isinstance(result, dict):
                                try:
                                    st = int(result.get("status"))
                                except Exception:
                                    st = -1
                                if st == 1 and isinstance(result.get("file"), str) and result["file"].strip():
                                    audio_url = self._normalize_file_url(result["file"])
                                    break
                                if st == -1:
                                    raise RuntimeError(str(result.get("error") or "SpeechGen status=-1"))
                            time.sleep(max(0.1, self._poll_interval_s))
                        if not audio_url:
                            raise TimeoutError(f"SpeechGen did not complete within {self._total_timeout_s}s")
                    else:
                        raise RuntimeError(str(resp.get("error") or f"SpeechGen status={status_raw}"))

                    audio_req = urllib.request.Request(
                        audio_url,
                        headers={"User-Agent": "persian-voice/speechgen"},
                        method="GET",
                    )
                    with urllib.request.urlopen(audio_req, timeout=self._timeout_s) as resp2:
                        audio_bytes = resp2.read()
                    if not audio_bytes:
                        raise RuntimeError("SpeechGen returned empty audio.")

                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "format": fmt,
                        "speed": speed,
                        "pitch": pitch,
                        "emotion": emotion,
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
                    raise RuntimeError(f"SpeechGen HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("SpeechGen TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

