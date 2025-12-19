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


class LovoTTSProvider(Provider):
    """
    LOVO (Genny) TTS provider.

    API base: https://api.genny.lovo.ai
    Docs: https://docs.genny.lovo.ai/
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_LOVO_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_LOVO_MAX_RETRIES", 2)
        self._cached_speakers: list[tuple[str, str, str | None]] | None = None

    @property
    def provider_id(self) -> str:
        return "lovo"

    @property
    def provider_label(self) -> str:
        return "LOVO"

    def _api_key(self) -> str | None:
        return os.environ.get("LOVO_API_KEY")

    def _base_url(self) -> str:
        return (os.environ.get("LOVO_BASE_URL") or "https://api.genny.lovo.ai").rstrip("/")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "LOVO_API_KEY not set"
        return True, None

    def _speaker_ids_from_env(self) -> list[str]:
        raw = (os.environ.get("LOVO_SPEAKER_IDS") or "").strip()
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]

    def _fetch_speakers(self) -> list[tuple[str, str, str | None]]:
        key = self._api_key()
        if not key:
            return []
        base = self._base_url()
        max_speakers = _env_int("LOVO_MAX_SPEAKERS", 2)
        locale_filter = (os.environ.get("LOVO_LOCALE_FILTER") or "fa-IR").strip()

        params = {"limit": str(max(1, max_speakers * 10)), "page": "0"}
        url = f"{base}/api/v1/speakers?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"X-API-KEY": key, "Accept": "application/json", "User-Agent": "persian-voice/lovo-tts"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []

        speakers: list[tuple[str, str, str | None]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "").strip()
            if not sid:
                continue
            name = str(item.get("displayName") or sid).strip() or sid
            locale = item.get("locale")
            locale_str = str(locale).strip() if isinstance(locale, str) else None
            speakers.append((sid, name, locale_str))

        if locale_filter:
            filtered = [s for s in speakers if (s[2] or "").lower() == locale_filter.lower()]
            if filtered:
                speakers = filtered

        speakers.sort(key=lambda t: t[1].lower())
        if max_speakers > 0:
            speakers = speakers[:max_speakers]
        return speakers

    def _speakers(self) -> list[tuple[str, str, str | None]]:
        if self._cached_speakers is not None:
            return self._cached_speakers

        env_ids = self._speaker_ids_from_env()
        if env_ids:
            self._cached_speakers = [(sid, sid, None) for sid in env_ids]
            return self._cached_speakers

        available, _ = self.is_available()
        if available:
            try:
                speakers = self._fetch_speakers()
                if speakers:
                    self._cached_speakers = speakers
                    return self._cached_speakers
            except Exception:
                pass

        self._cached_speakers = []
        return self._cached_speakers

    def list_model_variants(self) -> Iterable[ModelVariant]:
        engine_id = "api/v1/tts/sync"
        input_kinds: list[TEXT_KIND] = ["fa", "fa_latn", "latn"]

        available, reason = self.is_available()
        speakers = self._speakers()
        if not speakers:
            speakers = [("", "speaker", None)]

        for speaker_id, speaker_name, locale in speakers:
            voice_label = speaker_name if speaker_id else "speaker"
            group = f"{self.provider_label} · {voice_label}" + (f" · {locale}" if locale else "")
            mv_available = available and bool(speaker_id)
            mv_reason = reason if not available else ("LOVO_SPEAKER_IDS not set" if not speaker_id else None)
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{speaker_id or 'speaker'}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=speaker_id or None,
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
            raise RuntimeError("LOVO_API_KEY not set")
        if not model.voice_id:
            raise RuntimeError("LOVO requires a speaker id (set LOVO_SPEAKER_IDS)")

        base = self._base_url()
        url = f"{base}/api/v1/tts/sync"

        speed = _env_float("LOVO_SPEED", 1.0)
        speaker_style = (os.environ.get("LOVO_SPEAKER_STYLE_ID") or "").strip() or None
        payload: dict[str, object] = {"text": text, "speaker": model.voice_id, "speed": speed}
        if speaker_style:
            payload["speakerStyle"] = speaker_style

        headers = {
            "X-API-KEY": key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "persian-voice/lovo-tts",
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
                        data = json.loads(resp.read().decode("utf-8"))

                    # data.data[0].urls[0] is the audio URL
                    items = data.get("data") if isinstance(data, dict) else None
                    if not isinstance(items, list) or not items:
                        raise RuntimeError("LOVO sync TTS returned no data[]")
                    first = items[0]
                    if not isinstance(first, dict):
                        raise RuntimeError("LOVO sync TTS returned invalid data[0]")
                    urls = first.get("urls")
                    if not isinstance(urls, list) or not urls or not isinstance(urls[0], str) or not urls[0].strip():
                        raise RuntimeError("LOVO sync TTS returned no urls")
                    audio_url = urls[0].strip()

                    # Download audio
                    audio_req = urllib.request.Request(audio_url, headers={"User-Agent": "persian-voice/lovo-tts"})
                    with urllib.request.urlopen(audio_req, timeout=self._timeout_s) as resp:
                        audio_bytes = resp.read()
                    if not audio_bytes:
                        raise RuntimeError("LOVO returned empty audio.")

                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "speed": speed,
                        "speaker_style": speaker_style,
                        "audio_url": audio_url,
                        "job_id": data.get("id") if isinstance(data, dict) else None,
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
                    raise RuntimeError(f"LOVO HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("LOVO TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

