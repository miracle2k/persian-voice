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


class CambAITTSProvider(Provider):
    """
    CAMB.AI TTS provider (REST API).

    API base: https://client.camb.ai/apis
    (See https://github.com/Camb-ai/cambai-python-sdk for the OpenAPI-generated client.)
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_CAMB_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_CAMB_MAX_RETRIES", 2)
        self._poll_interval_s = _env_float("PERSIAN_VOICE_CAMB_POLL_INTERVAL", 1.0)
        self._total_timeout_s = _env_float("PERSIAN_VOICE_CAMB_TOTAL_TIMEOUT", 120.0)
        self._voice_cache: dict[int, dict[str, int | str | None]] | None = None

    @property
    def provider_id(self) -> str:
        return "cambai"

    @property
    def provider_label(self) -> str:
        return "CAMB.AI"

    def _api_key(self) -> str | None:
        return os.environ.get("CAMB_API_KEY")

    def _base_url(self) -> str:
        return (os.environ.get("CAMB_BASE_URL") or "https://client.camb.ai/apis").rstrip("/")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key():
            return False, "CAMB_API_KEY not set"
        return True, None

    def _voice_ids_from_env(self) -> list[int]:
        raw = (os.environ.get("CAMB_VOICE_IDS") or "").strip()
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

    def _fetch_voices(self) -> dict[int, dict[str, int | str | None]]:
        key = self._api_key()
        if not key:
            return {}
        url = f"{self._base_url()}/list-voices"
        req = urllib.request.Request(url, headers={"x-api-key": key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, list):
            return {}

        out: dict[int, dict[str, int | str | None]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                vid = int(item.get("id"))
            except Exception:
                continue
            voice_name = item.get("voice_name")
            language = item.get("language")
            try:
                language_id = int(language) if language is not None else None
            except Exception:
                language_id = None
            out[vid] = {"voice_name": str(voice_name) if voice_name is not None else None, "language_id": language_id}
        return out

    def _voices(self) -> dict[int, dict[str, int | str | None]]:
        if self._voice_cache is not None:
            return self._voice_cache
        available, _ = self.is_available()
        if available:
            try:
                self._voice_cache = self._fetch_voices()
                return self._voice_cache
            except Exception:
                self._voice_cache = {}
                return self._voice_cache
        self._voice_cache = {}
        return self._voice_cache

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        engine_id = "tts"
        available, reason = self.is_available()

        voice_ids = self._voice_ids_from_env()
        voice_meta = self._voices()

        if not voice_ids and voice_meta:
            max_voices = _env_int("CAMB_MAX_VOICES", 2)
            voice_ids = sorted(voice_meta.keys())[:max_voices] if max_voices > 0 else sorted(voice_meta.keys())

        if not voice_ids:
            voice_ids = [0]

        for vid in voice_ids:
            name = voice_meta.get(vid, {}).get("voice_name") if voice_meta else None
            voice_label = f"{vid}" + (f" · {name}" if isinstance(name, str) and name.strip() else "")
            group = f"{self.provider_label} · {voice_label}"
            mv_available = available and vid != 0
            mv_reason = reason if not available else ("CAMB_VOICE_IDS not set" if vid == 0 else None)
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{vid}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=str(vid),
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="flac",
                    available=mv_available,
                    unavailable_reason=mv_reason,
                )

    def _request_json(self, *, method: str, url: str, payload: dict | None = None) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("CAMB_API_KEY not set")
        headers = {"x-api-key": key, "Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _request_bytes(self, *, method: str, url: str, accept: str) -> bytes:
        key = self._api_key()
        if not key:
            raise RuntimeError("CAMB_API_KEY not set")
        headers = {"x-api-key": key, "Accept": accept}
        req = urllib.request.Request(url, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return resp.read()

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("CAMB_API_KEY not set")
        if not model.voice_id:
            raise RuntimeError("CAMB_VOICE_IDS not set")

        try:
            voice_id = int(model.voice_id)
        except ValueError as exc:
            raise RuntimeError("CAMB voice_id must be an integer") from exc

        # Determine language ID based on input_kind or environment override
        # Persian (fa-ir) = 70, English (en-us) = 1
        language_id_raw = (os.environ.get("CAMB_LANGUAGE_ID") or "").strip()
        language_id: int | None = None
        if language_id_raw:
            try:
                language_id = int(language_id_raw)
            except ValueError:
                language_id = None
        if language_id is None:
            # Use Persian language ID for Persian text input kinds
            if model.input_kind in ("fa", "fa_diac"):
                language_id = 70  # Persian (Iran)
            elif model.input_kind == "latn":
                # Latin transliteration - use Persian since it's Persian words
                language_id = 70
            else:
                language_id = 1  # Default to English

        base = self._base_url()

        # 1) Create task
        create_url = f"{base}/tts"
        create_payload = {"text": text, "voice_id": voice_id, "language": language_id}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            last_exc: Exception | None = None
            for attempt in range(self._max_retries + 1):
                try:
                    created = self._request_json(method="POST", url=create_url, payload=create_payload)
                    task_id = created.get("task_id") if isinstance(created, dict) else None
                    if not isinstance(task_id, str) or not task_id.strip():
                        raise RuntimeError("CAMB create_tts returned no task_id")

                    # 2) Poll until run_id is available
                    poll_url = f"{base}/tts/{urllib.parse.quote(task_id)}"
                    start = time.time()
                    run_id: int | None = None
                    status: str | None = None
                    while time.time() - start < self._total_timeout_s:
                        polled = self._request_json(method="GET", url=poll_url)
                        if isinstance(polled, dict):
                            status = str(polled.get("status") or "")
                            rid = polled.get("run_id")
                            if rid is not None:
                                try:
                                    run_id = int(rid)
                                except Exception:
                                    run_id = None
                        if run_id is not None:
                            break
                        if status in {"ERROR", "TIMEOUT", "PAYMENT_REQUIRED"}:
                            raise RuntimeError(f"CAMB TTS task failed with status {status}")
                        time.sleep(max(0.1, self._poll_interval_s))
                    if run_id is None:
                        raise TimeoutError(f"CAMB TTS timed out after {self._total_timeout_s}s (status={status})")

                    # 3) Fetch audio bytes
                    result_url = (
                        f"{base}/tts-result/{run_id}?"
                        + urllib.parse.urlencode({"output_type": "raw_bytes"})
                    )
                    audio_bytes = self._request_bytes(method="GET", url=result_url, accept="audio/flac")
                    if not audio_bytes:
                        raise RuntimeError("CAMB returned empty audio.")
                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                        "language_id": language_id,
                        "run_id": run_id,
                        "task_id": task_id,
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
                    raise RuntimeError(f"CAMB HTTP {exc.code}: {body}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        continue
                    raise
            raise RuntimeError("CAMB TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
