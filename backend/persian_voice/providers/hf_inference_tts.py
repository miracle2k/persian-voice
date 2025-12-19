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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class HuggingFaceInferenceTTSProvider(Provider):
    """
    Hugging Face Inference API TTS provider.

    Useful for benchmarking open-weight models without downloading them locally.
    Example Persian model: facebook/mms-tts-fas

    Docs:
      - https://huggingface.co/docs/api-inference/index
    """

    def __init__(self) -> None:
        self._timeout_s = _env_float("PERSIAN_VOICE_HF_TIMEOUT", 30.0)
        self._max_retries = _env_int("PERSIAN_VOICE_HF_MAX_RETRIES", 2)
        self._wait_for_model = _env_bool("HF_INFERENCE_WAIT_FOR_MODEL", True)
        self._model_live_cache: dict[str, tuple[bool, str | None]] = {}

    @property
    def provider_id(self) -> str:
        return "hf_inference"

    @property
    def provider_label(self) -> str:
        return "Hugging Face Inference"

    def _token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")

    def is_available(self) -> tuple[bool, str | None]:
        if not self._token():
            return False, "HF_TOKEN (or HUGGINGFACE_API_TOKEN) not set"
        return True, None

    def _default_model_ids(self) -> list[str]:
        # Curated set of Persian-capable open(-ish) weights that we want to keep
        # permanently visible in the comparison project (no env var needed).
        #
        # Note: some models may not be enabled on Hugging Face Inference for your
        # account/plan; failures will be recorded in clips.json.
        return [
            # Meta MMS TTS Persian checkpoint
            "facebook/mms-tts-fas",
            # OuteTTS 1.0 (1B) claims Persian support
            "OuteAI/Llama-OuteTTS-1.0-1B",
        ]

    def _base_url(self) -> str:
        return (os.environ.get("HF_INFERENCE_BASE_URL") or "https://router.huggingface.co/hf-inference/models").rstrip(
            "/"
        )

    def _check_model_live(self, model_id: str) -> tuple[bool, str | None]:
        """
        HF Inference "proxy" returns 200 Ok for models it can serve, and 404 for
        models that aren't deployed/available for serverless inference.
        """

        cached = self._model_live_cache.get(model_id)
        if cached is not None:
            return cached

        token = self._token()
        if not token:
            out = (False, "HF_TOKEN (or HUGGINGFACE_API_TOKEN) not set")
            self._model_live_cache[model_id] = out
            return out

        # Keep slashes in repo ids; URL-encode everything else.
        url = f"{self._base_url()}/{urllib.parse.quote(model_id, safe='/')}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "persian-voice/hf-inference-tts"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                resp.read(2)  # drain a tiny body ("Ok")
            out = (True, None)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                out = (
                    False,
                    "Model not available on Hugging Face serverless inference (404). "
                    "Use a different provider or deploy/run it yourself.",
                )
            elif exc.code in {401, 403}:
                out = (False, f"Hugging Face auth rejected the token (HTTP {exc.code}).")
            else:
                out = (False, f"Hugging Face preflight failed (HTTP {exc.code}).")
        except Exception as exc:  # noqa: BLE001
            out = (False, f"Hugging Face preflight failed: {type(exc).__name__}: {exc}")

        self._model_live_cache[model_id] = out
        return out

    def _model_ids(self) -> list[str]:
        # Optional override for experimentation; the repo has a curated default list.
        raw = (os.environ.get("HF_INFERENCE_MODEL_IDS") or "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        return self._default_model_ids()

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        provider_available, provider_reason = self.is_available()

        for hf_model_id in self._model_ids():
            model_available = provider_available
            model_reason = provider_reason
            if model_available:
                model_available, model_reason = self._check_model_live(hf_model_id)
            engine_id = hf_model_id
            group = f"{self.provider_label} · {hf_model_id}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/default/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=None,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="wav",
                    available=model_available,
                    unavailable_reason=model_reason,
                )

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        token = self._token()
        if not token:
            raise RuntimeError("HF_TOKEN (or HUGGINGFACE_API_TOKEN) not set")

        base = self._base_url()
        params = {"wait_for_model": "true"} if self._wait_for_model else {}
        # Keep slashes in repo ids; URL-encode everything else.
        url = f"{base}/{urllib.parse.quote(model.engine_id, safe='/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        payload = {"inputs": text}
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "audio/wav,application/json",
            "User-Agent": "persian-voice/hf-inference-tts",
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
                        content_type = (resp.headers.get("content-type") or "").lower()
                        body = resp.read()

                    if not body:
                        raise RuntimeError("HF Inference returned empty response.")

                    audio_bytes: bytes | None = None
                    if content_type.startswith("audio/") or content_type == "application/octet-stream":
                        audio_bytes = body
                    else:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                        if isinstance(data, dict):
                            if isinstance(data.get("error"), str):
                                raise RuntimeError(data["error"])
                            # Some endpoints might return a base64 field.
                            for key in ("audio", "audio_content", "audioContent"):
                                val = data.get(key)
                                if isinstance(val, str) and val.strip():
                                    audio_bytes = base64.b64decode(val)
                                    break
                        if audio_bytes is None:
                            raise RuntimeError("HF Inference returned JSON but no audio payload.")

                    if not audio_bytes:
                        raise RuntimeError("HF Inference returned empty audio.")

                    tmp_path.write_bytes(audio_bytes)
                    tmp_path.replace(out_path)
                    return {
                        "provider_id": self.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": None,
                        "input_kind": model.input_kind,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                except urllib.error.HTTPError as exc:
                    last_exc = exc
                    retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                    if attempt < self._max_retries and retryable:
                        time.sleep(min(0.5 * (2**attempt), 6.0))
                        continue
                    body_text = ""
                    try:
                        body_text = exc.read(800).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    raise RuntimeError(f"HF Inference HTTP {exc.code}: {body_text}".strip()) from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        time.sleep(min(0.5 * (2**attempt), 6.0))
                        continue
                    raise

            raise RuntimeError("HF Inference TTS failed.") from last_exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
