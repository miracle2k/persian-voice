from __future__ import annotations

import importlib.util
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from ..schema import ModelVariant, TEXT_KIND
from .base import Provider
from . import replicate_utils


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


BackendType = Literal["replicate", "local"]


class ChatterboxTTSProvider(Provider):
    """
    Chatterbox TTS Persian provider with multiple backend options.

    Uses the Chatterbox multilingual model with Persian fine-tuned weights from
    Thomcles/Chatterbox-TTS-Persian-Farsi on HuggingFace.

    Backends:
      - replicate: Cloud API via Replicate (no download, requires REPLICATE_API_TOKEN)
      - local: Local inference (~3.2GB download, requires chatterbox-tts)

    Backend selection (in order of priority):
      1. Explicit: CHATTERBOX_BACKEND=replicate|local
      2. Auto-detect: Replicate if token set, else local if enabled

    Environment Variables:
      - CHATTERBOX_BACKEND: Force backend (replicate|local)
      - CHATTERBOX_ENABLED: Set to 1 to enable local backend (default: 0)
      - CHATTERBOX_DEVICE: Device for local (cpu/cuda/mps, default: auto-detect)
      - CHATTERBOX_CACHE_DIR: Cache directory (default: ~/.cache/chatterbox-persian)
      - CHATTERBOX_EXAGGERATION: Voice exaggeration factor (default: 0.5)
      - CHATTERBOX_CFG_WEIGHT: Classifier-free guidance weight (default: 0.5)

    License: CC BY-NC 4.0 (non-commercial use only without permission)
    """

    _HF_BASE_URL = "https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi/resolve/main"
    _PERSIAN_WEIGHTS_FILE = "t3_fa.safetensors"
    _SAMPLE_RATE = 24000  # Chatterbox default sample rate

    _REPLICATE_MODEL = "miracle2k/chatterbox-persian:1d661e4e5e741084b58f5e13333082d92fdd3f50469854863f1209b7c85b10e2"

    def __init__(self) -> None:
        self._cache: tuple[object, object] | None = None  # (model, device)

    @property
    def provider_id(self) -> str:
        return "chatterbox"

    @property
    def provider_label(self) -> str:
        return "Chatterbox TTS Persian"

    def _local_enabled(self) -> bool:
        return _env_bool("CHATTERBOX_ENABLED", False)

    def _get_cache_dir(self) -> Path:
        cache_dir = _env_str("CHATTERBOX_CACHE_DIR", "~/.cache/chatterbox-persian")
        return Path(cache_dir).expanduser()

    def _detect_backend(self) -> BackendType | None:
        """Auto-detect the best available backend."""
        explicit = (os.environ.get("CHATTERBOX_BACKEND") or "").strip().lower()
        if explicit in ("replicate", "local"):
            return explicit  # type: ignore[return-value]

        # Auto-detect: prefer Replicate (no download needed)
        if replicate_utils.get_replicate_token():
            return "replicate"
        if self._local_enabled():
            return "local"
        return None

    def is_available(self) -> tuple[bool, str | None]:
        backend = self._detect_backend()
        if backend is None:
            return False, "Set REPLICATE_API_TOKEN (cloud) or CHATTERBOX_ENABLED=1 (local; ~3.2GB)"

        if backend == "replicate":
            return replicate_utils.is_replicate_available()

        # Local backend
        if not self._local_enabled():
            return False, "CHATTERBOX_ENABLED not set (set to 1 to enable local model)"
        if importlib.util.find_spec("torch") is None:
            return False, "`torch` not installed (install local-model deps)"
        if importlib.util.find_spec("chatterbox") is None:
            return False, "`chatterbox-tts` not installed (pip install chatterbox-tts)"
        if importlib.util.find_spec("safetensors") is None:
            return False, "`safetensors` not installed (pip install safetensors)"

        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        available, reason = self.is_available()
        backend = self._detect_backend()

        # Show backend in group label for clarity
        backend_label = {"replicate": "Cloud", "local": "Local"}.get(
            backend or "local", "Local"
        )

        engine_id = "chatterbox-persian"
        group = f"{self.provider_label} ({backend_label})"
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
                available=available,
                unavailable_reason=reason,
            )

    def _download_persian_weights(self) -> Path:
        """Download Persian weights from HuggingFace if not cached."""
        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        weights_path = cache_dir / self._PERSIAN_WEIGHTS_FILE

        if not weights_path.exists():
            url = f"{self._HF_BASE_URL}/{self._PERSIAN_WEIGHTS_FILE}"
            tmp_path = weights_path.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.replace(weights_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        return weights_path

    def _load(self) -> tuple[object, object]:
        """Load the Chatterbox model with Persian weights."""
        if self._cache is not None:
            return self._cache

        import torch  # type: ignore[import-not-found]
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import-not-found]
        from safetensors.torch import load_file as load_safetensors  # type: ignore[import-not-found]

        device_raw = (os.environ.get("CHATTERBOX_DEVICE") or "").strip().lower()
        if device_raw:
            device = device_raw
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Load the base multilingual model
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)

        # Download and load Persian fine-tuned weights
        persian_weights_path = self._download_persian_weights()
        t3_state = load_safetensors(str(persian_weights_path), device="cpu")
        model.t3.load_state_dict(t3_state)
        model.t3.to(device).eval()

        self._cache = (model, device)
        return self._cache

    def _synthesize_replicate(
        self, *, text: str, exaggeration: float, cfg_weight: float, out_path: Path
    ) -> dict:
        """Synthesize using Replicate cloud API."""
        output_url = replicate_utils.run_replicate_model(
            self._REPLICATE_MODEL,
            {
                "text": text,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            },
        )
        replicate_utils.download_replicate_output(str(output_url), out_path)
        return {"backend": "replicate", "replicate_model": self._REPLICATE_MODEL}

    def _synthesize_local(
        self, *, text: str, exaggeration: float, cfg_weight: float, out_path: Path
    ) -> dict:
        """Synthesize using local model."""
        import torchaudio as ta  # type: ignore[import-not-found]

        tts_model, device = self._load()

        # Generate speech with parameters
        wav = tts_model.generate(  # type: ignore[union-attr]
            text,
            language_id=None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        # Get sample rate from model
        sample_rate = getattr(tts_model, "sr", self._SAMPLE_RATE)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            ta.save(str(tmp_path), wav, sample_rate)
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("Chatterbox produced no audio output.")
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return {
            "backend": "local",
            "chatterbox_device": str(device),
            "sample_rate": sample_rate,
        }

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason or "Chatterbox provider unavailable")

        exaggeration = _env_float("CHATTERBOX_EXAGGERATION", 0.5)
        cfg_weight = _env_float("CHATTERBOX_CFG_WEIGHT", 0.5)
        backend = self._detect_backend()

        if backend == "replicate":
            backend_info = self._synthesize_replicate(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                out_path=out_path,
            )
        else:
            backend_info = self._synthesize_local(
                text=text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                out_path=out_path,
            )

        return {
            "provider_id": self.provider_id,
            "engine_id": model.engine_id,
            "voice_id": None,
            "input_kind": model.input_kind,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **backend_info,
        }
