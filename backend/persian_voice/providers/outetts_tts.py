from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..schema import ModelVariant, TEXT_KIND
from .base import Provider


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class OuteTTSTTSProvider(Provider):
    """
    OuteTTS 1.0 (1B) local provider via the `outetts` library.

    This downloads model weights from Hugging Face when first used.

    Model:
      - OuteAI/Llama-OuteTTS-1.0-1B
    """

    _MODEL_ID = "OuteAI/Llama-OuteTTS-1.0-1B"
    _DEFAULT_SPEAKERS = ["EN-FEMALE-1-NEUTRAL"]

    def __init__(self) -> None:
        self._interface_cache: dict[tuple[str, str | None], object] = {}
        self._speaker_cache: dict[tuple[tuple[str, str | None], str], object] = {}

    @property
    def provider_id(self) -> str:
        return "outetts"

    @property
    def provider_label(self) -> str:
        return "OuteTTS"

    def _enabled(self) -> bool:
        return _env_bool("OUTETTS_ENABLED", False)

    def is_available(self) -> tuple[bool, str | None]:
        if not self._enabled():
            return False, "OUTETTS_ENABLED not set (set to 1 to enable local model downloads)"
        if importlib.util.find_spec("outetts") is None:
            return False, "`outetts` not installed (install local-model deps)"
        return True, None

    def _speakers(self) -> list[str]:
        raw = (os.environ.get("OUTETTS_SPEAKERS") or "").strip()
        if raw:
            speakers = [s.strip() for s in raw.split(",") if s.strip()]
            if speakers:
                return speakers
        return list(self._DEFAULT_SPEAKERS)

    def _backend_key(self) -> tuple[str, str | None]:
        backend = (os.environ.get("OUTETTS_BACKEND") or "hf").strip().lower() or "hf"
        quant = (os.environ.get("OUTETTS_LLAMACPP_QUANTIZATION") or "").strip() or None
        if backend != "llamacpp":
            quant = None
        return backend, quant

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        available, reason = self.is_available()

        engine_id = self._MODEL_ID
        for speaker_id in self._speakers():
            group = f"{self.provider_label} · {engine_id} · {speaker_id}"
            for input_kind in input_kinds:
                model_id = f"{self.provider_id}/{engine_id}/{speaker_id}/{input_kind}"
                yield ModelVariant(
                    id=model_id,
                    provider_id=self.provider_id,
                    provider_label=self.provider_label,
                    engine_id=engine_id,
                    voice_id=speaker_id,
                    input_kind=input_kind,
                    label=f"{group} — {input_kind}",
                    group=group,
                    audio_format="wav",
                    available=available,
                    unavailable_reason=reason,
                )

    def _get_interface(self) -> tuple[tuple[str, str | None], object]:
        """
        Lazily initialize the OuteTTS interface.

        Avoids downloading weights until the first synthesis call.
        """

        backend, quant = self._backend_key()
        key = (backend, quant)
        cached = self._interface_cache.get(key)
        if cached is not None:
            return key, cached

        import outetts  # type: ignore[import-not-found]

        if backend == "llamacpp":
            quant_name = (quant or "FP16").strip()
            quant_enum = getattr(outetts.LlamaCppQuantization, quant_name, None)
            if quant_enum is None:
                raise RuntimeError(
                    f"Unknown OUTETTS_LLAMACPP_QUANTIZATION={quant_name!r}. "
                    "Expected a member of outetts.LlamaCppQuantization."
                )

            config = outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.LLAMACPP,
                quantization=quant_enum,
            )
        else:
            config = outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_1B,
                backend=outetts.Backend.HF,
            )

        interface = outetts.Interface(config=config)
        self._interface_cache[key] = interface
        return key, interface

    def _get_speaker(self, *, interface_key: tuple[str, str | None], interface: object, speaker_id: str) -> object:
        cached = self._speaker_cache.get((interface_key, speaker_id))
        if cached is not None:
            return cached

        speaker = interface.load_default_speaker(speaker_id)  # type: ignore[attr-defined]
        self._speaker_cache[(interface_key, speaker_id)] = speaker
        return speaker

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason or "OuteTTS provider unavailable")

        import outetts  # type: ignore[import-not-found]

        speaker_id = model.voice_id or self._DEFAULT_SPEAKERS[0]
        temperature = _env_float("OUTETTS_TEMPERATURE", 0.4)

        interface_key, interface = self._get_interface()
        speaker = self._get_speaker(interface_key=interface_key, interface=interface, speaker_id=speaker_id)

        generation = interface.generate(  # type: ignore[attr-defined]
            config=outetts.GenerationConfig(
                text=text,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(temperature=temperature),
            )
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            generation.save(str(tmp_path))  # type: ignore[attr-defined]
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("OuteTTS produced no audio output.")
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return {
            "provider_id": self.provider_id,
            "engine_id": model.engine_id,
            "voice_id": speaker_id,
            "input_kind": model.input_kind,
            "outetts_model_id": self._MODEL_ID,
            "outetts_backend": interface_key[0],
            "outetts_llamacpp_quantization": interface_key[1],
            "temperature": temperature,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
