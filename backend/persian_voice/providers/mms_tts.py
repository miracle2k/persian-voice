from __future__ import annotations

import array
import importlib.util
import os
import wave
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


class MMSTTSProvider(Provider):
    """
    Meta MMS TTS (Persian checkpoint) local provider.

    This downloads model weights from Hugging Face via `transformers` when first used.

    Model:
      - facebook/mms-tts-fas

    Known Issues:
      The MMS Persian model produces poor quality output - spoken words often sound
      completely different from the input text (e.g., "سلام" sounds like "vam").
      We verified this is NOT a HuggingFace implementation bug: the original fairseq
      inference produces identical output when using the same random seed. The issue
      appears to be with the facebook/mms-tts-fas model weights themselves.

    TODO: Investigate further by testing a non-Persian MMS model (e.g., German
      mms-tts-deu) to determine if this is a Persian-specific training issue or
      a broader problem with MMS TTS models.
    """

    _MODEL_ID = "facebook/mms-tts-fas"

    def __init__(self) -> None:
        self._cache: tuple[object, object, object] | None = None

    @property
    def provider_id(self) -> str:
        return "mms"

    @property
    def provider_label(self) -> str:
        return "Meta MMS TTS"

    def _enabled(self) -> bool:
        return _env_bool("MMS_TTS_ENABLED", False)

    def is_available(self) -> tuple[bool, str | None]:
        if not self._enabled():
            return False, "MMS_TTS_ENABLED not set (set to 1 to enable local model downloads)"

        if importlib.util.find_spec("torch") is None:
            return False, "`torch` not installed (install local-model deps)"
        if importlib.util.find_spec("transformers") is None:
            return False, "`transformers` not installed (install local-model deps)"

        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac", "latn"]
        available, reason = self.is_available()

        engine_id = self._MODEL_ID
        group = f"{self.provider_label} · {engine_id}"
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

    def _load(self) -> tuple[object, object, object]:
        if self._cache is not None:
            return self._cache

        import torch  # type: ignore[import-not-found]
        from transformers import AutoTokenizer, VitsModel  # type: ignore[import-not-found]

        device_raw = (os.environ.get("MMS_TTS_DEVICE") or "").strip().lower()
        if device_raw:
            device = torch.device(device_raw)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)
        model = VitsModel.from_pretrained(self._MODEL_ID)
        model.to(device)
        model.eval()

        self._cache = (tokenizer, model, device)
        return self._cache

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason or "MMS provider unavailable")

        import torch  # type: ignore[import-not-found]

        tokenizer_obj, model_obj, device_obj = self._load()
        tokenizer = tokenizer_obj
        tts_model = model_obj
        device = device_obj

        # Torch typing: keep this minimal without pulling in torch stubs.
        inputs = tokenizer(text, return_tensors="pt")  # type: ignore[misc]
        inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore[union-attr]

        with torch.no_grad():
            output = tts_model(**inputs)  # type: ignore[misc]

        waveform = output.waveform[0].detach().to("cpu")  # type: ignore[attr-defined]
        sample_rate = int(getattr(tts_model.config, "sampling_rate", 16000))  # type: ignore[union-attr]

        pcm16 = (waveform * 32767.0).clamp(-32768, 32767).to(torch.int16)  # type: ignore[attr-defined]
        frames = array.array("h", pcm16.tolist()).tobytes()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(frames)
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("MMS produced no audio output.")
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
            "voice_id": None,
            "input_kind": model.input_kind,
            "mms_model_id": self._MODEL_ID,
            "mms_device": str(device),
            "sample_rate": sample_rate,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

