from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import urllib.request
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


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


class ManaTTSProvider(Provider):
    """
    ManaTTS Persian Tacotron2 provider.

    Uses the Persian-MultiSpeaker-Tacotron2 model trained on ManaTTS dataset
    (114+ hours of high-quality Persian audio at 24kHz).

    Architecture:
      - Speaker Encoder: extracts speaker embedding from reference audio
      - Tacotron2 Synthesizer: generates mel spectrogram from text + embedding
      - Griffin-Lim Vocoder: converts spectrogram to waveform (built-in)

    Voice:
      - mana (default): Female voice from ManaTTS sample

    Model Sizes:
      - synthesizer.pt: ~354 MB
      - encoder.pt: ~16 MB
      - sample.wav: ~0.5 MB
      - Total: ~370 MB

    Environment Variables:
      - MANATTS_ENABLED: Set to 1 to enable (default: 0)
      - MANATTS_CACHE_DIR: Cache directory (default: ~/.cache/manatts)

    Note: First use will clone repo and download models (~370 MB).
    Uses Griffin-Lim vocoder for waveform synthesis.
    """

    _REPO_URL = "https://github.com/MahtaFetrat/Persian-MultiSpeaker-Tacotron2.git"
    _HF_BASE = (
        "https://huggingface.co/MahtaFetrat/Persian-Tacotron2-on-ManaTTS/resolve/main"
    )
    _DEFAULT_VOICE = "mana"

    def __init__(self) -> None:
        self._cache: dict | None = None

    @property
    def provider_id(self) -> str:
        return "manatts"

    @property
    def provider_label(self) -> str:
        return "ManaTTS"

    def _enabled(self) -> bool:
        return _env_bool("MANATTS_ENABLED", False)

    def _get_cache_dir(self) -> Path:
        cache_dir = _env_str("MANATTS_CACHE_DIR", "~/.cache/manatts")
        return Path(cache_dir).expanduser()

    def is_available(self) -> tuple[bool, str | None]:
        if not self._enabled():
            return False, "MANATTS_ENABLED not set (set to 1 to enable)"

        if importlib.util.find_spec("torch") is None:
            return False, "`torch` not installed"
        if importlib.util.find_spec("librosa") is None:
            return False, "`librosa` not installed"
        if importlib.util.find_spec("soundfile") is None:
            return False, "`soundfile` not installed"

        return True, None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        # ManaTTS only supports Persian script (trained on Persian text only)
        input_kinds: list[TEXT_KIND] = ["fa", "fa_diac"]
        available, reason = self.is_available()

        voice_id = self._DEFAULT_VOICE
        engine_id = "tacotron2-griffinlim"
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
                audio_format="wav",
                available=available,
                unavailable_reason=reason,
            )

    def _setup_repo(self) -> Path:
        """Clone the repo if not already present."""
        cache_dir = self._get_cache_dir()
        repo_dir = cache_dir / "repo"

        if not (repo_dir / "inference.py").exists():
            repo_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ManaTTS] Cloning repository to {repo_dir}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", self._REPO_URL, str(repo_dir)],
                check=True,
                capture_output=True,
            )

        return repo_dir

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file with atomic write."""
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            print(f"[ManaTTS] Downloading {dest.name}...")
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(dest)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _setup_models(self, repo_dir: Path) -> Path:
        """Download model files if not present."""
        cache_dir = self._get_cache_dir()
        models_dir = cache_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Synthesizer from HuggingFace
        synth_path = models_dir / "synthesizer.pt"
        if not synth_path.exists():
            self._download_file(f"{self._HF_BASE}/synthesizer.pt", synth_path)

        # Encoder from repo (already included)
        encoder_src = repo_dir / "saved_models" / "default" / "encoder.pt"
        encoder_path = models_dir / "encoder.pt"
        if not encoder_path.exists() and encoder_src.exists():
            import shutil

            shutil.copy(encoder_src, encoder_path)

        # Reference voice from HuggingFace
        ref_path = models_dir / "sample.wav"
        if not ref_path.exists():
            self._download_file(f"{self._HF_BASE}/sample.wav", ref_path)

        return models_dir

    def _load(self) -> dict:
        """Load all models into memory."""
        if self._cache is not None:
            return self._cache

        import numpy as np
        import torch

        # Setup repo and models
        repo_dir = self._setup_repo()
        models_dir = self._setup_models(repo_dir)

        # Add repo to path for imports
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        # Import modules from repo
        from encoder import inference as encoder_module
        from synthesizer.inference import Synthesizer

        # Load encoder
        encoder_path = models_dir / "encoder.pt"
        encoder_module.load_model(encoder_path)

        # Load synthesizer
        synth_path = models_dir / "synthesizer.pt"
        synthesizer = Synthesizer(synth_path, verbose=False)

        # Load reference voice embedding
        ref_path = models_dir / "sample.wav"
        ref_wav = Synthesizer.load_preprocess_wav(ref_path)
        encoder_wav = encoder_module.preprocess_wav(ref_wav)
        ref_embed, _, _ = encoder_module.embed_utterance(
            encoder_wav, return_partials=True
        )

        self._cache = {
            "encoder": encoder_module,
            "synthesizer": synthesizer,
            "ref_embed": ref_embed,
            "sample_rate": Synthesizer.sample_rate,
        }
        return self._cache

    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict:
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(reason or "ManaTTS provider unavailable")

        import numpy as np
        import soundfile as sf

        cache = self._load()
        synthesizer = cache["synthesizer"]
        ref_embed = cache["ref_embed"]
        sample_rate = cache["sample_rate"]

        # Import Synthesizer class for griffin_lim
        from synthesizer.inference import Synthesizer as SynthClass

        # Synthesize spectrogram
        specs = synthesizer.synthesize_spectrograms([text], [ref_embed])
        spec = np.concatenate(specs, axis=1)

        # Convert to waveform via Griffin-Lim
        wav = SynthClass.griffin_lim(spec)

        # Normalize
        wav = wav / np.abs(wav).max() * 0.97

        # Save output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            sf.write(str(tmp_path), wav, sample_rate, format="WAV")
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("ManaTTS produced no audio output.")
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
            "voice_id": model.voice_id,
            "input_kind": model.input_kind,
            "sample_rate": sample_rate,
            "vocoder": "griffin-lim",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
