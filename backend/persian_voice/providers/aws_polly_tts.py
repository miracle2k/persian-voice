from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..schema import ModelVariant, TEXT_KIND
from .base import Provider


class AWSPollyTTSProvider(Provider):
    """
    Amazon Polly provider (via AWS CLI).

    This avoids adding boto3/SigV4 dependencies. Requires `aws` CLI + credentials
    configured (env vars, shared config, SSO, etc.).

    Docs:
      - https://docs.aws.amazon.com/cli/latest/reference/polly/synthesize-speech.html
    """

    @property
    def provider_id(self) -> str:
        return "aws_polly"

    @property
    def provider_label(self) -> str:
        return "Amazon Polly"

    def is_available(self) -> tuple[bool, str | None]:
        if not shutil.which("aws"):
            return False, "`aws` CLI not found in PATH"
        enabled = (os.environ.get("AWS_POLLY_ENABLED") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
        if not enabled:
            return False, "AWS_POLLY_ENABLED not set (set to 1 to enable)"
        return True, None

    def _voice_ids(self) -> list[str]:
        raw = (os.environ.get("AWS_POLLY_VOICE_IDS") or "Joanna").strip()
        return [v.strip() for v in raw.split(",") if v.strip()] or ["Joanna"]

    def _engine(self) -> str | None:
        raw = (os.environ.get("AWS_POLLY_ENGINE") or "").strip().lower()
        if raw in {"standard", "neural", "generative"}:
            return raw
        return None

    def _text_type(self) -> str | None:
        raw = (os.environ.get("AWS_POLLY_TEXT_TYPE") or "").strip().lower()
        if raw in {"text", "ssml"}:
            return raw
        return None

    def list_model_variants(self) -> Iterable[ModelVariant]:
        input_kinds: list[TEXT_KIND] = ["fa", "fa_latn", "latn"]
        engine = self._engine()
        engine_id = "polly" + (f":{engine}" if engine else "")

        available, reason = self.is_available()
        for voice_id in self._voice_ids():
            group = f"{self.provider_label} · {voice_id}" + (f" · {engine}" if engine else "")
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
        if not shutil.which("aws"):
            raise RuntimeError("`aws` CLI not found in PATH")
        if not model.voice_id:
            raise RuntimeError("AWS Polly requires a voice_id")

        engine = self._engine()
        text_type = self._text_type()
        region = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip() or None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        cmd = [
            "aws",
            "polly",
            "synthesize-speech",
            "--output-format",
            model.audio_format,
            "--voice-id",
            model.voice_id,
            "--text",
            text,
            str(tmp_path),
        ]
        if region:
            cmd.extend(["--region", region])
        if engine:
            cmd.extend(["--engine", engine])
        if text_type:
            cmd.extend(["--text-type", text_type])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                msg = stderr or stdout or f"aws exited with code {proc.returncode}"
                raise RuntimeError(f"AWS Polly synthesize-speech failed: {msg}")
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("AWS Polly produced no audio output.")
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
            "aws_region": region,
            "aws_engine": engine,
            "aws_text_type": text_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
