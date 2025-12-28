"""
Shared utilities for Replicate-based providers.

This module provides common functionality for providers that use
the Replicate API to run models in the cloud.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def get_replicate_token() -> str | None:
    """Get the Replicate API token from environment."""
    return os.environ.get("REPLICATE_API_TOKEN")


def is_replicate_available() -> tuple[bool, str | None]:
    """Check if Replicate API is available."""
    if not get_replicate_token():
        return False, "REPLICATE_API_TOKEN not set"
    return True, None


def run_replicate_model(model_ref: str, input_dict: dict) -> str:
    """
    Run a model on Replicate and return the output.

    Args:
        model_ref: Model reference (e.g., "owner/model:version")
        input_dict: Input parameters for the model

    Returns:
        The model output (typically a URL for file outputs)
    """
    import replicate

    return replicate.run(model_ref, input=input_dict)


def download_replicate_output(url: str, out_path: Path) -> None:
    """
    Download a file from a Replicate output URL.

    Args:
        url: URL to download from
        out_path: Local path to save the file
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
