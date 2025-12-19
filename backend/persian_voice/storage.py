from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def public_dir() -> Path:
    override = os.environ.get("PERSIAN_VOICE_PUBLIC_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "web" / "public"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(encoded)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)

