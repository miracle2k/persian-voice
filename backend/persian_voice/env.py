from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def load_dotenv(*, override: bool = False) -> Path | None:
    """
    Load environment variables from a repo-local `.env` file if present.

    - Does nothing if `.env` does not exist.
    - Does not override already-set env vars unless `override=True`.
    - Keeps parsing intentionally simple (`KEY=VALUE`, optional `export ` prefix).
    """

    repo_root = _find_repo_root(Path(__file__).resolve())
    dotenv_path = repo_root / ".env"
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]

        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    return dotenv_path
