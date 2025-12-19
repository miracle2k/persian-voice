from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ..schema import ModelVariant


class Provider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str: ...

    @property
    @abstractmethod
    def provider_label(self) -> str: ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str | None]: ...

    @abstractmethod
    def list_model_variants(self) -> Iterable[ModelVariant]: ...

    @abstractmethod
    def synthesize(self, *, model: ModelVariant, text: str, out_path: Path) -> dict: ...

