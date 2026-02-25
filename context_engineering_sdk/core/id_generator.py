"""ID generation utilities."""

from __future__ import annotations

import uuid
from typing import Protocol


class IdGenerator(Protocol):
    def generate(self) -> str: ...


class UuidV4Generator:
    """Default implementation: UUID v4."""

    def generate(self) -> str:
        return str(uuid.uuid4())
