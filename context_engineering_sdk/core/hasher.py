"""Content hashing for deduplication."""

from __future__ import annotations

import hashlib
from typing import Protocol


class Hasher(Protocol):
    def digest(self, content: str) -> str: ...


class Sha256Hasher:
    """Default implementation: SHA-256."""

    def digest(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
