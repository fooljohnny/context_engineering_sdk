"""Clock abstraction for injectable time source."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now_iso(self) -> str: ...


class SystemClock:
    """Default implementation: system UTC clock."""

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
