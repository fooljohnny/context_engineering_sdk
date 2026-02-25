"""Event bus for SDK observability."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Event:
    event_type: str
    session_id: str
    turn_id: str | None = None
    task_id: str | None = None
    tool_call_id: str | None = None
    model_usage_id: str | None = None
    ts: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[Event], Any]


class EventBus(Protocol):
    async def emit(self, event: Event) -> None: ...
    def on(self, event_type: str, handler: EventHandler) -> None: ...
    def off(self, event_type: str, handler: EventHandler) -> None: ...
    def on_all(self, handler: EventHandler) -> None: ...


class InMemoryEventBus:
    """In-memory event bus with sync and async handler support."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._history: list[Event] = []

    async def emit(self, event: Event) -> None:
        self._history.append(event)

        handlers = list(self._global_handlers)
        if event.event_type in self._handlers:
            handlers.extend(self._handlers[event.event_type])

        for handler in handlers:
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    def on(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def on_all(self, handler: EventHandler) -> None:
        self._global_handlers.append(handler)

    @property
    def history(self) -> list[Event]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
