"""Tests for EventBus."""

from __future__ import annotations

import pytest

from context_engineering_sdk.observability.event_bus import Event, InMemoryEventBus


@pytest.mark.asyncio
async def test_emit_and_handle():
    bus = InMemoryEventBus()
    received = []

    def handler(event: Event):
        received.append(event)

    bus.on("TestEvent", handler)
    await bus.emit(Event(event_type="TestEvent", session_id="s1"))

    assert len(received) == 1
    assert received[0].event_type == "TestEvent"


@pytest.mark.asyncio
async def test_async_handler():
    bus = InMemoryEventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.on("TestEvent", handler)
    await bus.emit(Event(event_type="TestEvent", session_id="s1"))

    assert len(received) == 1


@pytest.mark.asyncio
async def test_on_all():
    bus = InMemoryEventBus()
    received = []

    bus.on_all(lambda e: received.append(e))
    await bus.emit(Event(event_type="A", session_id="s1"))
    await bus.emit(Event(event_type="B", session_id="s1"))

    assert len(received) == 2


@pytest.mark.asyncio
async def test_off():
    bus = InMemoryEventBus()
    received = []

    def handler(event: Event):
        received.append(event)

    bus.on("TestEvent", handler)
    bus.off("TestEvent", handler)
    await bus.emit(Event(event_type="TestEvent", session_id="s1"))

    assert len(received) == 0


@pytest.mark.asyncio
async def test_history():
    bus = InMemoryEventBus()
    await bus.emit(Event(event_type="A", session_id="s1"))
    await bus.emit(Event(event_type="B", session_id="s1"))

    assert len(bus.history) == 2
    bus.clear_history()
    assert len(bus.history) == 0


@pytest.mark.asyncio
async def test_no_handler_no_error():
    bus = InMemoryEventBus()
    await bus.emit(Event(event_type="Unhandled", session_id="s1"))
