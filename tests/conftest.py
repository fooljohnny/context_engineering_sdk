"""Shared test fixtures."""

from __future__ import annotations

import pytest

from context_engineering_sdk.builder.summarizer import LlmAdapter, LlmRequest, LlmResponse
from context_engineering_sdk.core.clock import Clock
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.core.types import MessagePart, Role
from context_engineering_sdk.store.memory import MemoryStore


class MockLlmAdapter:
    """LLM adapter that echoes a canned summary."""

    def __init__(self, response_content: str = "This is a summary of the conversation."):
        self._content = response_content
        self.calls: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return LlmResponse(
            content=self._content,
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            latency_ms=10,
        )


class FixedClock:
    """Clock that returns a fixed timestamp."""

    def __init__(self, ts: str = "2026-02-25T10:00:00+00:00"):
        self._ts = ts

    def now_iso(self) -> str:
        return self._ts


@pytest.fixture
def memory_store():
    return MemoryStore()


@pytest.fixture
def token_estimator():
    return CharBasedEstimator()


@pytest.fixture
def mock_llm():
    return MockLlmAdapter()


@pytest.fixture
def fixed_clock():
    return FixedClock()
