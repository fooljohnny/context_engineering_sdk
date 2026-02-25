"""End-to-end smoke tests for the full ContextEngine pipeline."""

from __future__ import annotations

import pytest

from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.config import BudgetConfig, RuntimeConfig, SummaryConfig
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.core.types import (
    Evidence,
    EvidenceLinks,
    EvidenceSource,
    EvidenceType,
    Message,
    ModelUsage,
    ModelUsageStage,
    Ref,
    Role,
    SourceKind,
    ToolCall,
    ToolCallStatus,
    ToolProvider,
    ProviderKind,
)
from context_engineering_sdk.engine import DefaultContextEngine, create_context_engine
from context_engineering_sdk.store.memory import MemoryStore

from tests.conftest import MockLlmAdapter, FixedClock


@pytest.mark.asyncio
async def test_e2e_smoke():
    """Basic smoke test: prepare_turn -> commit_assistant_message."""
    engine = create_context_engine(
        store=MemoryStore(),
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    cfg = RuntimeConfig()
    result = await engine.prepare_turn(
        "s1",
        Message(role=Role.USER, content="hello"),
        cfg,
    )
    assert result.assembled_input.total_tokens > 0
    assert result.report.errors == []

    put_result = await engine.commit_assistant_message(
        "s1", Message(role=Role.ASSISTANT, content="hi there!")
    )
    assert put_result.success


@pytest.mark.asyncio
async def test_e2e_streaming():
    """Test streaming workflow: prepare -> chunks -> finalize."""
    engine = create_context_engine(
        store=MemoryStore(),
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    cfg = RuntimeConfig()
    result = await engine.prepare_turn(
        "s1",
        Message(role=Role.USER, content="Tell me a story"),
        cfg,
    )

    # Simulate streaming chunks
    await engine.commit_assistant_chunk("s1", "Once upon ", 0)
    await engine.commit_assistant_chunk("s1", "a time...", 1)
    put_result = await engine.finalize_assistant_message("s1")
    assert put_result.success

    # Verify the message was assembled correctly
    doc = await engine._store.get_session("s1")
    assert doc is not None
    assistant_msgs = [m for m in doc.session.messages if m.role == Role.ASSISTANT]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == "Once upon a time..."


@pytest.mark.asyncio
async def test_e2e_tool_call_and_evidence():
    """Test tool call recording and evidence ingestion flow."""
    store = MemoryStore()
    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    cfg = RuntimeConfig()

    # First turn
    result = await engine.prepare_turn(
        "s1",
        Message(role=Role.USER, content="What is order 123?"),
        cfg,
    )

    # Record a tool call
    tc = ToolCall(
        tool_call_id="tc1",
        tool="getOrder",
        provider=ToolProvider(kind=ProviderKind.BUILTIN, name="order-svc"),
        status=ToolCallStatus.SUCCESS,
        duration_ms=50,
    )
    await engine.record_tool_call("s1", tc, result_evidence_ids=[])

    # Ingest tool result as evidence
    from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions

    ingestor = DefaultEvidenceIngestor(store=store)
    evidence, _ = await ingestor.ingest(
        session_id="s1",
        content='{"order_id": "123", "status": "shipped"}',
        source=EvidenceSource(kind=SourceKind.TOOL, name="getOrder"),
        evidence_type=EvidenceType.TOOL_RESULT,
        links=EvidenceLinks(tool_call_id="tc1"),
        options=IngestOptions(redact=False),
    )

    # Record model usage
    mu = ModelUsage(
        model_usage_id="mu1",
        provider="openai",
        model="gpt-4",
        stage=ModelUsageStage.ANSWER,
        total_tokens=500,
    )
    await engine.record_model_usage("s1", mu)

    # Commit assistant message
    await engine.commit_assistant_message(
        "s1",
        Message(role=Role.ASSISTANT, content="Order 123 has been shipped."),
        refs=[Ref(evidence_id=evidence.evidence_id)],
    )

    # Verify state
    doc = await store.get_session("s1")
    assert len(doc.session.tool_state.tool_calls) == 1
    assert len(doc.session.model_usage) == 1
    assert evidence.evidence_id in doc.evidences


@pytest.mark.asyncio
async def test_e2e_multi_turn():
    """Test multiple conversation turns."""
    engine = create_context_engine(
        store=MemoryStore(),
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    cfg = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=4096, reserved_reply_tokens=512),
        summary=SummaryConfig(enabled=False),
    )

    # Turn 1
    r1 = await engine.prepare_turn(
        "s1", Message(role=Role.USER, content="Hello"), cfg
    )
    await engine.commit_assistant_message(
        "s1", Message(role=Role.ASSISTANT, content="Hi! How can I help?")
    )

    # Turn 2
    r2 = await engine.prepare_turn(
        "s1", Message(role=Role.USER, content="What can you do?"), cfg
    )
    await engine.commit_assistant_message(
        "s1",
        Message(role=Role.ASSISTANT, content="I can help with many things!"),
    )

    # Turn 3
    r3 = await engine.prepare_turn(
        "s1", Message(role=Role.USER, content="Tell me about context engineering"), cfg
    )

    assert r3.assembled_input.total_tokens > r1.assembled_input.total_tokens
    doc = await engine._store.get_session("s1")
    # 3 user messages + 2 assistant messages = 5 messages
    assert len(doc.session.messages) == 5


@pytest.mark.asyncio
async def test_e2e_budget_constraint():
    """Test that pruning respects budget constraints."""
    engine = create_context_engine(
        store=MemoryStore(),
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    # Very tight budget
    cfg = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=200, reserved_reply_tokens=50),
        summary=SummaryConfig(enabled=False),
    )

    result = await engine.prepare_turn(
        "s1",
        Message(role=Role.USER, content="A" * 500),  # Large message
        cfg,
    )

    # Report should show pruning decisions
    assert len(result.report.prune_decisions) > 0


@pytest.mark.asyncio
async def test_e2e_with_evidences_in_second_turn():
    """Test that evidences ingested between turns appear in context."""
    store = MemoryStore()
    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    cfg = RuntimeConfig(summary=SummaryConfig(enabled=False))

    # Turn 1
    await engine.prepare_turn(
        "s1", Message(role=Role.USER, content="Search for docs"), cfg
    )
    await engine.commit_assistant_message(
        "s1", Message(role=Role.ASSISTANT, content="Searching...")
    )

    # Ingest evidence between turns
    from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions

    ingestor = DefaultEvidenceIngestor(store=store)
    ev, _ = await ingestor.ingest(
        session_id="s1",
        content="Important document content about context engineering.",
        source=EvidenceSource(kind=SourceKind.RAG, name="wiki"),
        evidence_type=EvidenceType.RAG_DOC,
        options=IngestOptions(redact=False),
    )

    # Turn 2: should include the evidence
    result = await engine.prepare_turn(
        "s1", Message(role=Role.USER, content="What did you find?"), cfg
    )

    # Check that assembled input contains evidence content
    assembled_text = result.assembled_input.text or ""
    assert "context engineering" in assembled_text.lower() or len(result.report.new_block_ids) > 0
