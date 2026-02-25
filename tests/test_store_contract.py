"""Contract tests for Store implementations.

All Store implementations must pass these tests.
"""

from __future__ import annotations

import pytest

from context_engineering_sdk.core.errors import SessionNotFoundError, VersionConflictError
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Evidence,
    EvidenceLinks,
    EvidenceSource,
    EvidenceType,
    Message,
    ModelUsage,
    ModelUsageStage,
    ModelUsageStatus,
    Priority,
    PutResult,
    Role,
    Session,
    SessionDocument,
    SourceKind,
    Task,
    TaskStatus,
    ToolCall,
    ToolCallStatus,
    ToolProvider,
    ProviderKind,
)
from context_engineering_sdk.store.base import BlockFilter, EvidenceFilter
from context_engineering_sdk.store.memory import MemoryStore


def _make_doc(session_id: str = "s1") -> SessionDocument:
    return SessionDocument(
        schema_version="1.0",
        session=Session(
            session_id=session_id,
            messages=[
                Message(role=Role.SYSTEM, content="You are a helper."),
                Message(role=Role.USER, content="Hello!"),
            ],
        ),
    )


def _make_evidence(eid: str = "ev1") -> Evidence:
    return Evidence(
        evidence_id=eid,
        type=EvidenceType.TOOL_RESULT,
        source=EvidenceSource(kind=SourceKind.TOOL, name="test_tool"),
        content="some result content",
        confidence=0.9,
    )


# ---- Session tests ----

@pytest.mark.asyncio
async def test_put_and_get_session(memory_store):
    store = memory_store
    doc = _make_doc()
    result = await store.put_session("s1", doc)
    assert result.success
    assert result.version == 1

    loaded = await store.get_session("s1")
    assert loaded is not None
    assert loaded.session.session_id == "s1"
    assert len(loaded.session.messages) == 2


@pytest.mark.asyncio
async def test_get_nonexistent_session(memory_store):
    result = await memory_store.get_session("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_append_messages_ordering(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    msg = Message(role=Role.ASSISTANT, content="Hi there!")
    result = await store.append_messages("s1", [msg])
    assert result.success

    loaded = await store.get_session("s1")
    assert len(loaded.session.messages) == 3
    assert loaded.session.messages[-1].content == "Hi there!"


@pytest.mark.asyncio
async def test_append_messages_creates_session_if_missing(memory_store):
    store = memory_store
    msg = Message(role=Role.USER, content="Hello!")
    result = await store.append_messages("new_session", [msg])
    assert result.success

    loaded = await store.get_session("new_session")
    assert loaded is not None
    assert len(loaded.session.messages) == 1


@pytest.mark.asyncio
async def test_version_conflict_on_put(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)  # version = 1

    doc2 = _make_doc()
    with pytest.raises(VersionConflictError):
        await store.put_session("s1", doc2, expected_version=999)


@pytest.mark.asyncio
async def test_put_session_version_increment(memory_store):
    store = memory_store
    doc = _make_doc()
    r1 = await store.put_session("s1", doc)
    assert r1.version == 1

    r2 = await store.put_session("s1", doc, expected_version=1)
    assert r2.version == 2


# ---- Evidence tests ----

@pytest.mark.asyncio
async def test_put_and_get_evidence(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    ev = _make_evidence("ev1")
    eid = await store.put_evidence("s1", ev)
    assert eid == "ev1"

    loaded = await store.get_evidence("s1", "ev1")
    assert loaded is not None
    assert loaded.content == "some result content"


@pytest.mark.asyncio
async def test_get_nonexistent_evidence(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)
    result = await store.get_evidence("s1", "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_evidences_with_filter(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    ev1 = Evidence(
        evidence_id="ev1",
        type=EvidenceType.TOOL_RESULT,
        source=EvidenceSource(kind=SourceKind.TOOL),
        content="tool result",
    )
    ev2 = Evidence(
        evidence_id="ev2",
        type=EvidenceType.RAG_DOC,
        source=EvidenceSource(kind=SourceKind.RAG),
        content="rag doc",
    )
    ev3 = Evidence(
        evidence_id="ev3",
        type=EvidenceType.TOOL_RESULT,
        source=EvidenceSource(kind=SourceKind.TOOL),
        content="another tool result",
    )

    await store.put_evidence("s1", ev1)
    await store.put_evidence("s1", ev2)
    await store.put_evidence("s1", ev3)

    # Filter by type
    results = await store.list_evidences(
        "s1", EvidenceFilter(types=[EvidenceType.TOOL_RESULT])
    )
    assert len(results) == 2

    # Filter by source kind
    results = await store.list_evidences(
        "s1", EvidenceFilter(source_kinds=[SourceKind.RAG])
    )
    assert len(results) == 1
    assert results[0].evidence_id == "ev2"

    # Filter with limit
    results = await store.list_evidences(
        "s1", EvidenceFilter(limit=1)
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_idempotent_evidence_put(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    ev = _make_evidence("ev1")
    await store.put_evidence("s1", ev)
    await store.put_evidence("s1", ev)  # idempotent (same id overwrites)

    results = await store.list_evidences("s1")
    assert len(results) == 1


# ---- Context Blocks tests ----

@pytest.mark.asyncio
async def test_put_and_list_context_blocks(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    blocks = [
        ContextBlock(
            block_id="b1",
            block_type=BlockType.INSTRUCTION,
            priority=Priority.MUST,
            token_estimate=100,
            content="instructions",
        ),
        ContextBlock(
            block_id="b2",
            block_type=BlockType.EVIDENCE,
            priority=Priority.MEDIUM,
            token_estimate=200,
            content="evidence content",
        ),
    ]

    result = await store.put_context_blocks("s1", blocks)
    assert result.success

    loaded = await store.list_context_blocks("s1")
    assert len(loaded) == 2


@pytest.mark.asyncio
async def test_list_context_blocks_with_filter(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    blocks = [
        ContextBlock(block_id="b1", block_type=BlockType.INSTRUCTION,
                     priority=Priority.MUST, token_estimate=100),
        ContextBlock(block_id="b2", block_type=BlockType.EVIDENCE,
                     priority=Priority.LOW, token_estimate=200),
        ContextBlock(block_id="b3", block_type=BlockType.CONVERSATION,
                     priority=Priority.HIGH, token_estimate=150),
    ]
    await store.put_context_blocks("s1", blocks)

    # Filter by type
    results = await store.list_context_blocks(
        "s1", BlockFilter(block_types=[BlockType.INSTRUCTION])
    )
    assert len(results) == 1

    # Filter by min_priority
    results = await store.list_context_blocks(
        "s1", BlockFilter(min_priority=Priority.HIGH)
    )
    assert len(results) == 2  # MUST and HIGH


# ---- Tool / Model / Task tests ----

@pytest.mark.asyncio
async def test_append_tool_calls(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    tc = ToolCall(
        tool_call_id="tc1",
        tool="getOrder",
        provider=ToolProvider(kind=ProviderKind.BUILTIN, name="order-svc"),
        status=ToolCallStatus.SUCCESS,
        duration_ms=50,
    )
    result = await store.append_tool_calls("s1", [tc])
    assert result.success

    loaded = await store.get_session("s1")
    assert len(loaded.session.tool_state.tool_calls) == 1


@pytest.mark.asyncio
async def test_append_model_usage(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    mu = ModelUsage(
        model_usage_id="mu1",
        provider="openai",
        model="gpt-4",
        stage=ModelUsageStage.ANSWER,
        total_tokens=500,
    )
    result = await store.append_model_usage("s1", [mu])
    assert result.success

    loaded = await store.get_session("s1")
    assert len(loaded.session.model_usage) == 1
    assert loaded.session.model_usage[0].total_tokens == 500


@pytest.mark.asyncio
async def test_upsert_tasks(memory_store):
    store = memory_store
    doc = _make_doc()
    await store.put_session("s1", doc)

    t1 = Task(task_id="t1", name="Research", status=TaskStatus.PENDING)
    await store.upsert_tasks("s1", [t1])

    loaded = await store.get_session("s1")
    assert len(loaded.session.task_state.todo_list.tasks) == 1

    # Update existing task
    t1_updated = Task(task_id="t1", name="Research", status=TaskStatus.COMPLETED)
    await store.upsert_tasks("s1", [t1_updated])

    loaded = await store.get_session("s1")
    assert len(loaded.session.task_state.todo_list.tasks) == 1
    assert loaded.session.task_state.todo_list.tasks[0].status == TaskStatus.COMPLETED
