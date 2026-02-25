"""Tests for FileStore implementation."""

from __future__ import annotations

import tempfile

import pytest

from context_engineering_sdk.core.errors import SessionNotFoundError, VersionConflictError
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Evidence,
    EvidenceSource,
    EvidenceType,
    Message,
    ModelUsage,
    ModelUsageStage,
    Priority,
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
from context_engineering_sdk.store.file import FileStore


@pytest.fixture
def file_store(tmp_path):
    return FileStore(tmp_path / "sessions")


@pytest.mark.asyncio
async def test_file_store_put_and_get(file_store):
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(
            session_id="s1",
            messages=[
                Message(role=Role.SYSTEM, content="Hello"),
                Message(role=Role.USER, content="Hi"),
            ],
        ),
    )
    result = await file_store.put_session("s1", doc)
    assert result.success

    loaded = await file_store.get_session("s1")
    assert loaded is not None
    assert loaded.session.session_id == "s1"
    assert len(loaded.session.messages) == 2
    assert loaded.session.messages[0].role == Role.SYSTEM


@pytest.mark.asyncio
async def test_file_store_nonexistent(file_store):
    result = await file_store.get_session("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_file_store_append_messages(file_store):
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(session_id="s1", messages=[]),
    )
    await file_store.put_session("s1", doc)
    await file_store.append_messages("s1", [Message(role=Role.USER, content="New msg")])

    loaded = await file_store.get_session("s1")
    assert len(loaded.session.messages) == 1
    assert loaded.session.messages[0].content == "New msg"


@pytest.mark.asyncio
async def test_file_store_evidence_roundtrip(file_store):
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(session_id="s1"),
    )
    await file_store.put_session("s1", doc)

    ev = Evidence(
        evidence_id="ev1",
        type=EvidenceType.TOOL_RESULT,
        source=EvidenceSource(kind=SourceKind.TOOL, name="test"),
        content="tool output",
        confidence=0.9,
    )
    await file_store.put_evidence("s1", ev)

    loaded = await file_store.get_evidence("s1", "ev1")
    assert loaded is not None
    assert loaded.type == EvidenceType.TOOL_RESULT
    assert loaded.content == "tool output"
    assert loaded.confidence == 0.9


@pytest.mark.asyncio
async def test_file_store_context_blocks(file_store):
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(session_id="s1"),
    )
    await file_store.put_session("s1", doc)

    blocks = [
        ContextBlock(
            block_id="b1",
            block_type=BlockType.INSTRUCTION,
            priority=Priority.MUST,
            token_estimate=100,
            content="Be helpful",
        ),
    ]
    await file_store.put_context_blocks("s1", blocks)

    loaded = await file_store.list_context_blocks("s1")
    assert len(loaded) == 1
    assert loaded[0].block_type == BlockType.INSTRUCTION
    assert loaded[0].priority == Priority.MUST


@pytest.mark.asyncio
async def test_file_store_tool_calls(file_store):
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(session_id="s1"),
    )
    await file_store.put_session("s1", doc)

    tc = ToolCall(
        tool_call_id="tc1",
        tool="search",
        provider=ToolProvider(kind=ProviderKind.BUILTIN, name="svc"),
        status=ToolCallStatus.SUCCESS,
    )
    await file_store.append_tool_calls("s1", [tc])

    loaded = await file_store.get_session("s1")
    assert len(loaded.session.tool_state.tool_calls) == 1
    assert loaded.session.tool_state.tool_calls[0].tool == "search"


@pytest.mark.asyncio
async def test_file_store_version_conflict(file_store):
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(session_id="s1"),
    )
    await file_store.put_session("s1", doc)

    with pytest.raises(VersionConflictError):
        await file_store.put_session("s1", doc, expected_version=999)
