"""Tests for builder components: Ingestor, Deriver, Renderer, Assembler."""

from __future__ import annotations

import pytest

from context_engineering_sdk.builder.assembler import DefaultAssembler
from context_engineering_sdk.builder.deriver import DefaultBlockDeriver, DeriveOptions
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.renderer import DefaultEvidenceResolver, DefaultRenderer
from context_engineering_sdk.core.id_generator import UuidV4Generator
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Evidence,
    EvidenceLinks,
    EvidenceSource,
    EvidenceType,
    Message,
    Priority,
    Ref,
    RenderedBlock,
    Role,
    Session,
    SessionDocument,
    SourceKind,
    Task,
    TaskState,
    TaskStatus,
    TodoList,
)
from context_engineering_sdk.store.memory import MemoryStore


# ---- Ingestor ----

@pytest.mark.asyncio
async def test_ingestor_basic():
    store = MemoryStore()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    await store.put_session("s1", doc)

    ingestor = DefaultEvidenceIngestor(store=store)
    evidence, redaction = await ingestor.ingest(
        session_id="s1",
        content="Order #12345 shipped",
        source=EvidenceSource(kind=SourceKind.TOOL, name="order-svc"),
        evidence_type=EvidenceType.TOOL_RESULT,
    )
    assert evidence.evidence_id
    assert evidence.content == "Order #12345 shipped"

    loaded = await store.get_evidence("s1", evidence.evidence_id)
    assert loaded is not None


@pytest.mark.asyncio
async def test_ingestor_dedup():
    store = MemoryStore()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    await store.put_session("s1", doc)

    ingestor = DefaultEvidenceIngestor(store=store)
    source = EvidenceSource(kind=SourceKind.TOOL, name="tool1", uri="http://tool1")
    ev1, _ = await ingestor.ingest(
        session_id="s1", content="same content",
        source=source, evidence_type=EvidenceType.TOOL_RESULT,
    )
    ev2, _ = await ingestor.ingest(
        session_id="s1", content="same content",
        source=source, evidence_type=EvidenceType.TOOL_RESULT,
    )
    assert ev1.evidence_id == ev2.evidence_id


@pytest.mark.asyncio
async def test_ingestor_no_dedup():
    store = MemoryStore()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    await store.put_session("s1", doc)

    ingestor = DefaultEvidenceIngestor(store=store)
    source = EvidenceSource(kind=SourceKind.TOOL, name="tool1")
    ev1, _ = await ingestor.ingest(
        session_id="s1", content="same content",
        source=source, evidence_type=EvidenceType.TOOL_RESULT,
        options=IngestOptions(dedup=False),
    )
    ev2, _ = await ingestor.ingest(
        session_id="s1", content="same content",
        source=source, evidence_type=EvidenceType.TOOL_RESULT,
        options=IngestOptions(dedup=False),
    )
    assert ev1.evidence_id != ev2.evidence_id


@pytest.mark.asyncio
async def test_ingestor_redaction():
    store = MemoryStore()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    await store.put_session("s1", doc)

    ingestor = DefaultEvidenceIngestor(store=store)
    evidence, redaction = await ingestor.ingest(
        session_id="s1",
        content="Contact: john@example.com",
        source=EvidenceSource(kind=SourceKind.USER),
        evidence_type=EvidenceType.USER_INPUT,
    )
    assert "[REDACTED:email]" in evidence.content
    assert redaction is not None
    assert redaction.fields_redacted > 0


@pytest.mark.asyncio
async def test_ingestor_with_links():
    store = MemoryStore()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    await store.put_session("s1", doc)

    ingestor = DefaultEvidenceIngestor(store=store)
    links = EvidenceLinks(tool_call_id="tc1")
    evidence, _ = await ingestor.ingest(
        session_id="s1",
        content="result data",
        source=EvidenceSource(kind=SourceKind.TOOL, name="tool1"),
        evidence_type=EvidenceType.TOOL_RESULT,
        links=links,
    )
    assert evidence.links.tool_call_id == "tc1"


# ---- Deriver ----

@pytest.mark.asyncio
async def test_deriver_conversation_blocks():
    deriver = DefaultBlockDeriver()
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(
            session_id="s1",
            messages=[
                Message(role=Role.SYSTEM, content="You are a helper."),
                Message(role=Role.USER, content="What is 2+2?"),
                Message(role=Role.ASSISTANT, content="4"),
            ],
        ),
    )
    blocks = await deriver.derive(doc, [])
    conversation_blocks = [b for b in blocks if b.block_type == BlockType.CONVERSATION]
    assert len(conversation_blocks) == 3
    # System message should be MUST priority
    system_blocks = [b for b in conversation_blocks if b.priority == Priority.MUST]
    assert len(system_blocks) == 1


@pytest.mark.asyncio
async def test_deriver_state_blocks():
    deriver = DefaultBlockDeriver()
    doc = SessionDocument(
        schema_version="1.0",
        session=Session(
            session_id="s1",
            task_state=TaskState(
                todo_list=TodoList(
                    tasks=[
                        Task(task_id="t1", name="Research", status=TaskStatus.IN_PROGRESS),
                    ]
                )
            ),
        ),
    )
    blocks = await deriver.derive(doc, [], DeriveOptions(include_conversation=False))
    state_blocks = [b for b in blocks if b.block_type == BlockType.STATE]
    assert len(state_blocks) == 1
    assert "Research" in state_blocks[0].content


@pytest.mark.asyncio
async def test_deriver_evidence_blocks():
    deriver = DefaultBlockDeriver()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    evidences = [
        Evidence(
            evidence_id="ev1",
            type=EvidenceType.TOOL_RESULT,
            source=EvidenceSource(kind=SourceKind.TOOL),
            content="Some tool output",
            confidence=0.95,
        ),
    ]
    blocks = await deriver.derive(
        doc, evidences,
        DeriveOptions(include_conversation=False, include_state=False),
    )
    evidence_blocks = [b for b in blocks if b.block_type == BlockType.EVIDENCE]
    assert len(evidence_blocks) == 1
    assert evidence_blocks[0].refs[0].evidence_id == "ev1"


@pytest.mark.asyncio
async def test_deriver_custom_instructions():
    deriver = DefaultBlockDeriver()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    custom = ContextBlock(
        block_id="custom1",
        block_type=BlockType.INSTRUCTION,
        priority=Priority.MUST,
        content="Always respond in JSON format.",
    )
    blocks = await deriver.derive(
        doc, [],
        DeriveOptions(
            include_conversation=False,
            include_state=False,
            include_evidences=False,
            custom_instructions=[custom],
        ),
    )
    assert len(blocks) == 1
    assert blocks[0].block_id == "custom1"


# ---- Renderer ----

@pytest.mark.asyncio
async def test_renderer_plain_content():
    renderer = DefaultRenderer()
    block = ContextBlock(
        block_id="b1",
        block_type=BlockType.EVIDENCE,
        priority=Priority.HIGH,
        token_estimate=50,
        content="Plain content here",
    )

    class NoopResolver:
        async def resolve(self, evidence_id, selector=None):
            return ""

    rendered = await renderer.render_block(block, NoopResolver())
    assert rendered.rendered_content == "Plain content here"


@pytest.mark.asyncio
async def test_renderer_with_refs():
    store = MemoryStore()
    doc = SessionDocument(
        schema_version="1.0", session=Session(session_id="s1")
    )
    await store.put_session("s1", doc)
    await store.put_evidence(
        "s1",
        Evidence(
            evidence_id="ev1",
            type=EvidenceType.TOOL_RESULT,
            source=EvidenceSource(kind=SourceKind.TOOL),
            content="Resolved evidence content",
        ),
    )

    resolver = DefaultEvidenceResolver(store=store, session_id="s1")
    renderer = DefaultRenderer()

    block = ContextBlock(
        block_id="b1",
        block_type=BlockType.EVIDENCE,
        priority=Priority.HIGH,
        token_estimate=50,
        content="fallback",
        refs=[Ref(evidence_id="ev1")],
    )
    rendered = await renderer.render_block(block, resolver)
    assert rendered.rendered_content == "Resolved evidence content"


# ---- Assembler ----

def test_assembler_basic():
    assembler = DefaultAssembler()
    messages = [
        Message(role=Role.USER, content="Hello"),
    ]
    rendered = [
        RenderedBlock(
            block_id="b1",
            block_type=BlockType.INSTRUCTION,
            priority=Priority.MUST,
            rendered_content="Be helpful.",
            token_estimate=10,
        ),
    ]
    result = assembler.assemble(messages, rendered)
    assert len(result.parts) >= 2  # system (from blocks) + user message
    assert result.total_tokens > 0
    assert result.text is not None


def test_assembler_empty():
    assembler = DefaultAssembler()
    result = assembler.assemble([], [])
    assert len(result.parts) == 0
    assert result.total_tokens == 0


def test_assembler_evidence_blocks():
    assembler = DefaultAssembler()
    messages = [Message(role=Role.USER, content="What is X?")]
    rendered = [
        RenderedBlock(
            block_id="b1",
            block_type=BlockType.EVIDENCE,
            priority=Priority.HIGH,
            rendered_content="X is a framework.",
            token_estimate=20,
        ),
    ]
    result = assembler.assemble(messages, rendered)
    # System part should contain evidence
    system_parts = [p for p in result.parts if p.role == Role.SYSTEM]
    assert len(system_parts) == 1
    assert "[Evidence]" in system_parts[0].content
