"""Store protocol and query filter types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Evidence,
    EvidenceType,
    Message,
    ModelUsage,
    Priority,
    PutResult,
    SessionDocument,
    SourceKind,
    Task,
    ToolCall,
)


@dataclass
class EvidenceFilter:
    types: list[EvidenceType] | None = None
    source_kinds: list[SourceKind] | None = None
    since: str | None = None
    until: str | None = None
    tags: dict[str, str] | None = None
    limit: int | None = None


@dataclass
class BlockFilter:
    block_types: list[BlockType] | None = None
    priorities: list[Priority] | None = None
    min_priority: Priority | None = None
    limit: int | None = None


class Store(Protocol):
    # --- Session ---
    async def get_session(self, session_id: str) -> SessionDocument | None: ...

    async def put_session(
        self,
        session_id: str,
        doc: SessionDocument,
        expected_version: int | None = None,
    ) -> PutResult: ...

    async def append_messages(
        self, session_id: str, messages: list[Message]
    ) -> PutResult: ...

    async def patch_session(
        self,
        session_id: str,
        patch: dict,
        expected_version: int | None = None,
    ) -> PutResult: ...

    # --- Evidence ---
    async def put_evidence(
        self, session_id: str, evidence: Evidence
    ) -> str: ...

    async def get_evidence(
        self, session_id: str, evidence_id: str
    ) -> Evidence | None: ...

    async def list_evidences(
        self, session_id: str, filter: EvidenceFilter | None = None
    ) -> list[Evidence]: ...

    # --- Context Blocks ---
    async def put_context_blocks(
        self, session_id: str, blocks: list[ContextBlock]
    ) -> PutResult: ...

    async def list_context_blocks(
        self, session_id: str, filter: BlockFilter | None = None
    ) -> list[ContextBlock]: ...

    # --- Tool / Model / Task ---
    async def append_tool_calls(
        self,
        session_id: str,
        tool_calls: list[ToolCall],
        expected_version: int | None = None,
    ) -> PutResult: ...

    async def append_model_usage(
        self,
        session_id: str,
        usages: list[ModelUsage],
        expected_version: int | None = None,
    ) -> PutResult: ...

    async def upsert_tasks(
        self,
        session_id: str,
        tasks: list[Task],
        expected_version: int | None = None,
    ) -> PutResult: ...
