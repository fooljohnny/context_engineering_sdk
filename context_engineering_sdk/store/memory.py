"""In-memory Store implementation for testing and prototyping."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from context_engineering_sdk.core.errors import (
    SessionNotFoundError,
    VersionConflictError,
)
from context_engineering_sdk.core.types import (
    ContextBlock,
    Evidence,
    Message,
    ModelUsage,
    Priority,
    PutResult,
    Session,
    SessionDocument,
    Task,
    TaskState,
    ToolCall,
    ToolState,
    PRIORITY_RANK,
)
from context_engineering_sdk.store.base import BlockFilter, EvidenceFilter


@dataclass
class _SessionEntry:
    doc: SessionDocument
    version: int = 1


class MemoryStore:
    """Thread-unsafe, in-memory Store for testing and prototyping."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionEntry] = {}

    def _get_entry(self, session_id: str) -> _SessionEntry:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        return entry

    def _check_version(
        self, entry: _SessionEntry, expected_version: int | None
    ) -> None:
        if expected_version is not None and entry.version != expected_version:
            raise VersionConflictError(
                f"Expected version {expected_version}, got {entry.version}"
            )

    def _bump(self, entry: _SessionEntry) -> PutResult:
        entry.version += 1
        return PutResult(success=True, version=entry.version)

    # --- Session ---

    async def get_session(self, session_id: str) -> SessionDocument | None:
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        return copy.deepcopy(entry.doc)

    async def put_session(
        self,
        session_id: str,
        doc: SessionDocument,
        expected_version: int | None = None,
    ) -> PutResult:
        entry = self._sessions.get(session_id)
        if entry is None:
            self._sessions[session_id] = _SessionEntry(
                doc=copy.deepcopy(doc), version=1
            )
            return PutResult(success=True, version=1)
        self._check_version(entry, expected_version)
        entry.doc = copy.deepcopy(doc)
        return self._bump(entry)

    async def append_messages(
        self, session_id: str, messages: list[Message]
    ) -> PutResult:
        entry = self._sessions.get(session_id)
        if entry is None:
            doc = SessionDocument(
                schema_version="1.0",
                session=Session(session_id=session_id, messages=list(messages)),
            )
            self._sessions[session_id] = _SessionEntry(doc=doc, version=1)
            return PutResult(success=True, version=1)
        entry.doc.session.messages.extend(messages)
        return self._bump(entry)

    async def patch_session(
        self,
        session_id: str,
        patch: dict,
        expected_version: int | None = None,
    ) -> PutResult:
        entry = self._get_entry(session_id)
        self._check_version(entry, expected_version)
        session = entry.doc.session
        if "summary" in patch:
            session.summary = patch["summary"]
        if "task_state" in patch:
            session.task_state = patch["task_state"]
        if "tool_state" in patch:
            session.tool_state = patch["tool_state"]
        if "model_usage" in patch:
            session.model_usage = patch["model_usage"]
        return self._bump(entry)

    # --- Evidence ---

    async def put_evidence(
        self, session_id: str, evidence: Evidence
    ) -> str:
        entry = self._sessions.get(session_id)
        if entry is None:
            doc = SessionDocument(
                schema_version="1.0",
                session=Session(session_id=session_id),
                evidences={evidence.evidence_id: evidence},
            )
            self._sessions[session_id] = _SessionEntry(doc=doc, version=1)
            return evidence.evidence_id
        entry.doc.evidences[evidence.evidence_id] = evidence
        entry.version += 1
        return evidence.evidence_id

    async def get_evidence(
        self, session_id: str, evidence_id: str
    ) -> Evidence | None:
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        return entry.doc.evidences.get(evidence_id)

    async def list_evidences(
        self, session_id: str, filter: EvidenceFilter | None = None
    ) -> list[Evidence]:
        entry = self._sessions.get(session_id)
        if entry is None:
            return []
        results = list(entry.doc.evidences.values())
        if filter:
            if filter.types:
                results = [e for e in results if e.type in filter.types]
            if filter.source_kinds:
                results = [e for e in results if e.source.kind in filter.source_kinds]
            if filter.limit:
                results = results[: filter.limit]
        return results

    # --- Context Blocks ---

    async def put_context_blocks(
        self, session_id: str, blocks: list[ContextBlock]
    ) -> PutResult:
        entry = self._get_entry(session_id)
        existing_ids = {b.block_id for b in entry.doc.context_blocks}
        for block in blocks:
            if block.block_id in existing_ids:
                entry.doc.context_blocks = [
                    block if b.block_id == block.block_id else b
                    for b in entry.doc.context_blocks
                ]
            else:
                entry.doc.context_blocks.append(block)
        return self._bump(entry)

    async def list_context_blocks(
        self, session_id: str, filter: BlockFilter | None = None
    ) -> list[ContextBlock]:
        entry = self._sessions.get(session_id)
        if entry is None:
            return []
        results = list(entry.doc.context_blocks)
        if filter:
            if filter.block_types:
                results = [b for b in results if b.block_type in filter.block_types]
            if filter.priorities:
                results = [b for b in results if b.priority in filter.priorities]
            if filter.min_priority:
                threshold = PRIORITY_RANK[filter.min_priority]
                results = [
                    b for b in results if PRIORITY_RANK[b.priority] <= threshold
                ]
            if filter.limit:
                results = results[: filter.limit]
        return results

    # --- Tool / Model / Task ---

    async def append_tool_calls(
        self,
        session_id: str,
        tool_calls: list[ToolCall],
        expected_version: int | None = None,
    ) -> PutResult:
        entry = self._get_entry(session_id)
        self._check_version(entry, expected_version)
        entry.doc.session.tool_state.tool_calls.extend(tool_calls)
        return self._bump(entry)

    async def append_model_usage(
        self,
        session_id: str,
        usages: list[ModelUsage],
        expected_version: int | None = None,
    ) -> PutResult:
        entry = self._get_entry(session_id)
        self._check_version(entry, expected_version)
        entry.doc.session.model_usage.extend(usages)
        return self._bump(entry)

    async def upsert_tasks(
        self,
        session_id: str,
        tasks: list[Task],
        expected_version: int | None = None,
    ) -> PutResult:
        entry = self._get_entry(session_id)
        self._check_version(entry, expected_version)
        existing = {
            t.task_id: i
            for i, t in enumerate(
                entry.doc.session.task_state.todo_list.tasks
            )
        }
        for task in tasks:
            if task.task_id in existing:
                idx = existing[task.task_id]
                entry.doc.session.task_state.todo_list.tasks[idx] = task
            else:
                entry.doc.session.task_state.todo_list.tasks.append(task)
        return self._bump(entry)
