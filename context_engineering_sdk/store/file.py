"""File-based Store implementation (local JSON files)."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from context_engineering_sdk.core.errors import (
    SessionNotFoundError,
    VersionConflictError,
)
from context_engineering_sdk.core.types import (
    Author,
    AuthorKind,
    BlockType,
    ContextBlock,
    Evidence,
    EvidenceLinks,
    EvidenceSource,
    EvidenceType,
    Message,
    MessageIndexRange,
    ModelUsage,
    ModelUsageStage,
    ModelUsageStatus,
    Priority,
    PutResult,
    Ref,
    Role,
    Session,
    SessionDocument,
    SourceKind,
    Summary,
    Task,
    TaskState,
    TaskStatus,
    TodoList,
    ToolCall,
    ToolCallStatus,
    ToolProvider,
    ToolState,
    ToolType,
    ProviderKind,
    PRIORITY_RANK,
)
from context_engineering_sdk.store.base import BlockFilter, EvidenceFilter


def _serialize(obj) -> dict | list | str | int | float | bool | None:
    """Recursively serialize dataclass/enum objects to JSON-compatible dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _deserialize_message(data: dict) -> Message:
    author = None
    if data.get("author"):
        author = Author(kind=AuthorKind(data["author"]["kind"]), id=data["author"]["id"])
    refs = None
    if data.get("refs"):
        refs = [Ref(evidence_id=r["evidence_id"], selector=r.get("selector")) for r in data["refs"]]
    return Message(
        role=Role(data["role"]),
        content=data["content"],
        author=author,
        at=data.get("at"),
        refs=refs,
    )


def _deserialize_evidence(data: dict) -> Evidence:
    return Evidence(
        evidence_id=data["evidence_id"],
        type=EvidenceType(data["type"]),
        source=EvidenceSource(
            kind=SourceKind(data["source"]["kind"]),
            name=data["source"].get("name", ""),
            uri=data["source"].get("uri", ""),
        ),
        content=data.get("content", ""),
        confidence=data.get("confidence"),
        metadata=data.get("metadata", {}),
        links=EvidenceLinks(
            model_usage_id=data.get("links", {}).get("model_usage_id"),
            tool_call_id=data.get("links", {}).get("tool_call_id"),
        ),
    )


def _deserialize_block(data: dict) -> ContextBlock:
    refs = [Ref(evidence_id=r["evidence_id"], selector=r.get("selector")) for r in data.get("refs", [])]
    return ContextBlock(
        block_id=data["block_id"],
        block_type=BlockType(data["block_type"]),
        priority=Priority(data["priority"]),
        token_estimate=data.get("token_estimate", 0),
        content=data.get("content", ""),
        refs=refs,
    )


def _deserialize_tool_call(data: dict) -> ToolCall:
    return ToolCall(
        tool_call_id=data["tool_call_id"],
        tool=data["tool"],
        provider=ToolProvider(
            kind=ProviderKind(data["provider"]["kind"]),
            name=data["provider"].get("name", ""),
            uri=data["provider"].get("uri", ""),
        ),
        type=ToolType(data.get("type", "tool")),
        called_at=data.get("called_at", ""),
        args_digest=data.get("args_digest", {}),
        status=ToolCallStatus(data.get("status", "success")),
        duration_ms=data.get("duration_ms", 0),
        result_evidence_ids=data.get("result_evidence_ids", []),
        task_id=data.get("task_id"),
    )


def _deserialize_model_usage(data: dict) -> ModelUsage:
    return ModelUsage(
        model_usage_id=data["model_usage_id"],
        provider=data["provider"],
        model=data["model"],
        stage=ModelUsageStage(data.get("stage", "other")),
        params=data.get("params", {}),
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
        first_token_latency_ms=data.get("first_token_latency_ms"),
        latency_ms=data.get("latency_ms", 0),
        status=ModelUsageStatus(data.get("status", "success")),
        error=data.get("error", ""),
        task_id=data.get("task_id"),
    )


def _deserialize_task(data: dict) -> Task:
    return Task(
        task_id=data["task_id"],
        name=data["name"],
        status=TaskStatus(data.get("status", "pending")),
        depends_on=data.get("depends_on", []),
        result_evidence_ids=data.get("result_evidence_ids", []),
        error=data.get("error", ""),
    )


def _deserialize_doc(data: dict) -> SessionDocument:
    session_data = data.get("session", {})
    summary = None
    if session_data.get("summary") and session_data["summary"].get("content"):
        s = session_data["summary"]
        mir = s.get("message_index_range", {})
        summary = Summary(
            content=s["content"],
            updated_at=s.get("updated_at", ""),
            message_index_range=MessageIndexRange(
                from_index=mir.get("from_index", 0),
                to_index=mir.get("to_index", 0),
            ),
        )

    tasks_data = session_data.get("task_state", {}).get("todo_list", {}).get("tasks", [])
    tool_calls_data = session_data.get("tool_state", {}).get("tool_calls", [])
    model_usage_data = session_data.get("model_usage", [])

    session = Session(
        session_id=session_data.get("session_id", ""),
        messages=[_deserialize_message(m) for m in session_data.get("messages", [])],
        summary=summary,
        task_state=TaskState(
            todo_list=TodoList(
                tasks=[_deserialize_task(t) for t in tasks_data]
            )
        ),
        tool_state=ToolState(
            tool_calls=[_deserialize_tool_call(tc) for tc in tool_calls_data]
        ),
        model_usage=[_deserialize_model_usage(mu) for mu in model_usage_data],
    )

    evidences = {}
    for eid, ev_data in data.get("evidences", {}).items():
        evidences[eid] = _deserialize_evidence(ev_data)

    blocks = [_deserialize_block(b) for b in data.get("context_blocks", [])]

    return SessionDocument(
        schema_version=data.get("schema_version", "1.0"),
        session=session,
        evidences=evidences,
        context_blocks=blocks,
    )


class FileStore:
    """File-based Store that persists sessions as JSON files."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, int] = {}

    def _session_path(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.json"

    def _read(self, session_id: str) -> SessionDocument | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_doc(data)

    def _write(self, session_id: str, doc: SessionDocument) -> None:
        path = self._session_path(session_id)
        data = _serialize(doc)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_version(self, session_id: str) -> int:
        return self._versions.get(session_id, 0)

    def _check_version(self, session_id: str, expected: int | None) -> None:
        if expected is not None and self._get_version(session_id) != expected:
            raise VersionConflictError(
                f"Expected version {expected}, got {self._get_version(session_id)}"
            )

    def _bump_version(self, session_id: str) -> int:
        v = self._versions.get(session_id, 0) + 1
        self._versions[session_id] = v
        return v

    async def get_session(self, session_id: str) -> SessionDocument | None:
        return self._read(session_id)

    async def put_session(
        self,
        session_id: str,
        doc: SessionDocument,
        expected_version: int | None = None,
    ) -> PutResult:
        if self._session_path(session_id).exists():
            self._check_version(session_id, expected_version)
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)

    async def append_messages(
        self, session_id: str, messages: list[Message]
    ) -> PutResult:
        doc = self._read(session_id)
        if doc is None:
            doc = SessionDocument(
                schema_version="1.0",
                session=Session(session_id=session_id, messages=list(messages)),
            )
        else:
            doc.session.messages.extend(messages)
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)

    async def patch_session(
        self,
        session_id: str,
        patch: dict,
        expected_version: int | None = None,
    ) -> PutResult:
        doc = self._read(session_id)
        if doc is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        self._check_version(session_id, expected_version)
        if "summary" in patch:
            doc.session.summary = patch["summary"]
        if "task_state" in patch:
            doc.session.task_state = patch["task_state"]
        if "tool_state" in patch:
            doc.session.tool_state = patch["tool_state"]
        if "model_usage" in patch:
            doc.session.model_usage = patch["model_usage"]
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)

    async def put_evidence(
        self, session_id: str, evidence: Evidence
    ) -> str:
        doc = self._read(session_id)
        if doc is None:
            doc = SessionDocument(
                schema_version="1.0",
                session=Session(session_id=session_id),
                evidences={evidence.evidence_id: evidence},
            )
        else:
            doc.evidences[evidence.evidence_id] = evidence
        self._write(session_id, doc)
        self._bump_version(session_id)
        return evidence.evidence_id

    async def get_evidence(
        self, session_id: str, evidence_id: str
    ) -> Evidence | None:
        doc = self._read(session_id)
        if doc is None:
            return None
        return doc.evidences.get(evidence_id)

    async def list_evidences(
        self, session_id: str, filter: EvidenceFilter | None = None
    ) -> list[Evidence]:
        doc = self._read(session_id)
        if doc is None:
            return []
        results = list(doc.evidences.values())
        if filter:
            if filter.types:
                results = [e for e in results if e.type in filter.types]
            if filter.source_kinds:
                results = [e for e in results if e.source.kind in filter.source_kinds]
            if filter.limit:
                results = results[: filter.limit]
        return results

    async def put_context_blocks(
        self, session_id: str, blocks: list[ContextBlock]
    ) -> PutResult:
        doc = self._read(session_id)
        if doc is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        existing_ids = {b.block_id for b in doc.context_blocks}
        for block in blocks:
            if block.block_id in existing_ids:
                doc.context_blocks = [
                    block if b.block_id == block.block_id else b
                    for b in doc.context_blocks
                ]
            else:
                doc.context_blocks.append(block)
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)

    async def list_context_blocks(
        self, session_id: str, filter: BlockFilter | None = None
    ) -> list[ContextBlock]:
        doc = self._read(session_id)
        if doc is None:
            return []
        results = list(doc.context_blocks)
        if filter:
            if filter.block_types:
                results = [b for b in results if b.block_type in filter.block_types]
            if filter.priorities:
                results = [b for b in results if b.priority in filter.priorities]
            if filter.min_priority:
                threshold = PRIORITY_RANK[filter.min_priority]
                results = [b for b in results if PRIORITY_RANK[b.priority] <= threshold]
            if filter.limit:
                results = results[: filter.limit]
        return results

    async def append_tool_calls(
        self,
        session_id: str,
        tool_calls: list[ToolCall],
        expected_version: int | None = None,
    ) -> PutResult:
        doc = self._read(session_id)
        if doc is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        self._check_version(session_id, expected_version)
        doc.session.tool_state.tool_calls.extend(tool_calls)
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)

    async def append_model_usage(
        self,
        session_id: str,
        usages: list[ModelUsage],
        expected_version: int | None = None,
    ) -> PutResult:
        doc = self._read(session_id)
        if doc is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        self._check_version(session_id, expected_version)
        doc.session.model_usage.extend(usages)
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)

    async def upsert_tasks(
        self,
        session_id: str,
        tasks: list[Task],
        expected_version: int | None = None,
    ) -> PutResult:
        doc = self._read(session_id)
        if doc is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        self._check_version(session_id, expected_version)
        existing = {
            t.task_id: i
            for i, t in enumerate(doc.session.task_state.todo_list.tasks)
        }
        for task in tasks:
            if task.task_id in existing:
                idx = existing[task.task_id]
                doc.session.task_state.todo_list.tasks[idx] = task
            else:
                doc.session.task_state.todo_list.tasks.append(task)
        self._write(session_id, doc)
        v = self._bump_version(session_id)
        return PutResult(success=True, version=v)
