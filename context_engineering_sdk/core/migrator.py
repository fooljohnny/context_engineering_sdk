"""Schema version migration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from context_engineering_sdk.core.types import SessionDocument


CURRENT_SCHEMA_VERSION = "1.0"


@dataclass
class MigrateResult:
    doc: SessionDocument
    from_version: str
    to_version: str
    changes: list[str] = field(default_factory=list)


class Migrator(Protocol):
    def current_version(self) -> str: ...
    def can_migrate(self, from_version: str) -> bool: ...
    def migrate(self, doc: dict, from_version: str) -> MigrateResult: ...


class DefaultMigrator:
    """Identity migrator for schema version 1.0 (no migrations yet)."""

    def current_version(self) -> str:
        return CURRENT_SCHEMA_VERSION

    def can_migrate(self, from_version: str) -> bool:
        return from_version == CURRENT_SCHEMA_VERSION

    def migrate(self, doc: dict, from_version: str) -> MigrateResult:
        from context_engineering_sdk.core.errors import MigrationError

        if from_version != CURRENT_SCHEMA_VERSION:
            raise MigrationError(
                f"Cannot migrate from {from_version} to {CURRENT_SCHEMA_VERSION}"
            )
        # Identity migration: build SessionDocument from dict
        from context_engineering_sdk.core.types import (
            Session,
            SessionDocument,
            TaskState,
            ToolState,
        )

        session_data = doc.get("session", {})
        session = Session(
            session_id=session_data.get("session_id", ""),
            messages=session_data.get("messages", []),
            task_state=session_data.get("task_state", TaskState()),
            tool_state=session_data.get("tool_state", ToolState()),
            model_usage=session_data.get("model_usage", []),
        )
        result_doc = SessionDocument(
            schema_version=CURRENT_SCHEMA_VERSION,
            session=session,
            evidences=doc.get("evidences", {}),
            context_blocks=doc.get("context_blocks", []),
        )
        return MigrateResult(
            doc=result_doc,
            from_version=from_version,
            to_version=CURRENT_SCHEMA_VERSION,
            changes=[],
        )
