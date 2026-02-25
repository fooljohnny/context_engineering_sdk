"""Evidence ingestor: dedup, redaction, confidence, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from context_engineering_sdk.core.errors import RedactionError
from context_engineering_sdk.core.hasher import Hasher, Sha256Hasher
from context_engineering_sdk.core.id_generator import IdGenerator, UuidV4Generator
from context_engineering_sdk.core.redactor import Redactor, RedactionRules, RegexRedactor
from context_engineering_sdk.core.types import (
    Evidence,
    EvidenceLinks,
    EvidenceSource,
    EvidenceType,
    RedactionRecord,
)
from context_engineering_sdk.store.base import Store


@dataclass
class IngestOptions:
    redact: bool = True
    dedup: bool = True
    confidence: float | None = None


class EvidenceIngestor(Protocol):
    async def ingest(
        self,
        session_id: str,
        content: str,
        source: EvidenceSource,
        evidence_type: EvidenceType,
        links: EvidenceLinks | None = None,
        metadata: dict | None = None,
        options: IngestOptions | None = None,
    ) -> tuple[Evidence, RedactionRecord | None]: ...


class DefaultEvidenceIngestor:
    """Default ingestor with dedup (hash-based) and regex redaction."""

    def __init__(
        self,
        store: Store,
        id_generator: IdGenerator | None = None,
        hasher: Hasher | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._store = store
        self._id_gen = id_generator or UuidV4Generator()
        self._hasher = hasher or Sha256Hasher()
        self._redactor = redactor or RegexRedactor()
        self._dedup_cache: dict[str, str] = {}  # content_hash -> evidence_id

    async def ingest(
        self,
        session_id: str,
        content: str,
        source: EvidenceSource,
        evidence_type: EvidenceType,
        links: EvidenceLinks | None = None,
        metadata: dict | None = None,
        options: IngestOptions | None = None,
    ) -> tuple[Evidence, RedactionRecord | None]:
        opts = options or IngestOptions()

        # Dedup check
        if opts.dedup:
            dedup_key = f"{source.uri}:{self._hasher.digest(content)}"
            existing_id = self._dedup_cache.get(dedup_key)
            if existing_id:
                existing = await self._store.get_evidence(session_id, existing_id)
                if existing is not None:
                    return existing, None

        # Redaction
        redaction_record: RedactionRecord | None = None
        processed_content = content
        if opts.redact:
            try:
                result = await self._redactor.redact(content)
                processed_content = result.redacted_text
                if result.has_sensitive:
                    redaction_record = RedactionRecord(
                        evidence_id="",  # will be filled below
                        rules_applied=result.applied_rules,
                        fields_redacted=len(result.applied_rules),
                    )
            except Exception:
                # Degradation: skip redaction on failure
                processed_content = content

        evidence_id = self._id_gen.generate()
        evidence = Evidence(
            evidence_id=evidence_id,
            type=evidence_type,
            source=source,
            content=processed_content,
            confidence=opts.confidence,
            metadata=metadata or {},
            links=links or EvidenceLinks(),
        )

        await self._store.put_evidence(session_id, evidence)

        if redaction_record:
            redaction_record.evidence_id = evidence_id

        # Cache for dedup
        if opts.dedup:
            dedup_key = f"{source.uri}:{self._hasher.digest(content)}"
            self._dedup_cache[dedup_key] = evidence_id

        return evidence, redaction_record
