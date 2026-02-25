"""Exception hierarchy for the Context Engineering SDK."""


class ContextEngineError(Exception):
    """SDK base exception."""


class SessionNotFoundError(ContextEngineError):
    """Session does not exist."""


class VersionConflictError(ContextEngineError):
    """Optimistic-lock version mismatch on Store write."""


class BudgetExceededError(ContextEngineError):
    """Must-priority blocks already exceed the token budget."""


class SelectorResolveError(ContextEngineError):
    """Selector parsing or extraction failed."""


class EvidenceNotFoundError(ContextEngineError):
    """Referenced evidence_id not found in Store."""


class SchemaValidationError(ContextEngineError):
    """Data violates schema constraints."""


class MigrationError(ContextEngineError):
    """Schema version migration failed."""


class RedactionError(ContextEngineError):
    """Redaction processing failed."""


class SummarizerError(ContextEngineError):
    """Summary generation failed (LLM call timeout / error)."""
