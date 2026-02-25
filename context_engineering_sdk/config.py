"""Runtime configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field

from context_engineering_sdk.core.redactor import RedactionRules
from context_engineering_sdk.core.types import Priority


@dataclass
class BudgetConfig:
    max_input_tokens: int = 8192
    reserved_reply_tokens: int = 1024


@dataclass
class PruneConfig:
    must_keep_priorities: list[Priority] = field(
        default_factory=lambda: [Priority.MUST]
    )
    ttl_seconds: dict[str, int] | None = None


@dataclass
class RedactionConfig:
    enabled: bool = True
    mode: str = "store_redacted_only"  # "store_redacted_only" | "dual_store"
    rules: RedactionRules | None = None


@dataclass
class ObservabilityConfig:
    emit_events: bool = True
    include_reports_in_store: bool = True


@dataclass
class SummaryConfig:
    enabled: bool = True
    max_summary_tokens: int = 512
    trigger_message_count: int = 20
    trigger_token_threshold: int = 4096
    preserve_recent_messages: int = 5


@dataclass
class RuntimeConfig:
    model_hint: str | None = None
    provider_hint: str | None = None
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
