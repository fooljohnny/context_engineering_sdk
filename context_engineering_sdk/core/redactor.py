"""Redaction (PII / sensitive data removal) utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RedactionRules:
    patterns: list[str] = field(default_factory=list)
    field_paths: list[str] = field(default_factory=list)
    dictionaries: list[str] = field(default_factory=list)


@dataclass
class RedactResult:
    redacted_text: str
    applied_rules: list[str]
    has_sensitive: bool


class Redactor(Protocol):
    async def redact(
        self, text: str, rules: RedactionRules | None = None
    ) -> RedactResult: ...


_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone_number"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
    (r"\b(?:\d{4}[- ]?){3}\d{4}\b", "credit_card"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "ip_address"),
]


class RegexRedactor:
    """Default redactor using regex pattern matching."""

    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns = list(_DEFAULT_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    async def redact(
        self, text: str, rules: RedactionRules | None = None
    ) -> RedactResult:
        applied: list[str] = []
        result = text
        patterns_to_use = list(self._patterns)

        if rules and rules.patterns:
            for i, p in enumerate(rules.patterns):
                patterns_to_use.append((p, f"custom_{i}"))

        for pattern, name in patterns_to_use:
            compiled = re.compile(pattern, re.IGNORECASE)
            if compiled.search(result):
                result = compiled.sub(f"[REDACTED:{name}]", result)
                applied.append(name)

        return RedactResult(
            redacted_text=result,
            applied_rules=applied,
            has_sensitive=len(applied) > 0,
        )
