"""Pruning strategies for context blocks within token budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from context_engineering_sdk.core.errors import BudgetExceededError
from context_engineering_sdk.core.types import (
    ContextBlock,
    Priority,
    PruneDecision,
    PRIORITY_RANK,
)


@dataclass
class PruneBudget:
    max_tokens: int
    tier_budgets: dict[Priority, int] | None = None


@dataclass
class PruneRules:
    must_keep_priorities: list[Priority] = field(
        default_factory=lambda: [Priority.MUST]
    )
    ttl_seconds: dict[Priority, int] | None = None
    now: str | None = None


@dataclass
class PruneResult:
    kept: list[ContextBlock]
    dropped: list[ContextBlock]
    decisions: list[PruneDecision]


class Pruner(Protocol):
    def prune(
        self,
        blocks: list[ContextBlock],
        budget: PruneBudget,
        rules: PruneRules | None = None,
    ) -> PruneResult: ...


class GreedyPriorityPruner:
    """MVP pruner: must-blocks always kept, rest sorted by priority then recency."""

    def prune(
        self,
        blocks: list[ContextBlock],
        budget: PruneBudget,
        rules: PruneRules | None = None,
    ) -> PruneResult:
        if not blocks:
            return PruneResult(kept=[], dropped=[], decisions=[])

        rules = rules or PruneRules()
        must_priorities = set(rules.must_keep_priorities)

        must_blocks = [b for b in blocks if b.priority in must_priorities]
        other_blocks = [b for b in blocks if b.priority not in must_priorities]

        # Check if must-blocks alone exceed budget
        must_tokens = sum(b.token_estimate for b in must_blocks)
        if must_tokens > budget.max_tokens:
            raise BudgetExceededError(
                f"Must-priority blocks require {must_tokens} tokens, "
                f"but budget is {budget.max_tokens}"
            )

        # Sort non-must blocks: lower priority rank first (higher priority),
        # then by original order (recency proxy - later = more recent)
        sorted_others = sorted(
            enumerate(other_blocks),
            key=lambda pair: (PRIORITY_RANK[pair[1].priority], -pair[0]),
        )

        kept: list[ContextBlock] = list(must_blocks)
        dropped: list[ContextBlock] = []
        decisions: list[PruneDecision] = []

        for b in must_blocks:
            decisions.append(
                PruneDecision(
                    block_id=b.block_id,
                    action="kept",
                    reason="must_priority",
                    token_estimate=b.token_estimate,
                )
            )

        remaining_budget = budget.max_tokens - must_tokens
        for _, block in sorted_others:
            if block.token_estimate <= remaining_budget:
                kept.append(block)
                remaining_budget -= block.token_estimate
                decisions.append(
                    PruneDecision(
                        block_id=block.block_id,
                        action="kept",
                        reason="within_budget",
                        token_estimate=block.token_estimate,
                    )
                )
            else:
                dropped.append(block)
                decisions.append(
                    PruneDecision(
                        block_id=block.block_id,
                        action="dropped",
                        reason="budget_exceeded",
                        token_estimate=block.token_estimate,
                    )
                )

        return PruneResult(kept=kept, dropped=dropped, decisions=decisions)
