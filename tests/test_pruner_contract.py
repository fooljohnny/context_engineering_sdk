"""Contract tests for Pruner implementations."""

from __future__ import annotations

import pytest

from context_engineering_sdk.builder.pruner import (
    GreedyPriorityPruner,
    PruneBudget,
    PruneRules,
)
from context_engineering_sdk.core.errors import BudgetExceededError
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Priority,
)


def _block(block_id: str, priority: Priority, tokens: int) -> ContextBlock:
    return ContextBlock(
        block_id=block_id,
        block_type=BlockType.EVIDENCE,
        priority=priority,
        token_estimate=tokens,
        content=f"content-{block_id}",
    )


@pytest.fixture
def pruner():
    return GreedyPriorityPruner()


def test_must_blocks_never_dropped(pruner):
    blocks = [
        _block("b1", Priority.MUST, 100),
        _block("b2", Priority.HIGH, 200),
        _block("b3", Priority.LOW, 300),
    ]
    result = pruner.prune(blocks, PruneBudget(max_tokens=250))
    kept_ids = {b.block_id for b in result.kept}
    assert "b1" in kept_ids


def test_budget_respected(pruner):
    blocks = [
        _block("b1", Priority.HIGH, 100),
        _block("b2", Priority.MEDIUM, 200),
        _block("b3", Priority.LOW, 300),
    ]
    result = pruner.prune(blocks, PruneBudget(max_tokens=250))
    total = sum(b.token_estimate for b in result.kept)
    assert total <= 250


def test_empty_blocks_returns_empty(pruner):
    result = pruner.prune([], PruneBudget(max_tokens=1000))
    assert result.kept == []
    assert result.dropped == []
    assert result.decisions == []


def test_decisions_cover_all_blocks(pruner):
    blocks = [
        _block("b1", Priority.HIGH, 100),
        _block("b2", Priority.MEDIUM, 200),
        _block("b3", Priority.LOW, 300),
    ]
    result = pruner.prune(blocks, PruneBudget(max_tokens=500))
    decision_ids = {d.block_id for d in result.decisions}
    block_ids = {b.block_id for b in blocks}
    assert decision_ids == block_ids


def test_must_exceeds_budget_raises(pruner):
    blocks = [
        _block("b1", Priority.MUST, 1000),
    ]
    with pytest.raises(BudgetExceededError):
        pruner.prune(blocks, PruneBudget(max_tokens=500))


def test_priority_ordering(pruner):
    blocks = [
        _block("low1", Priority.LOW, 100),
        _block("high1", Priority.HIGH, 100),
        _block("med1", Priority.MEDIUM, 100),
    ]
    result = pruner.prune(blocks, PruneBudget(max_tokens=200))
    kept_ids = [b.block_id for b in result.kept]
    assert "high1" in kept_ids
    assert len(result.kept) == 2  # only 200 tokens, so 2 blocks of 100 each


def test_all_fit_within_budget(pruner):
    blocks = [
        _block("b1", Priority.HIGH, 100),
        _block("b2", Priority.MEDIUM, 100),
        _block("b3", Priority.LOW, 100),
    ]
    result = pruner.prune(blocks, PruneBudget(max_tokens=1000))
    assert len(result.kept) == 3
    assert len(result.dropped) == 0
