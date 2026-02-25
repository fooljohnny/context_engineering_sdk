"""Assembler: combines messages and rendered blocks into final model input."""

from __future__ import annotations

from typing import Protocol

from context_engineering_sdk.core.token_estimator import CharBasedEstimator, TokenEstimator
from context_engineering_sdk.core.types import (
    AssembledInput,
    BlockType,
    Message,
    MessagePart,
    RenderedBlock,
    Role,
)


class Assembler(Protocol):
    def assemble(
        self,
        messages: list[Message],
        rendered_blocks: list[RenderedBlock],
        model_hint: str | None = None,
    ) -> AssembledInput: ...


class DefaultAssembler:
    """Assembles messages + rendered context blocks into AssembledInput."""

    def __init__(self, token_estimator: TokenEstimator | None = None) -> None:
        self._tok = token_estimator or CharBasedEstimator()

    def assemble(
        self,
        messages: list[Message],
        rendered_blocks: list[RenderedBlock],
        model_hint: str | None = None,
    ) -> AssembledInput:
        parts: list[MessagePart] = []

        # Group blocks by type for structured injection
        instruction_blocks = [
            b for b in rendered_blocks if b.block_type == BlockType.INSTRUCTION
        ]
        state_blocks = [
            b
            for b in rendered_blocks
            if b.block_type in (BlockType.STATE, BlockType.PLAN)
        ]
        evidence_blocks = [
            b for b in rendered_blocks if b.block_type == BlockType.EVIDENCE
        ]
        memory_blocks = [
            b for b in rendered_blocks if b.block_type == BlockType.MEMORY
        ]
        # conversation blocks from rendered_blocks are handled via messages

        # System-level context: instructions + state + evidence + memory
        system_sections: list[str] = []

        if instruction_blocks:
            system_sections.append(
                "\n\n".join(b.rendered_content for b in instruction_blocks)
            )

        if state_blocks:
            system_sections.append(
                "\n\n".join(b.rendered_content for b in state_blocks)
            )

        if evidence_blocks:
            evidence_text = "\n\n".join(
                f"[Evidence] {b.rendered_content}" for b in evidence_blocks
            )
            system_sections.append(evidence_text)

        if memory_blocks:
            memory_text = "\n\n".join(
                f"[Memory] {b.rendered_content}" for b in memory_blocks
            )
            system_sections.append(memory_text)

        if system_sections:
            system_content = "\n\n---\n\n".join(system_sections)
            parts.append(MessagePart(role=Role.SYSTEM, content=system_content))

        # Add conversation messages
        for msg in messages:
            parts.append(MessagePart(role=msg.role, content=msg.content))

        # Build text representation
        text_lines: list[str] = []
        for part in parts:
            text_lines.append(f"<|{part.role.value}|>\n{part.content}")
        text = "\n\n".join(text_lines)

        total_tokens = self._tok.estimate_text(text, model_hint)

        return AssembledInput(
            parts=parts,
            text=text,
            total_tokens=total_tokens,
        )
