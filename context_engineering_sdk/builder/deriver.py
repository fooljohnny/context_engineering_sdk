"""Block deriver: derives context blocks from session and evidences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from context_engineering_sdk.core.id_generator import IdGenerator, UuidV4Generator
from context_engineering_sdk.core.token_estimator import CharBasedEstimator, TokenEstimator
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Evidence,
    Priority,
    Ref,
    SessionDocument,
)


@dataclass
class DeriveOptions:
    include_conversation: bool = True
    include_state: bool = True
    include_evidences: bool = True
    custom_instructions: list[ContextBlock] | None = None


class BlockDeriver(Protocol):
    async def derive(
        self,
        session_doc: SessionDocument,
        evidences: list[Evidence],
        options: DeriveOptions | None = None,
    ) -> list[ContextBlock]: ...


class DefaultBlockDeriver:
    """Derives context blocks from session messages, task state, and evidences."""

    def __init__(
        self,
        token_estimator: TokenEstimator | None = None,
        id_generator: IdGenerator | None = None,
    ) -> None:
        self._tok = token_estimator or CharBasedEstimator()
        self._id_gen = id_generator or UuidV4Generator()

    async def derive(
        self,
        session_doc: SessionDocument,
        evidences: list[Evidence],
        options: DeriveOptions | None = None,
    ) -> list[ContextBlock]:
        opts = options or DeriveOptions()
        blocks: list[ContextBlock] = []

        # Custom instruction blocks (injected, highest priority)
        if opts.custom_instructions:
            blocks.extend(opts.custom_instructions)

        session = session_doc.session

        if opts.include_conversation:
            blocks.extend(self._derive_conversation_blocks(session_doc))

        if opts.include_state:
            blocks.extend(self._derive_state_blocks(session_doc))

        if opts.include_evidences:
            blocks.extend(self._derive_evidence_blocks(evidences))

        return blocks

    def _derive_conversation_blocks(
        self, doc: SessionDocument
    ) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        session = doc.session

        # If summary exists, create a high-priority summary block
        if session.summary and session.summary.content:
            blocks.append(
                ContextBlock(
                    block_id=self._id_gen.generate(),
                    block_type=BlockType.CONVERSATION,
                    priority=Priority.HIGH,
                    token_estimate=self._tok.estimate_text(session.summary.content),
                    content=session.summary.content,
                )
            )
            # Only include messages after summary range
            summary_end = session.summary.message_index_range.to_index
            messages = session.messages[summary_end:]
        else:
            messages = session.messages

        total = len(messages)
        for i, msg in enumerate(messages):
            recency = total - i
            if msg.role.value == "system":
                priority = Priority.MUST
            elif recency <= 3:
                priority = Priority.HIGH
            elif recency <= 10:
                priority = Priority.MEDIUM
            else:
                priority = Priority.LOW

            blocks.append(
                ContextBlock(
                    block_id=self._id_gen.generate(),
                    block_type=BlockType.CONVERSATION,
                    priority=priority,
                    token_estimate=self._tok.estimate_text(msg.content),
                    content=f"[{msg.role.value}]: {msg.content}",
                )
            )
        return blocks

    def _derive_state_blocks(self, doc: SessionDocument) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        tasks = doc.session.task_state.todo_list.tasks
        if tasks:
            lines = []
            for t in tasks:
                lines.append(f"- [{t.status.value}] {t.name} (id={t.task_id})")
            content = "## Current Tasks\n" + "\n".join(lines)
            blocks.append(
                ContextBlock(
                    block_id=self._id_gen.generate(),
                    block_type=BlockType.STATE,
                    priority=Priority.HIGH,
                    token_estimate=self._tok.estimate_text(content),
                    content=content,
                )
            )
        return blocks

    def _derive_evidence_blocks(
        self, evidences: list[Evidence]
    ) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        for ev in evidences:
            if not ev.content:
                continue
            confidence = ev.confidence or 0.5
            if confidence >= 0.8:
                priority = Priority.HIGH
            elif confidence >= 0.5:
                priority = Priority.MEDIUM
            else:
                priority = Priority.LOW

            blocks.append(
                ContextBlock(
                    block_id=self._id_gen.generate(),
                    block_type=BlockType.EVIDENCE,
                    priority=priority,
                    token_estimate=self._tok.estimate_text(ev.content),
                    content=ev.content,
                    refs=[Ref(evidence_id=ev.evidence_id)],
                )
            )
        return blocks
