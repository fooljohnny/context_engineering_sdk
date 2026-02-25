"""Summarizer: rolling summary generation for conversation compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from context_engineering_sdk.core.errors import SummarizerError
from context_engineering_sdk.core.token_estimator import CharBasedEstimator, TokenEstimator
from context_engineering_sdk.core.types import (
    Message,
    MessageIndexRange,
    MessagePart,
    Role,
    Session,
    Summary,
)


@dataclass
class SummarizeConfig:
    max_summary_tokens: int = 512
    trigger_message_count: int = 20
    trigger_token_threshold: int = 4096
    preserve_recent_messages: int = 5


class LlmAdapter(Protocol):
    async def generate(self, request: "LlmRequest") -> "LlmResponse": ...


@dataclass
class LlmRequest:
    messages: list[MessagePart]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.3


@dataclass
class LlmResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


class Summarizer(Protocol):
    async def should_summarize(
        self, session: Session, config: SummarizeConfig
    ) -> bool: ...

    async def summarize(
        self,
        messages: list[Message],
        existing_summary: Summary | None = None,
        config: SummarizeConfig | None = None,
    ) -> Summary: ...


class DefaultSummarizer:
    """Default summarizer that uses an LLM adapter to generate summaries."""

    def __init__(
        self,
        llm_adapter: LlmAdapter,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._llm = llm_adapter
        self._tok = token_estimator or CharBasedEstimator()

    async def should_summarize(
        self, session: Session, config: SummarizeConfig
    ) -> bool:
        msg_count = len(session.messages)
        if msg_count < config.trigger_message_count:
            return False

        total_tokens = sum(
            self._tok.estimate_text(m.content) for m in session.messages
        )
        return total_tokens >= config.trigger_token_threshold

    async def summarize(
        self,
        messages: list[Message],
        existing_summary: Summary | None = None,
        config: SummarizeConfig | None = None,
    ) -> Summary:
        cfg = config or SummarizeConfig()

        # Determine which messages to summarize
        preserve = cfg.preserve_recent_messages
        if preserve >= len(messages):
            # Nothing to summarize
            return existing_summary or Summary(
                content="",
                updated_at="",
                message_index_range=MessageIndexRange(from_index=0, to_index=0),
            )

        to_summarize = messages[:-preserve] if preserve > 0 else messages
        summary_end_idx = len(messages) - preserve

        prompt_parts: list[MessagePart] = []

        system_prompt = (
            "You are a conversation summarizer. Create a concise summary of "
            "the following conversation, preserving key information, decisions, "
            "and context needed for future turns. Output ONLY the summary text."
        )
        prompt_parts.append(MessagePart(role=Role.SYSTEM, content=system_prompt))

        if existing_summary and existing_summary.content:
            prompt_parts.append(
                MessagePart(
                    role=Role.USER,
                    content=f"Previous summary:\n{existing_summary.content}\n\n"
                    f"New messages to incorporate:",
                )
            )
            # Only include messages after the existing summary range
            start = existing_summary.message_index_range.to_index
            new_messages = to_summarize[start:]
        else:
            prompt_parts.append(
                MessagePart(
                    role=Role.USER,
                    content="Summarize the following conversation:",
                )
            )
            new_messages = to_summarize

        conversation_text = "\n".join(
            f"[{m.role.value}]: {m.content}" for m in new_messages
        )
        prompt_parts.append(
            MessagePart(role=Role.USER, content=conversation_text)
        )

        try:
            response = await self._llm.generate(
                LlmRequest(
                    messages=prompt_parts,
                    max_tokens=cfg.max_summary_tokens,
                    temperature=0.3,
                )
            )
        except Exception as e:
            raise SummarizerError(f"LLM call failed: {e}") from e

        from context_engineering_sdk.core.clock import SystemClock

        clock = SystemClock()

        return Summary(
            content=response.content,
            updated_at=clock.now_iso(),
            message_index_range=MessageIndexRange(
                from_index=0, to_index=summary_end_idx
            ),
        )
