"""ContextEngine: the main orchestration entry point."""

from __future__ import annotations

import asyncio
from typing import Protocol

from context_engineering_sdk.builder.assembler import Assembler, DefaultAssembler
from context_engineering_sdk.builder.deriver import (
    BlockDeriver,
    DefaultBlockDeriver,
    DeriveOptions,
)
from context_engineering_sdk.builder.ingestor import (
    DefaultEvidenceIngestor,
    EvidenceIngestor,
    IngestOptions,
)
from context_engineering_sdk.builder.pruner import (
    GreedyPriorityPruner,
    PruneBudget,
    PruneRules,
    Pruner,
)
from context_engineering_sdk.builder.renderer import (
    DefaultEvidenceResolver,
    DefaultRenderer,
    Renderer,
)
from context_engineering_sdk.builder.summarizer import (
    DefaultSummarizer,
    LlmAdapter,
    Summarizer,
    SummarizeConfig,
)
from context_engineering_sdk.config import RuntimeConfig
from context_engineering_sdk.core.clock import Clock, SystemClock
from context_engineering_sdk.core.errors import (
    BudgetExceededError,
    SessionNotFoundError,
    SummarizerError,
)
from context_engineering_sdk.core.hasher import Hasher, Sha256Hasher
from context_engineering_sdk.core.id_generator import IdGenerator, UuidV4Generator
from context_engineering_sdk.core.redactor import Redactor, RegexRedactor
from context_engineering_sdk.core.ref_selector import RefSelector
from context_engineering_sdk.core.token_estimator import CharBasedEstimator, TokenEstimator
from context_engineering_sdk.core.types import (
    AssembledInput,
    ContextBlock,
    Message,
    MessagePart,
    ModelUsage,
    PrepareResult,
    PruneDecision,
    PutResult,
    Ref,
    Report,
    Role,
    Session,
    SessionDocument,
    ToolCall,
)
from context_engineering_sdk.observability.event_bus import (
    Event,
    EventBus,
    InMemoryEventBus,
)
from context_engineering_sdk.store.base import Store


class ContextEngine(Protocol):
    async def prepare_turn(
        self,
        session_id: str,
        user_message: Message,
        runtime_config: RuntimeConfig,
    ) -> PrepareResult: ...

    async def commit_assistant_message(
        self,
        session_id: str,
        assistant_message: Message,
        refs: list[Ref] | None = None,
    ) -> PutResult: ...

    async def commit_assistant_chunk(
        self, session_id: str, chunk: str, chunk_index: int
    ) -> None: ...

    async def finalize_assistant_message(
        self, session_id: str, refs: list[Ref] | None = None
    ) -> PutResult: ...

    async def record_tool_call(
        self,
        session_id: str,
        tool_call: ToolCall,
        result_evidence_ids: list[str] | None = None,
    ) -> PutResult: ...

    async def record_model_usage(
        self,
        session_id: str,
        model_usage: ModelUsage,
        llm_output_evidence_id: str | None = None,
    ) -> PutResult: ...


class DefaultContextEngine:
    """Default ContextEngine: orchestrates the full context building pipeline."""

    def __init__(
        self,
        store: Store,
        token_estimator: TokenEstimator,
        llm_adapter: LlmAdapter,
        *,
        ingestor: EvidenceIngestor | None = None,
        deriver: BlockDeriver | None = None,
        pruner: Pruner | None = None,
        renderer: Renderer | None = None,
        assembler: Assembler | None = None,
        summarizer: Summarizer | None = None,
        event_bus: EventBus | None = None,
        id_generator: IdGenerator | None = None,
        clock: Clock | None = None,
        hasher: Hasher | None = None,
        redactor: Redactor | None = None,
        ref_selector: RefSelector | None = None,
    ) -> None:
        self._store = store
        self._tok = token_estimator
        self._llm = llm_adapter

        self._id_gen = id_generator or UuidV4Generator()
        self._clock = clock or SystemClock()
        self._hasher = hasher or Sha256Hasher()
        self._redactor = redactor or RegexRedactor()
        self._ref_selector = ref_selector or RefSelector()
        self._event_bus = event_bus or InMemoryEventBus()

        self._ingestor = ingestor or DefaultEvidenceIngestor(
            store=store,
            id_generator=self._id_gen,
            hasher=self._hasher,
            redactor=self._redactor,
        )
        self._deriver = deriver or DefaultBlockDeriver(
            token_estimator=token_estimator, id_generator=self._id_gen
        )
        self._pruner = pruner or GreedyPriorityPruner()
        self._renderer = renderer or DefaultRenderer()
        self._assembler = assembler or DefaultAssembler(
            token_estimator=token_estimator
        )
        self._summarizer = summarizer or DefaultSummarizer(
            llm_adapter=llm_adapter, token_estimator=token_estimator
        )

        # Per-session streaming chunk buffers
        self._chunk_buffers: dict[str, dict[int, str]] = {}
        # Per-session locks for serializing operations
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def _emit(self, event_type: str, session_id: str, **kwargs) -> None:
        event = Event(
            event_type=event_type,
            session_id=session_id,
            ts=self._clock.now_iso(),
            **kwargs,
        )
        await self._event_bus.emit(event)

    async def prepare_turn(
        self,
        session_id: str,
        user_message: Message,
        runtime_config: RuntimeConfig,
    ) -> PrepareResult:
        async with self._get_lock(session_id):
            return await self._prepare_turn_impl(
                session_id, user_message, runtime_config
            )

    async def _prepare_turn_impl(
        self,
        session_id: str,
        user_message: Message,
        cfg: RuntimeConfig,
    ) -> PrepareResult:
        turn_id = self._id_gen.generate()
        degradations: list[str] = []
        errors: list[str] = []
        new_evidence_ids: list[str] = []
        new_block_ids: list[str] = []
        redaction_records = []
        conflict_records = []

        # 1. Load or create session
        doc = await self._store.get_session(session_id)
        if doc is None:
            doc = SessionDocument(
                schema_version="1.0",
                session=Session(session_id=session_id),
            )
            await self._store.put_session(session_id, doc)
            await self._emit("SessionLoaded", session_id,
                             payload={"created": True}, turn_id=turn_id)
        else:
            await self._emit("SessionLoaded", session_id,
                             payload={"created": False}, turn_id=turn_id)

        # 2. Append user message
        if not user_message.at:
            user_message.at = self._clock.now_iso()
        await self._store.append_messages(session_id, [user_message])
        doc = await self._store.get_session(session_id)
        await self._emit("MessageAppended", session_id,
                         payload={"role": user_message.role.value}, turn_id=turn_id)

        # 3. Summarize if needed
        if cfg.summary.enabled:
            summarize_cfg = SummarizeConfig(
                max_summary_tokens=cfg.summary.max_summary_tokens,
                trigger_message_count=cfg.summary.trigger_message_count,
                trigger_token_threshold=cfg.summary.trigger_token_threshold,
                preserve_recent_messages=cfg.summary.preserve_recent_messages,
            )
            try:
                should = await self._summarizer.should_summarize(
                    doc.session, summarize_cfg
                )
                if should:
                    summary = await self._summarizer.summarize(
                        doc.session.messages,
                        existing_summary=doc.session.summary,
                        config=summarize_cfg,
                    )
                    doc.session.summary = summary
                    await self._store.patch_session(
                        session_id, {"summary": summary}
                    )
                    await self._emit("SummaryGenerated", session_id,
                                     turn_id=turn_id)
            except SummarizerError as e:
                degradations.append(f"summarize_failed: {e}")
                await self._emit("Degraded", session_id,
                                 payload={"reason": "summarize_failed"},
                                 turn_id=turn_id)

        # 4. List existing evidences
        evidences = await self._store.list_evidences(session_id)

        # 5. Derive blocks
        blocks = await self._deriver.derive(doc, evidences)

        # 5b. Append extra context blocks (e.g. cross-session memory)
        if cfg.extra_context_blocks:
            blocks.extend(cfg.extra_context_blocks)

        new_block_ids = [b.block_id for b in blocks]
        await self._emit("BlocksDerived", session_id,
                         payload={"count": len(blocks)}, turn_id=turn_id)

        # 6. Prune blocks
        max_context_tokens = (
            cfg.budget.max_input_tokens - cfg.budget.reserved_reply_tokens
        )
        budget = PruneBudget(max_tokens=max_context_tokens)
        prune_rules = PruneRules(
            must_keep_priorities=cfg.prune.must_keep_priorities
        )

        prune_decisions: list[PruneDecision] = []
        try:
            prune_result = self._pruner.prune(blocks, budget, prune_rules)
            kept_blocks = prune_result.kept
            prune_decisions = prune_result.decisions
        except BudgetExceededError as e:
            errors.append(str(e))
            kept_blocks = blocks
            await self._emit("Error", session_id,
                             payload={"reason": "must_exceeded_budget"},
                             turn_id=turn_id)

        await self._emit("PruneCompleted", session_id,
                         payload={"kept": len(kept_blocks),
                                  "dropped": len(blocks) - len(kept_blocks)},
                         turn_id=turn_id)

        # 7. Persist derived blocks
        if kept_blocks:
            await self._store.put_context_blocks(session_id, kept_blocks)

        # 8. Render blocks
        resolver = DefaultEvidenceResolver(
            store=self._store,
            session_id=session_id,
            ref_selector=self._ref_selector,
        )
        rendered_blocks = []
        for block in kept_blocks:
            try:
                rendered = await self._renderer.render_block(block, resolver)
                rendered_blocks.append(rendered)
            except Exception as e:
                degradations.append(f"render_failed:{block.block_id}: {e}")

        # 9. Assemble
        # Filter conversation blocks from rendered (they'll come from messages)
        from context_engineering_sdk.core.types import BlockType

        non_conversation_rendered = [
            b for b in rendered_blocks if b.block_type != BlockType.CONVERSATION
        ]
        conversation_rendered = [
            b for b in rendered_blocks if b.block_type == BlockType.CONVERSATION
        ]

        # Build message list for assembler
        # Use the most recent messages (after summary if any)
        if doc.session.summary and doc.session.summary.content:
            summary_end = doc.session.summary.message_index_range.to_index
            recent_messages = doc.session.messages[summary_end:]
        else:
            recent_messages = doc.session.messages

        assembled = self._assembler.assemble(
            recent_messages, rendered_blocks, cfg.model_hint
        )

        await self._emit("Assembled", session_id,
                         payload={"total_tokens": assembled.total_tokens},
                         turn_id=turn_id)

        token_used = assembled.total_tokens
        report = Report(
            turn_id=turn_id,
            new_evidence_ids=new_evidence_ids,
            new_block_ids=new_block_ids,
            prune_decisions=prune_decisions,
            conflicts=conflict_records,
            redactions=redaction_records,
            token_budget=max_context_tokens,
            token_used=token_used,
            degradations=degradations,
            errors=errors,
        )

        return PrepareResult(
            assembled_input=assembled,
            report=report,
        )

    async def commit_assistant_message(
        self,
        session_id: str,
        assistant_message: Message,
        refs: list[Ref] | None = None,
    ) -> PutResult:
        async with self._get_lock(session_id):
            if refs:
                assistant_message.refs = refs
            if not assistant_message.at:
                assistant_message.at = self._clock.now_iso()
            result = await self._store.append_messages(
                session_id, [assistant_message]
            )
            await self._emit(
                "AssistantMessageFinalized",
                session_id,
                payload={"content_length": len(assistant_message.content)},
            )
            return result

    async def commit_assistant_chunk(
        self, session_id: str, chunk: str, chunk_index: int
    ) -> None:
        if session_id not in self._chunk_buffers:
            self._chunk_buffers[session_id] = {}
        self._chunk_buffers[session_id][chunk_index] = chunk
        await self._emit(
            "AssistantChunkReceived",
            session_id,
            payload={"chunk_index": chunk_index, "chunk_length": len(chunk)},
        )

    async def finalize_assistant_message(
        self, session_id: str, refs: list[Ref] | None = None
    ) -> PutResult:
        async with self._get_lock(session_id):
            chunks = self._chunk_buffers.pop(session_id, {})
            if not chunks:
                return PutResult(success=True, version=0)

            sorted_indices = sorted(chunks.keys())
            full_content = "".join(chunks[i] for i in sorted_indices)

            assistant_message = Message(
                role=Role.ASSISTANT,
                content=full_content,
                at=self._clock.now_iso(),
                refs=refs,
            )
            result = await self._store.append_messages(
                session_id, [assistant_message]
            )
            await self._emit(
                "AssistantMessageFinalized",
                session_id,
                payload={"content_length": len(full_content), "chunks": len(chunks)},
            )
            return result

    async def record_tool_call(
        self,
        session_id: str,
        tool_call: ToolCall,
        result_evidence_ids: list[str] | None = None,
    ) -> PutResult:
        async with self._get_lock(session_id):
            if result_evidence_ids:
                tool_call.result_evidence_ids = result_evidence_ids
            result = await self._store.append_tool_calls(
                session_id, [tool_call]
            )
            await self._emit(
                "ToolCallCompleted",
                session_id,
                tool_call_id=tool_call.tool_call_id,
                payload={
                    "tool": tool_call.tool,
                    "status": tool_call.status.value,
                    "duration_ms": tool_call.duration_ms,
                },
            )
            return result

    async def record_model_usage(
        self,
        session_id: str,
        model_usage: ModelUsage,
        llm_output_evidence_id: str | None = None,
    ) -> PutResult:
        async with self._get_lock(session_id):
            result = await self._store.append_model_usage(
                session_id, [model_usage]
            )
            await self._emit(
                "ModelUsageRecorded",
                session_id,
                model_usage_id=model_usage.model_usage_id,
                payload={
                    "model": model_usage.model,
                    "total_tokens": model_usage.total_tokens,
                    "llm_output_evidence_id": llm_output_evidence_id,
                },
            )
            return result


def create_context_engine(
    store: Store,
    token_estimator: TokenEstimator,
    llm_adapter: LlmAdapter,
    config: RuntimeConfig | None = None,
) -> DefaultContextEngine:
    """One-line factory to create a ContextEngine with all default components."""
    return DefaultContextEngine(
        store=store,
        token_estimator=token_estimator,
        llm_adapter=llm_adapter,
    )
