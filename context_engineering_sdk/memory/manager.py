"""MemoryManager: orchestrates cross-session user memory lifecycle."""

from __future__ import annotations

from context_engineering_sdk.core.clock import Clock, SystemClock
from context_engineering_sdk.core.id_generator import IdGenerator, UuidV4Generator
from context_engineering_sdk.core.token_estimator import CharBasedEstimator, TokenEstimator
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Priority,
    Ref,
)
from context_engineering_sdk.memory.user_memory_store import UserMemory, UserMemoryStore


class MemoryManager:
    """Manages cross-session user memories and builds context blocks from them.

    Typical usage::

        mgr = MemoryManager(user_memory_store)
        await mgr.add_preference(user_id, "language", "Prefers Chinese")
        await mgr.add_session_summary(user_id, session_id, "Discussed Python decorators")
        blocks = await mgr.build_memory_blocks(user_id)
        # Pass blocks via RuntimeConfig.extra_context_blocks
    """

    def __init__(
        self,
        user_memory_store: UserMemoryStore,
        *,
        id_generator: IdGenerator | None = None,
        clock: Clock | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._store = user_memory_store
        self._id_gen = id_generator or UuidV4Generator()
        self._clock = clock or SystemClock()
        self._tok = token_estimator or CharBasedEstimator()

    async def add_preference(
        self,
        user_id: str,
        category: str,
        content: str,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> UserMemory:
        """Save a user preference (e.g., language, style, domain expertise)."""
        now = self._clock.now_iso()
        mem = UserMemory(
            memory_id=self._id_gen.generate(),
            user_id=user_id,
            category=f"preference:{category}",
            content=content,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            source_session_id=session_id,
        )
        await self._store.save_memory(mem)
        return mem

    async def add_user_fact(
        self,
        user_id: str,
        content: str,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> UserMemory:
        """Save a factual observation about the user (profile info, behavior)."""
        now = self._clock.now_iso()
        mem = UserMemory(
            memory_id=self._id_gen.generate(),
            user_id=user_id,
            category="profile",
            content=content,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            source_session_id=session_id,
        )
        await self._store.save_memory(mem)
        return mem

    async def add_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary_content: str,
        metadata: dict | None = None,
    ) -> UserMemory:
        """Save a session summary as cross-session memory."""
        now = self._clock.now_iso()
        mem = UserMemory(
            memory_id=self._id_gen.generate(),
            user_id=user_id,
            category="session_summary",
            content=summary_content,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            source_session_id=session_id,
        )
        await self._store.save_memory(mem)
        return mem

    async def get_all_memories(
        self, user_id: str, limit: int | None = None
    ) -> list[UserMemory]:
        return await self._store.get_memories(user_id, limit=limit)

    async def get_preferences(self, user_id: str) -> list[UserMemory]:
        all_mems = await self._store.get_memories(user_id)
        return [m for m in all_mems if m.category.startswith("preference:")]

    async def get_profile_facts(self, user_id: str) -> list[UserMemory]:
        return await self._store.get_memories(user_id, category="profile")

    async def get_session_summaries(
        self, user_id: str, limit: int | None = None
    ) -> list[UserMemory]:
        return await self._store.get_memories(
            user_id, category="session_summary", limit=limit
        )

    async def build_memory_blocks(
        self,
        user_id: str,
        *,
        max_tokens: int | None = None,
        include_preferences: bool = True,
        include_profile: bool = True,
        include_session_summaries: bool = True,
        max_session_summaries: int = 5,
    ) -> list[ContextBlock]:
        """Build ContextBlocks from user memories for injection into the pipeline.

        Returns blocks sorted by priority: preferences (HIGH), profile (HIGH),
        session summaries (MEDIUM).
        """
        blocks: list[ContextBlock] = []
        token_budget = max_tokens
        consumed = 0

        if include_preferences:
            prefs = await self.get_preferences(user_id)
            if prefs:
                lines = []
                for p in prefs:
                    cat = p.category.removeprefix("preference:")
                    lines.append(f"- [{cat}] {p.content}")
                content = "## User Preferences\n" + "\n".join(lines)
                est = self._tok.estimate_text(content)
                if token_budget is None or consumed + est <= token_budget:
                    blocks.append(ContextBlock(
                        block_id=self._id_gen.generate(),
                        block_type=BlockType.MEMORY,
                        priority=Priority.HIGH,
                        token_estimate=est,
                        content=content,
                    ))
                    consumed += est

        if include_profile:
            facts = await self.get_profile_facts(user_id)
            if facts:
                lines = [f"- {f.content}" for f in facts]
                content = "## User Profile\n" + "\n".join(lines)
                est = self._tok.estimate_text(content)
                if token_budget is None or consumed + est <= token_budget:
                    blocks.append(ContextBlock(
                        block_id=self._id_gen.generate(),
                        block_type=BlockType.MEMORY,
                        priority=Priority.HIGH,
                        token_estimate=est,
                        content=content,
                    ))
                    consumed += est

        if include_session_summaries:
            summaries = await self.get_session_summaries(
                user_id, limit=max_session_summaries
            )
            if summaries:
                lines = []
                for s in summaries:
                    src = s.source_session_id or "unknown"
                    lines.append(f"- [session:{src}] {s.content}")
                content = "## Previous Session Summaries\n" + "\n".join(lines)
                est = self._tok.estimate_text(content)
                if token_budget is None or consumed + est <= token_budget:
                    blocks.append(ContextBlock(
                        block_id=self._id_gen.generate(),
                        block_type=BlockType.MEMORY,
                        priority=Priority.MEDIUM,
                        token_estimate=est,
                        content=content,
                    ))
                    consumed += est

        return blocks
