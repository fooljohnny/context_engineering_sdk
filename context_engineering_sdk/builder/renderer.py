"""Renderer and EvidenceResolver: resolve refs and render blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from context_engineering_sdk.core.ref_selector import RefSelector
from context_engineering_sdk.core.types import ContextBlock, RenderedBlock
from context_engineering_sdk.store.base import Store


@dataclass
class RenderHints:
    max_chars: int | None = None
    format: str = "text"  # "text" | "markdown" | "json"


class EvidenceResolver(Protocol):
    async def resolve(
        self, evidence_id: str, selector: str | None = None
    ) -> str: ...


class DefaultEvidenceResolver:
    """Resolves evidence content from Store, applying selectors."""

    def __init__(
        self,
        store: Store,
        session_id: str,
        ref_selector: RefSelector | None = None,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._ref_selector = ref_selector or RefSelector()

    async def resolve(
        self, evidence_id: str, selector: str | None = None
    ) -> str:
        evidence = await self._store.get_evidence(self._session_id, evidence_id)
        if evidence is None:
            return ""
        content = evidence.content
        if selector:
            try:
                content = self._ref_selector.extract(content, selector)
            except Exception:
                pass  # Degradation: use full content on selector failure
        return content


class Renderer(Protocol):
    async def render_block(
        self,
        block: ContextBlock,
        resolver: EvidenceResolver,
        hints: RenderHints | None = None,
    ) -> RenderedBlock: ...


class DefaultRenderer:
    """Renders blocks by resolving refs and applying format hints."""

    async def render_block(
        self,
        block: ContextBlock,
        resolver: EvidenceResolver,
        hints: RenderHints | None = None,
    ) -> RenderedBlock:
        hints = hints or RenderHints()

        if block.refs:
            resolved_parts: list[str] = []
            for ref in block.refs:
                fragment = await resolver.resolve(ref.evidence_id, ref.selector)
                if fragment:
                    resolved_parts.append(fragment)
            if resolved_parts:
                content = "\n---\n".join(resolved_parts)
            else:
                content = block.content
        else:
            content = block.content

        if hints.max_chars and len(content) > hints.max_chars:
            content = content[: hints.max_chars] + "..."

        return RenderedBlock(
            block_id=block.block_id,
            block_type=block.block_type,
            priority=block.priority,
            rendered_content=content,
            token_estimate=block.token_estimate,
        )
