"""Token estimation utilities."""

from __future__ import annotations

import re
from typing import Protocol

from context_engineering_sdk.core.types import Message, Role


class TokenEstimator(Protocol):
    def estimate_text(self, text: str, model_hint: str | None = None) -> int: ...

    def estimate_message(
        self, role: Role, content: str, model_hint: str | None = None
    ) -> int: ...

    def estimate_messages(
        self, messages: list[Message], model_hint: str | None = None
    ) -> int: ...


class CharBasedEstimator:
    """Fallback estimator: ~2 tokens per CJK char, ~0.75 tokens per word."""

    _CJK_RANGES = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
        r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
        r"\U0002b740-\U0002b81f\U0002b820-\U0002ceaf"
        r"\U0002ceb0-\U0002ebef\U00030000-\U0003134f"
        r"\u3000-\u303f\uff00-\uffef]"
    )

    def estimate_text(self, text: str, model_hint: str | None = None) -> int:
        if not text:
            return 0
        cjk_count = len(self._CJK_RANGES.findall(text))
        non_cjk = self._CJK_RANGES.sub("", text)
        word_count = len(non_cjk.split())
        # ~2 tokens per CJK char, ~1.33 tokens per English word
        return int(cjk_count * 2 + word_count * 1.33) + 4  # overhead per segment

    def estimate_message(
        self, role: Role, content: str, model_hint: str | None = None
    ) -> int:
        return self.estimate_text(content, model_hint) + 4  # message framing overhead

    def estimate_messages(
        self, messages: list[Message], model_hint: str | None = None
    ) -> int:
        total = 3  # priming overhead
        for msg in messages:
            total += self.estimate_message(msg.role, msg.content, model_hint)
        return total
