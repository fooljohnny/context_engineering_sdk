"""
个性化记忆示例
==============

演示如何使用 Context Engineering SDK 实现跨会话的个性化记忆功能：
- 用户偏好（语言、风格、专业领域）作为持久化证据存储
- 历史会话摘要作为长期记忆跨会话携带
- 通过 memory 类型的 context_block 将记忆注入上下文
- 记忆的优先级管理与预算裁剪

运行方式:
    python examples/personalized_memory.py
"""

import asyncio

from context_engineering_sdk import (
    Message,
    Role,
    Author,
    AuthorKind,
    RuntimeConfig,
    BudgetConfig,
    SummaryConfig,
    create_context_engine,
)
from context_engineering_sdk.core.id_generator import UuidV4Generator
from context_engineering_sdk.core.clock import SystemClock
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.core.types import (
    BlockType,
    ContextBlock,
    Evidence,
    EvidenceSource,
    EvidenceType,
    Priority,
    Ref,
    SourceKind,
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.deriver import DefaultBlockDeriver, DeriveOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# Mock LLM Adapter
# ---------------------------------------------------------------------------

class MockLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="摘要内容", model="mock-model")


# ---------------------------------------------------------------------------
# 用户画像 / 偏好管理器
# ---------------------------------------------------------------------------

class UserProfileManager:
    """管理用户个性化偏好，将其作为证据存入 SDK。"""

    def __init__(self, store, ingestor, id_gen):
        self._store = store
        self._ingestor = ingestor
        self._id_gen = id_gen

    async def save_preference(
        self, session_id: str, category: str, content: str
    ) -> Evidence:
        """将用户偏好作为证据存储。"""
        evidence, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=content,
            source=EvidenceSource(
                kind=SourceKind.USER,
                name=f"user_preference:{category}",
            ),
            evidence_type=EvidenceType.USER_INPUT,
            metadata={"category": category, "is_preference": True},
            options=IngestOptions(redact=False, dedup=True),
        )
        return evidence

    async def save_memory(
        self, session_id: str, memory_content: str, source_session: str
    ) -> Evidence:
        """将跨会话记忆作为证据存储。"""
        evidence, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=memory_content,
            source=EvidenceSource(
                kind=SourceKind.SYSTEM,
                name="cross_session_memory",
                uri=f"session://{source_session}",
            ),
            evidence_type=EvidenceType.OTHER,
            metadata={"memory_type": "cross_session", "source_session": source_session},
            options=IngestOptions(redact=False, dedup=True),
        )
        return evidence

    def build_memory_blocks(
        self, preferences: list[Evidence], memories: list[Evidence]
    ) -> list[ContextBlock]:
        """将偏好和记忆转化为 context_blocks 注入上下文。"""
        blocks = []

        # 用户偏好作为高优先级 memory block
        if preferences:
            pref_lines = []
            refs = []
            for pref in preferences:
                category = pref.metadata.get("category", "general")
                pref_lines.append(f"- [{category}] {pref.content}")
                refs.append(Ref(evidence_id=pref.evidence_id))
            blocks.append(ContextBlock(
                block_id=self._id_gen.generate(),
                block_type=BlockType.MEMORY,
                priority=Priority.HIGH,
                content="## 用户偏好\n" + "\n".join(pref_lines),
                refs=refs,
            ))

        # 跨会话记忆
        for mem in memories:
            source = mem.metadata.get("source_session", "unknown")
            blocks.append(ContextBlock(
                block_id=self._id_gen.generate(),
                block_type=BlockType.MEMORY,
                priority=Priority.MEDIUM,
                content=f"## 历史记忆（来自会话 {source}）\n{mem.content}",
                refs=[Ref(evidence_id=mem.evidence_id)],
            ))

        return blocks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main():
    store = MemoryStore()
    token_estimator = CharBasedEstimator()
    id_gen = UuidV4Generator()
    clock = SystemClock()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=token_estimator,
        llm_adapter=MockLlmAdapter(),
    )

    profile_mgr = UserProfileManager(store, ingestor, id_gen)

    config = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=4096, reserved_reply_tokens=512),
        summary=SummaryConfig(enabled=False),
    )

    print("=" * 60)
    print("  Context Engineering SDK — 个性化记忆示例")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 会话 1：用户建立偏好
    # ---------------------------------------------------------------
    session1 = "session-memory-001"
    print("\n--- 会话 1：建立用户偏好 ---")

    # 初始化会话
    await engine.prepare_turn(
        session1,
        Message(role=Role.USER, content="你好"),
        config,
    )
    await engine.commit_assistant_message(
        session1, Message(role=Role.ASSISTANT, content="你好！")
    )

    # 存储用户偏好
    pref1 = await profile_mgr.save_preference(
        session1, "language", "用户偏好使用中文交流"
    )
    print(f"  [偏好] 存储语言偏好: {pref1.content}")

    pref2 = await profile_mgr.save_preference(
        session1, "style", "用户喜欢简洁的回答风格，不需要过多解释"
    )
    print(f"  [偏好] 存储风格偏好: {pref2.content}")

    pref3 = await profile_mgr.save_preference(
        session1, "domain", "用户是一名后端开发工程师，熟悉 Python 和 Go"
    )
    print(f"  [偏好] 存储领域偏好: {pref3.content}")

    # 模拟会话 1 的对话和总结
    await engine.prepare_turn(
        session1,
        Message(role=Role.USER, content="帮我写一个 Python 装饰器"),
        config,
    )
    await engine.commit_assistant_message(
        session1,
        Message(role=Role.ASSISTANT, content="```python\ndef my_decorator(func):\n    ...\n```"),
    )

    session1_summary = "用户是一名后端开发工程师，在第一次会话中请求了 Python 装饰器的帮助。"

    # ---------------------------------------------------------------
    # 会话 2：跨会话携带记忆
    # ---------------------------------------------------------------
    session2 = "session-memory-002"
    print("\n--- 会话 2：跨会话记忆 ---")

    # 初始化会话 2（注入会话 1 的摘要作为跨会话记忆）
    memory_ev = await profile_mgr.save_memory(
        session2, session1_summary, source_session=session1
    )
    print(f"  [记忆] 跨会话记忆已注入: {memory_ev.content}")

    # 将偏好也存入新会话
    for pref in [pref1, pref2, pref3]:
        await ingestor.ingest(
            session_id=session2,
            content=pref.content,
            source=pref.source,
            evidence_type=pref.type,
            metadata=pref.metadata,
            options=IngestOptions(redact=False, dedup=True),
        )

    # 准备个性化 memory blocks
    all_evidences = await store.list_evidences(session2)
    preferences = [e for e in all_evidences if e.metadata.get("is_preference")]
    memories = [e for e in all_evidences if e.metadata.get("memory_type") == "cross_session"]

    memory_blocks = profile_mgr.build_memory_blocks(preferences, memories)
    print(f"  [记忆] 构建了 {len(memory_blocks)} 个记忆块")
    for block in memory_blocks:
        print(f"    - [{block.priority.value}] {block.content[:60]}...")

    # 使用自定义 deriver 注入记忆块
    custom_deriver = DefaultBlockDeriver(token_estimator=token_estimator, id_generator=id_gen)
    engine._deriver = custom_deriver

    # 准备第一轮（带记忆的上下文）
    user_msg = Message(
        role=Role.USER,
        content="继续帮我写一个 Go 的中间件",
    )

    # 手动注入 memory blocks 通过 DeriveOptions
    original_derive = engine._deriver.derive

    async def derive_with_memory(session_doc, evidences, options=None):
        opts = options or DeriveOptions()
        opts.custom_instructions = memory_blocks
        return await original_derive(session_doc, evidences, opts)

    engine._deriver.derive = derive_with_memory

    result = await engine.prepare_turn(session2, user_msg, config)
    print(f"\n  [用户]: {user_msg.content}")
    print(f"  [装配] 输入包含 {len(result.assembled_input.parts)} 个部分")
    print(f"  [装配] 总 token 估算: {result.assembled_input.total_tokens}")

    # 验证记忆是否注入到了上下文中
    assembled_text = result.assembled_input.text or ""
    has_pref = "用户偏好" in assembled_text or "后端开发工程师" in assembled_text
    has_memory = "历史记忆" in assembled_text or "Python 装饰器" in assembled_text
    print(f"  [验证] 上下文包含用户偏好: {has_pref}")
    print(f"  [验证] 上下文包含跨会话记忆: {has_memory}")

    await engine.commit_assistant_message(
        session2,
        Message(role=Role.ASSISTANT, content="```go\nfunc Middleware(next http.Handler) ...\n```"),
    )

    # ---------------------------------------------------------------
    # 最终状态
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  最终状态汇总")
    print("=" * 60)

    for sid in [session1, session2]:
        doc = await store.get_session(sid)
        evidences = await store.list_evidences(sid)
        print(f"\n  会话 {sid}:")
        print(f"    消息数: {len(doc.session.messages)}")
        print(f"    证据数: {len(evidences)}")
        pref_count = sum(1 for e in evidences if e.metadata.get("is_preference"))
        mem_count = sum(1 for e in evidences if e.metadata.get("memory_type"))
        print(f"    偏好证据: {pref_count}")
        print(f"    跨会话记忆: {mem_count}")


if __name__ == "__main__":
    asyncio.run(main())
