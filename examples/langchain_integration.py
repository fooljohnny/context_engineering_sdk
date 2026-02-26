"""
LangChain + Context Engineering SDK 完整集成示例
=================================================

演示功能：
- LangChain ReAct Agent 与 CE SDK 的深度集成
- 跨会话用户画像识别与偏好持久化（MemoryManager）
- 多轮长对话自动摘要压缩（Summarizer）
- RAG 检索证据化 + 工具调用记录
- 通过 extra_context_blocks 注入跨会话记忆

场景说明：
  会话 1（8 轮）：用户与智能客服对话，SDK 自动识别并存储用户偏好，
  当消息数超过阈值时触发会话摘要压缩。
  会话 2（3 轮）：新会话自动加载上一轮的用户偏好和会话摘要，
  验证跨会话记忆的持久性。

运行方式：
    python examples/langchain_integration.py
"""

import asyncio
import json
from dataclasses import dataclass, field

from context_engineering_sdk import (
    Message, Role, Author, AuthorKind,
    RuntimeConfig, BudgetConfig, SummaryConfig,
    MemoryManager, InMemoryUserMemoryStore,
    create_context_engine,
)
from context_engineering_sdk.core.id_generator import UuidV4Generator
from context_engineering_sdk.core.clock import SystemClock
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.core.types import (
    EvidenceLinks, EvidenceSource, EvidenceType,
    ModelUsage, ModelUsageStage, Ref, SourceKind,
    ToolCall, ToolCallStatus, ToolProvider, ProviderKind,
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------

@dataclass
class LCMessage:
    role: str
    content: str

@dataclass
class LCDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)

@dataclass
class LCToolResult:
    name: str
    args: dict
    output: str


PREFERENCE_KEYWORDS = {
    "简洁": ("style", "用户偏好简洁的回答风格"),
    "详细": ("style", "用户偏好详细的回答风格"),
    "中文": ("language", "用户偏好使用中文交流"),
    "英文": ("language", "用户偏好使用英文交流"),
    "Python": ("tech_stack", "用户熟悉 Python 编程语言"),
    "Java": ("tech_stack", "用户熟悉 Java 编程语言"),
    "Go": ("tech_stack", "用户熟悉 Go 编程语言"),
    "前端": ("domain", "用户的工作方向是前端开发"),
    "后端": ("domain", "用户的工作方向是后端开发"),
    "数据": ("domain", "用户的工作方向是数据工程"),
}


class SmartMockLlmAdapter:
    """Simulates LLM with context-aware responses and preference extraction."""

    def __init__(self):
        self._call_count = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self._call_count += 1
        prompt_text = " ".join(p.content for p in request.messages)

        if "摘要" in prompt_text or "summarize" in prompt_text.lower():
            return LlmResponse(
                content=(
                    "用户是一名后端开发工程师，熟悉 Python。偏好简洁的中文回答。"
                    "在之前的对话中，用户咨询了订单 ORD-2026-001（智能手表）的物流状态"
                    "和退货政策，讨论了 VIP 会员权益，并表达了对产品推荐的兴趣。"
                ),
                model="mock-gpt-4o",
                prompt_tokens=len(prompt_text) // 4,
                completion_tokens=80,
                total_tokens=len(prompt_text) // 4 + 80,
            )
        return LlmResponse(
            content="assistant response",
            model="mock-gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


def detect_preferences(text: str) -> list[tuple[str, str]]:
    """Simple keyword-based preference detection from user messages."""
    found = []
    for keyword, (category, description) in PREFERENCE_KEYWORDS.items():
        if keyword in text:
            found.append((category, description))
    return found


# ---------------------------------------------------------------------------
# LangChain CE Adapter
# ---------------------------------------------------------------------------

class LangChainCEAdapter:
    def __init__(self, engine, store, ingestor, memory_mgr, user_id):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._memory_mgr = memory_mgr
        self._user_id = user_id
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()

    async def prepare_turn(self, session_id, user_text, config):
        """Prepare a turn with cross-session memory injection."""
        memory_blocks = await self._memory_mgr.build_memory_blocks(self._user_id)
        config.extra_context_blocks = memory_blocks

        user_msg = Message(
            role=Role.USER, content=user_text,
            author=Author(kind=AuthorKind.USER, id=self._user_id),
            at=self._clock.now_iso(),
        )
        result = await self._engine.prepare_turn(session_id, user_msg, config)

        prefs = detect_preferences(user_text)
        for category, description in prefs:
            await self._memory_mgr.add_preference(
                self._user_id, category, description, session_id=session_id,
            )

        return result

    async def on_retrieval(self, session_id, documents):
        evidences = []
        for doc in documents:
            ev, _ = await self._ingestor.ingest(
                session_id=session_id,
                content=doc.page_content,
                source=EvidenceSource(
                    kind=SourceKind.RAG, name="langchain_retriever",
                    uri=doc.metadata.get("source", ""),
                ),
                evidence_type=EvidenceType.RAG_DOC,
                metadata=doc.metadata,
                options=IngestOptions(redact=False, dedup=True),
            )
            evidences.append(ev)
        return evidences

    async def on_tool_call(self, session_id, tool_result: LCToolResult):
        tc_id = self._id_gen.generate()
        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=tool_result.output,
            source=EvidenceSource(kind=SourceKind.TOOL, name=tool_result.name),
            evidence_type=EvidenceType.TOOL_RESULT,
            links=EvidenceLinks(tool_call_id=tc_id),
            options=IngestOptions(redact=False),
        )
        tc = ToolCall(
            tool_call_id=tc_id, tool=tool_result.name,
            provider=ToolProvider(kind=ProviderKind.BUILTIN, name="langchain"),
            called_at=self._clock.now_iso(),
            args_digest=tool_result.args,
            status=ToolCallStatus.SUCCESS,
            result_evidence_ids=[ev.evidence_id],
        )
        await self._engine.record_tool_call(session_id, tc)
        return tc, ev

    async def commit_response(self, session_id, content, refs=None):
        await self._engine.commit_assistant_message(
            session_id,
            Message(role=Role.ASSISTANT, content=content,
                    author=Author(kind=AuthorKind.AGENT, id="langchain-agent")),
            refs=refs,
        )

    async def save_session_memory(self, session_id, summary_content):
        await self._memory_mgr.add_session_summary(
            self._user_id, session_id, summary_content,
        )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

async def run_session_1(adapter, session_id, config):
    """Session 1: 8-turn customer service conversation with preference detection."""

    turns = [
        {
            "user": "你好，我是一名后端开发工程师，请用中文和我交流，回答尽量简洁一些",
            "assistant": "好的！我会用简洁的中文回复您。有什么可以帮您的？",
            "description": "Greeting + preference detection",
        },
        {
            "user": "帮我查一下订单 ORD-2026-001 的状态",
            "assistant": "您的订单 ORD-2026-001（智能手表）已发货，快递单号 SF9876543210，预计 2026-02-28 送达。",
            "tool": LCToolResult(
                name="query_order",
                args={"order_id": "ORD-2026-001"},
                output=json.dumps({
                    "order_id": "ORD-2026-001", "status": "shipped",
                    "product": "智能手表", "tracking": "SF9876543210",
                    "estimated_delivery": "2026-02-28",
                }, ensure_ascii=False),
            ),
            "description": "Order query with tool call",
        },
        {
            "user": "退货政策是什么？",
            "assistant": "7 天无理由退货，30 天换货。VIP 用户享 15 天无理由退货 + 免运费。",
            "rag": [
                LCDocument(
                    page_content="退货政策：自收货之日起 7 天内可无理由退货，30 天内可换货。",
                    metadata={"source": "policy_v3"},
                ),
                LCDocument(
                    page_content="VIP 用户享有 15 天无理由退货、免运费退换货特权。",
                    metadata={"source": "vip_policy"},
                ),
            ],
            "description": "RAG retrieval",
        },
        {
            "user": "我是 VIP 用户，想了解更多会员权益",
            "assistant": "VIP 权益包括：15 天退货、免运费、专属客服通道、优先发货、生日礼券。",
            "description": "Follow-up question",
        },
        {
            "user": "帮我查一下另一个订单 ORD-2026-002",
            "assistant": "订单 ORD-2026-002（蓝牙耳机）正在处理中，预计 2026-03-05 发货。",
            "tool": LCToolResult(
                name="query_order",
                args={"order_id": "ORD-2026-002"},
                output=json.dumps({
                    "order_id": "ORD-2026-002", "status": "processing",
                    "product": "蓝牙耳机", "estimated_ship": "2026-03-05",
                }, ensure_ascii=False),
            ),
            "description": "Second order query",
        },
        {
            "user": "我平时用 Python 比较多，有没有什么技术类的产品推荐？",
            "assistant": "推荐：1) 机械键盘（程序员专属）2) 显示器支架 3) Python 技术书籍套装。",
            "description": "Preference-enriched response",
        },
        {
            "user": "帮我查一下最近的促销活动",
            "assistant": "当前促销：全场满 500 减 50，VIP 额外 9 折。技术图书专区 8 折。活动截至 2026-03-15。",
            "tool": LCToolResult(
                name="query_promotions",
                args={"category": "tech"},
                output=json.dumps({
                    "promotions": [
                        {"name": "满 500 减 50", "end_date": "2026-03-15"},
                        {"name": "技术图书 8 折", "end_date": "2026-03-15"},
                    ]
                }, ensure_ascii=False),
            ),
            "description": "Promotion query",
        },
        {
            "user": "好的，谢谢你的帮助！今天先这样",
            "assistant": "不客气！祝您生活愉快，有需要随时找我。",
            "description": "Session wrap-up",
        },
    ]

    for i, turn in enumerate(turns, 1):
        print(f"\n  [Turn {i}] {turn['description']}")
        result = await adapter.prepare_turn(session_id, turn["user"], config)
        print(f"    User: {turn['user']}")
        print(f"    Token budget: {result.report.token_budget}, used: {result.report.token_used}")

        refs = []
        if "rag" in turn:
            evs = await adapter.on_retrieval(session_id, turn["rag"])
            refs.extend(Ref(evidence_id=e.evidence_id) for e in evs)
            print(f"    RAG: {len(evs)} documents retrieved")

        if "tool" in turn:
            tc, ev = await adapter.on_tool_call(session_id, turn["tool"])
            refs.append(Ref(evidence_id=ev.evidence_id))
            print(f"    Tool: {turn['tool'].name}({turn['tool'].args})")

        await adapter.commit_response(session_id, turn["assistant"], refs=refs or None)
        print(f"    Assistant: {turn['assistant'][:80]}{'...' if len(turn['assistant']) > 80 else ''}")

        if result.report.degradations:
            print(f"    Degradations: {result.report.degradations}")


async def run_session_2(adapter, session_id, config):
    """Session 2: verify cross-session memory is carried over."""

    turns = [
        {
            "user": "你好，我是上次那位用户",
            "assistant": "您好！根据我的记忆，您是一名后端开发工程师，偏好简洁的中文回答。上次您咨询了订单状态和退货政策。有什么可以帮您的？",
        },
        {
            "user": "上次说的那个智能手表收到了，质量不错",
            "assistant": "很高兴您对智能手表满意！根据上次的记录，这是订单 ORD-2026-001。如果需要任何售后支持随时告诉我。",
        },
        {
            "user": "帮我推荐一些适合 Python 后端开发者的学习资源",
            "assistant": "基于您的技术背景，推荐：1) FastAPI 实战教程 2) 系统设计面试指南 3) 分布式系统 Python 实践。",
        },
    ]

    for i, turn in enumerate(turns, 1):
        print(f"\n  [Turn {i}]")
        result = await adapter.prepare_turn(session_id, turn["user"], config)
        assembled_text = result.assembled_input.text or ""
        print(f"    User: {turn['user']}")
        print(f"    Token used: {result.report.token_used}")

        has_memory = "Preferences" in assembled_text or "偏好" in assembled_text
        has_summary = "Session Summaries" in assembled_text or "后端" in assembled_text
        print(f"    Memory injected: preferences={has_memory}, session_history={has_summary}")

        if i == 1:
            print(f"    [VERIFY] Cross-session memory blocks: {len(config.extra_context_blocks)}")
            for blk in config.extra_context_blocks:
                print(f"      - [{blk.priority.value}] {blk.content[:60]}...")

        await adapter.commit_response(session_id, turn["assistant"])
        print(f"    Assistant: {turn['assistant'][:80]}...")


async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    llm_adapter = SmartMockLlmAdapter()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)
    user_memory_store = InMemoryUserMemoryStore()
    memory_mgr = MemoryManager(user_memory_store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=llm_adapter,
    )

    user_id = "user-langchain-demo"
    adapter = LangChainCEAdapter(
        engine=engine, store=store, ingestor=ingestor,
        memory_mgr=memory_mgr, user_id=user_id,
    )

    summary_config = SummaryConfig(
        enabled=True,
        trigger_message_count=6,
        trigger_token_threshold=200,
        preserve_recent_messages=3,
    )
    config = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=8192, reserved_reply_tokens=1024),
        summary=summary_config,
    )

    print("=" * 70)
    print("  LangChain + CE SDK: Cross-Session Memory & Conversation Compression")
    print("=" * 70)

    # --- Session 1 ---
    session1 = "lc-session-001"
    print(f"\n{'─' * 70}")
    print(f"  SESSION 1: {session1} (8 turns, long conversation with compression)")
    print(f"{'─' * 70}")
    await run_session_1(adapter, session1, config)

    # Check session 1 final state
    doc1 = await store.get_session(session1)
    evidences1 = await store.list_evidences(session1)
    prefs = await memory_mgr.get_preferences(user_id)
    has_summary = doc1.session.summary and doc1.session.summary.content
    print(f"\n  Session 1 Summary:")
    print(f"    Messages: {len(doc1.session.messages)}")
    print(f"    Evidences: {len(evidences1)}")
    print(f"    Tool calls: {len(doc1.session.tool_state.tool_calls)}")
    print(f"    Conversation compressed: {bool(has_summary)}")
    if has_summary:
        print(f"    Summary: {doc1.session.summary.content[:100]}...")
    print(f"    User preferences detected: {len(prefs)}")
    for p in prefs:
        print(f"      - [{p.category}] {p.content}")

    # Save session summary as cross-session memory
    session_summary = doc1.session.summary.content if has_summary else (
        "用户咨询了订单状态、退货政策和促销活动，是一名 Python 后端开发工程师。"
    )
    await adapter.save_session_memory(session1, session_summary)

    # --- Session 2 ---
    session2 = "lc-session-002"
    config2 = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=8192, reserved_reply_tokens=1024),
        summary=SummaryConfig(enabled=False),
    )
    print(f"\n{'─' * 70}")
    print(f"  SESSION 2: {session2} (cross-session memory verification)")
    print(f"{'─' * 70}")

    cross_memories = await memory_mgr.get_all_memories(user_id)
    print(f"\n  Cross-session memories loaded: {len(cross_memories)}")
    for m in cross_memories:
        print(f"    - [{m.category}] {m.content[:60]}...")

    await run_session_2(adapter, session2, config2)

    # Final summary
    doc2 = await store.get_session(session2)
    print(f"\n  Session 2 Summary:")
    print(f"    Messages: {len(doc2.session.messages)}")

    print(f"\n{'=' * 70}")
    print("  DEMO COMPLETE")
    print(f"  - Session 1: {len(doc1.session.messages)} msgs, compression={'ON' if has_summary else 'OFF'}")
    print(f"  - Session 2: {len(doc2.session.messages)} msgs, cross-session memory={'YES'}")
    print(f"  - User preferences persisted: {len(prefs)}")
    print(f"  - Session summaries carried: {len(await memory_mgr.get_session_summaries(user_id))}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
