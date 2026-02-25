"""
LangChain 集成示例
==================

演示如何将 Context Engineering SDK 与 LangChain 框架集成：
- 通过 LangChain 的 Callback 机制接入 SDK
- 在 LangChain Chain/Agent 运行前注入结构化上下文
- 将 LangChain 的工具调用、检索结果、模型用量记录到 SDK
- 实现多轮对话 + 工具使用的完整链路

注意：此示例使用模拟的 LangChain 接口来演示集成模式。
实际使用时需安装 langchain 包并替换为真实实现。

运行方式:
    python examples/langchain_integration.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

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
    Evidence,
    EvidenceLinks,
    EvidenceSource,
    EvidenceType,
    ModelUsage,
    ModelUsageStage,
    ModelUsageStatus,
    Ref,
    SourceKind,
    ToolCall,
    ToolCallStatus,
    ToolProvider,
    ProviderKind,
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# 模拟 LangChain 组件（实际使用时替换为真实的 LangChain 导入）
# ---------------------------------------------------------------------------

@dataclass
class LangChainMessage:
    """模拟 langchain_core.messages.BaseMessage"""
    role: str
    content: str
    additional_kwargs: dict = field(default_factory=dict)


@dataclass
class LangChainToolCall:
    """模拟 LangChain 工具调用结果"""
    name: str
    args: dict
    output: str
    run_id: str = ""


@dataclass
class LangChainLLMResult:
    """模拟 LangChain LLM 调用结果"""
    content: str
    model: str = "gpt-4o"
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 150, "completion_tokens": 80, "total_tokens": 230
    })


@dataclass
class LangChainDocument:
    """模拟 langchain_core.documents.Document"""
    page_content: str
    metadata: dict = field(default_factory=dict)


class MockLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="summary", model="mock")


# ---------------------------------------------------------------------------
# LangChain <-> CE SDK 适配器
# ---------------------------------------------------------------------------

class LangChainCEAdapter:
    """
    将 LangChain 的生命周期事件映射到 Context Engineering SDK。

    核心映射：
    - LangChain Message -> CE Message
    - LangChain ToolCall -> CE ToolCall + Evidence
    - LangChain Document (Retriever) -> CE Evidence[type=rag_doc]
    - LangChain LLMResult -> CE ModelUsage + Evidence[type=llm_output]
    """

    def __init__(self, engine, store, ingestor):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()

    def map_message(self, lc_msg: LangChainMessage) -> Message:
        """LangChain Message -> CE Message"""
        role_map = {
            "system": Role.SYSTEM,
            "human": Role.USER,
            "user": Role.USER,
            "ai": Role.ASSISTANT,
            "assistant": Role.ASSISTANT,
            "tool": Role.TOOL,
        }
        return Message(
            role=role_map.get(lc_msg.role, Role.USER),
            content=lc_msg.content,
            at=self._clock.now_iso(),
        )

    async def on_user_message(self, session_id: str, lc_msg: LangChainMessage):
        """在 LangChain 链开始前，准备上下文。"""
        ce_msg = self.map_message(lc_msg)
        result = await self._engine.prepare_turn(
            session_id=session_id,
            user_message=ce_msg,
            runtime_config=RuntimeConfig(
                summary=SummaryConfig(enabled=False),
            ),
        )
        return result

    async def on_retriever_result(
        self, session_id: str, documents: list[LangChainDocument]
    ) -> list[Evidence]:
        """将 LangChain Retriever 的命中文档落为证据。"""
        evidences = []
        for doc in documents:
            ev, _ = await self._ingestor.ingest(
                session_id=session_id,
                content=doc.page_content,
                source=EvidenceSource(
                    kind=SourceKind.RAG,
                    name="langchain_retriever",
                    uri=doc.metadata.get("source", ""),
                ),
                evidence_type=EvidenceType.RAG_DOC,
                metadata=doc.metadata,
                options=IngestOptions(redact=False, dedup=True),
            )
            evidences.append(ev)
        return evidences

    async def on_tool_call(
        self, session_id: str, lc_tool_call: LangChainToolCall
    ) -> tuple[ToolCall, Evidence]:
        """记录 LangChain 工具调用和结果。"""
        tc_id = lc_tool_call.run_id or self._id_gen.generate()

        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=lc_tool_call.output,
            source=EvidenceSource(
                kind=SourceKind.TOOL,
                name=lc_tool_call.name,
            ),
            evidence_type=EvidenceType.TOOL_RESULT,
            links=EvidenceLinks(tool_call_id=tc_id),
            options=IngestOptions(redact=False),
        )

        tc = ToolCall(
            tool_call_id=tc_id,
            tool=lc_tool_call.name,
            provider=ToolProvider(kind=ProviderKind.BUILTIN, name="langchain"),
            called_at=self._clock.now_iso(),
            args_digest=lc_tool_call.args,
            status=ToolCallStatus.SUCCESS,
            result_evidence_ids=[ev.evidence_id],
        )
        await self._engine.record_tool_call(session_id, tc)
        return tc, ev

    async def on_llm_result(
        self, session_id: str, lc_result: LangChainLLMResult
    ):
        """记录 LLM 调用结果并提交助手消息。"""
        usage_id = self._id_gen.generate()

        # LLM 输出作为证据
        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=lc_result.content,
            source=EvidenceSource(kind=SourceKind.LLM, name=lc_result.model),
            evidence_type=EvidenceType.LLM_OUTPUT,
            links=EvidenceLinks(model_usage_id=usage_id),
            options=IngestOptions(redact=False),
        )

        # 记录模型用量
        mu = ModelUsage(
            model_usage_id=usage_id,
            provider="openai",
            model=lc_result.model,
            stage=ModelUsageStage.ANSWER,
            prompt_tokens=lc_result.usage.get("prompt_tokens", 0),
            completion_tokens=lc_result.usage.get("completion_tokens", 0),
            total_tokens=lc_result.usage.get("total_tokens", 0),
        )
        await self._engine.record_model_usage(
            session_id, mu, llm_output_evidence_id=ev.evidence_id
        )

        # 提交助手消息
        await self._engine.commit_assistant_message(
            session_id,
            Message(role=Role.ASSISTANT, content=lc_result.content),
            refs=[Ref(evidence_id=ev.evidence_id)],
        )

        return mu, ev


# ---------------------------------------------------------------------------
# 模拟 LangChain Agent 执行
# ---------------------------------------------------------------------------

async def simulate_langchain_agent(adapter: LangChainCEAdapter, session_id: str):
    """模拟一个 LangChain ReAct Agent 的执行流程。"""

    # --- 第 1 轮：简单对话 ---
    print("\n--- 第 1 轮：简单对话 ---")
    user_msg = LangChainMessage(role="human", content="你好，我想了解一下订单的状态")
    result = await adapter.on_user_message(session_id, user_msg)
    print(f"[LangChain 用户]: {user_msg.content}")
    print(f"  [CE SDK] Token 预算: {result.report.token_budget}, 已用: {result.report.token_used}")

    # Agent 直接回复
    llm_result = LangChainLLMResult(
        content="好的，请提供您的订单号，我来帮您查询。"
    )
    await adapter.on_llm_result(session_id, llm_result)
    print(f"[LangChain 助手]: {llm_result.content}")

    # --- 第 2 轮：检索增强 + 工具调用 ---
    print("\n--- 第 2 轮：RAG + 工具调用 ---")
    user_msg2 = LangChainMessage(
        role="human", content="订单号是 ORD-2024-001，另外帮我查查退货政策"
    )
    result2 = await adapter.on_user_message(session_id, user_msg2)
    print(f"[LangChain 用户]: {user_msg2.content}")

    # 模拟 Retriever 返回文档
    docs = [
        LangChainDocument(
            page_content="退货政策：自收货之日起 7 天内可无理由退货，30 天内可换货。"
                         "退货商品需保持原包装完好。运费由买家承担，质量问题除外。",
            metadata={"source": "policy_doc_v3", "doc_version": "2026-01"},
        ),
        LangChainDocument(
            page_content="VIP 用户享有 15 天无理由退货、免运费退换货特权。",
            metadata={"source": "vip_policy", "doc_version": "2026-02"},
        ),
    ]
    rag_evidences = await adapter.on_retriever_result(session_id, docs)
    print(f"  [RAG] 检索到 {len(rag_evidences)} 个文档")
    for ev in rag_evidences:
        print(f"    - {ev.evidence_id[:8]}... content={ev.content[:40]}...")

    # 模拟工具调用查询订单
    tool_call = LangChainToolCall(
        name="query_order",
        args={"order_id": "ORD-2024-001"},
        output=json.dumps({
            "order_id": "ORD-2024-001",
            "status": "delivered",
            "product": "智能手表",
            "delivered_at": "2026-02-20",
        }, ensure_ascii=False),
        run_id="run-lc-001",
    )
    tc, tool_ev = await adapter.on_tool_call(session_id, tool_call)
    print(f"  [工具] {tool_call.name}({tool_call.args}) -> {tool_call.output[:50]}...")

    # Agent 生成最终回复
    llm_result2 = LangChainLLMResult(
        content=(
            "您的订单 ORD-2024-001（智能手表）已于 2026-02-20 签收。\n\n"
            "关于退货政策：自收货之日起 7 天内可无理由退货，30 天内可换货。"
            "如果您是 VIP 用户，还可享受 15 天无理由退货和免运费退换。"
        ),
        usage={"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
    )
    await adapter.on_llm_result(session_id, llm_result2)
    print(f"[LangChain 助手]: {llm_result2.content[:80]}...")

    # --- 第 3 轮：验证上下文积累 ---
    print("\n--- 第 3 轮：验证上下文积累 ---")
    user_msg3 = LangChainMessage(role="human", content="那我想退货，应该怎么操作？")
    result3 = await adapter.on_user_message(session_id, user_msg3)
    print(f"[LangChain 用户]: {user_msg3.content}")
    print(f"  [CE SDK] 上下文 token: {result3.assembled_input.total_tokens}")
    print(f"  [CE SDK] 裁剪决策数: {len(result3.report.prune_decisions)}")

    assembled_text = result3.assembled_input.text or ""
    print(f"  [验证] 上下文包含订单信息: {'ORD-2024-001' in assembled_text}")
    print(f"  [验证] 上下文包含退货政策: {'退货' in assembled_text}")

    llm_result3 = LangChainLLMResult(
        content="根据退货政策，您的订单在 7 天内可以无理由退货...",
    )
    await adapter.on_llm_result(session_id, llm_result3)
    print(f"[LangChain 助手]: {llm_result3.content}")


async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )

    adapter = LangChainCEAdapter(engine=engine, store=store, ingestor=ingestor)
    session_id = "langchain-demo-001"

    print("=" * 60)
    print("  Context Engineering SDK — LangChain 集成示例")
    print("=" * 60)

    await simulate_langchain_agent(adapter, session_id)

    # 最终状态
    print("\n" + "=" * 60)
    print("  最终状态汇总")
    print("=" * 60)
    doc = await store.get_session(session_id)
    evidences = await store.list_evidences(session_id)
    print(f"  消息数: {len(doc.session.messages)}")
    print(f"  证据总数: {len(evidences)}")
    rag_count = sum(1 for e in evidences if e.type == EvidenceType.RAG_DOC)
    tool_count = sum(1 for e in evidences if e.type == EvidenceType.TOOL_RESULT)
    llm_count = sum(1 for e in evidences if e.type == EvidenceType.LLM_OUTPUT)
    print(f"    RAG 文档: {rag_count}")
    print(f"    工具结果: {tool_count}")
    print(f"    LLM 输出: {llm_count}")
    print(f"  工具调用数: {len(doc.session.tool_state.tool_calls)}")
    print(f"  模型用量记录数: {len(doc.session.model_usage)}")
    total_tokens = sum(mu.total_tokens for mu in doc.session.model_usage)
    print(f"  累计 token 消耗: {total_tokens}")

    print("\n  --- 以上数据展示了 LangChain 的完整运行过程如何被 ---")
    print("  --- CE SDK 结构化记录，实现可审计、可追溯、可回放 ---")


if __name__ == "__main__":
    asyncio.run(main())
