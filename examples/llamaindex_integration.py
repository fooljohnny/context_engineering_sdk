"""
LlamaIndex 集成示例
===================

演示如何将 Context Engineering SDK 与 LlamaIndex 框架集成：
- 将 LlamaIndex 的 Node/Document 检索结果作为证据入库
- 记录 Response Synthesis 的中间产物
- 通过 CE SDK 管理多轮 RAG 对话的上下文
- 实现检索结果的版本/时效治理

注意：此示例使用模拟的 LlamaIndex 接口来演示集成模式。
实际使用时需安装 llama-index 包并替换为真实实现。

运行方式:
    python examples/llamaindex_integration.py
"""

import asyncio
import json
from dataclasses import dataclass, field

from context_engineering_sdk import (
    Message,
    Role,
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


class MockLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="summary", model="mock")


# ---------------------------------------------------------------------------
# 模拟 LlamaIndex 核心对象
# ---------------------------------------------------------------------------

@dataclass
class LlamaIndexNode:
    """模拟 llama_index.core.schema.TextNode"""
    node_id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class LlamaIndexResponse:
    """模拟 llama_index.core.response.Response"""
    response: str
    source_nodes: list[LlamaIndexNode] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LlamaIndex <-> CE SDK 适配器
# ---------------------------------------------------------------------------

class LlamaIndexCEAdapter:
    """
    将 LlamaIndex 的检索和生成流程映射到 CE SDK。

    核心映射：
    - LlamaIndex Node -> CE Evidence[type=rag_doc]
    - LlamaIndex retriever 调用 -> CE ToolCall (tool=rag_search)
    - LlamaIndex Response -> CE Evidence[type=llm_output] + Message
    - Node metadata -> CE Evidence metadata (doc_version, effective_at, etc.)
    """

    def __init__(self, engine, store, ingestor):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()

    async def on_retrieval(
        self,
        session_id: str,
        query: str,
        nodes: list[LlamaIndexNode],
        retriever_name: str = "default_retriever",
        duration_ms: int = 0,
    ) -> tuple[ToolCall, list[Evidence]]:
        """
        将 LlamaIndex 的检索结果映射为 CE SDK 的工具调用 + 证据。
        把 rag_search 视为工具调用，命中的 Node 各自落为 evidence[type=rag_doc]。
        """
        tc_id = self._id_gen.generate()
        evidences = []

        for node in nodes:
            ev, _ = await self._ingestor.ingest(
                session_id=session_id,
                content=node.text,
                source=EvidenceSource(
                    kind=SourceKind.RAG,
                    name=retriever_name,
                    uri=node.metadata.get("source_uri", ""),
                ),
                evidence_type=EvidenceType.RAG_DOC,
                links=EvidenceLinks(tool_call_id=tc_id),
                metadata={
                    "node_id": node.node_id,
                    "score": node.score,
                    "doc_version": node.metadata.get("doc_version", ""),
                    "effective_at": node.metadata.get("effective_at", ""),
                    "domain": node.metadata.get("domain", ""),
                    **{k: v for k, v in node.metadata.items()
                       if k not in ("doc_version", "effective_at", "domain", "source_uri")},
                },
                options=IngestOptions(
                    redact=False,
                    dedup=True,
                    confidence=node.score,
                ),
            )
            evidences.append(ev)

        tc = ToolCall(
            tool_call_id=tc_id,
            tool="rag_search",
            provider=ToolProvider(
                kind=ProviderKind.BUILTIN,
                name=f"llamaindex:{retriever_name}",
            ),
            called_at=self._clock.now_iso(),
            args_digest={"query": query, "top_k": len(nodes)},
            status=ToolCallStatus.SUCCESS,
            duration_ms=duration_ms,
            result_evidence_ids=[ev.evidence_id for ev in evidences],
        )
        await self._engine.record_tool_call(session_id, tc)

        return tc, evidences

    async def on_synthesis(
        self,
        session_id: str,
        response: LlamaIndexResponse,
        source_evidence_ids: list[str],
    ):
        """将 LlamaIndex 的 Response Synthesis 结果记录到 CE SDK。"""
        usage_id = self._id_gen.generate()

        # LLM 输出作为证据
        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=response.response,
            source=EvidenceSource(kind=SourceKind.LLM, name="llamaindex_synthesizer"),
            evidence_type=EvidenceType.LLM_OUTPUT,
            links=EvidenceLinks(model_usage_id=usage_id),
            options=IngestOptions(redact=False),
        )

        # 记录模型用量
        mu = ModelUsage(
            model_usage_id=usage_id,
            provider="openai",
            model="gpt-4o",
            stage=ModelUsageStage.ANSWER,
            total_tokens=response.metadata.get("total_tokens", 0),
        )
        await self._engine.record_model_usage(
            session_id, mu, llm_output_evidence_id=ev.evidence_id
        )

        # 构建引用列表：引用所有源证据
        refs = [Ref(evidence_id=eid) for eid in source_evidence_ids]
        refs.append(Ref(evidence_id=ev.evidence_id))

        # 提交助手消息
        await self._engine.commit_assistant_message(
            session_id,
            Message(role=Role.ASSISTANT, content=response.response),
            refs=refs,
        )

        return mu, ev


# ---------------------------------------------------------------------------
# 模拟知识库
# ---------------------------------------------------------------------------

def simulate_retrieval(query: str) -> list[LlamaIndexNode]:
    """模拟 LlamaIndex 检索器返回 Nodes。"""
    knowledge_base = {
        "context engineering": [
            LlamaIndexNode(
                node_id="node-001",
                text=(
                    "上下文工程（Context Engineering）是一种将 AI 系统中的上下文信息"
                    "进行结构化建模的方法论。它将散乱的 prompt 文本变成可裁剪、可复用、"
                    "可解释的结构化单元，包括会话消息、证据和上下文块三个核心概念。"
                ),
                score=0.95,
                metadata={
                    "source_uri": "https://docs.example.com/ce/overview",
                    "doc_version": "v2.0",
                    "effective_at": "2026-01-01",
                    "domain": "AI",
                },
            ),
            LlamaIndexNode(
                node_id="node-002",
                text=(
                    "Context Engineering SDK 提供了完整的上下文构建流水线：\n"
                    "- Ingestor: 证据入库（去重、脱敏、置信度评估）\n"
                    "- Deriver: 从会话和证据派生上下文块\n"
                    "- Pruner: 在 token 预算内智能裁剪\n"
                    "- Renderer: 解析引用并渲染上下文块\n"
                    "- Assembler: 装配最终模型输入"
                ),
                score=0.88,
                metadata={
                    "source_uri": "https://docs.example.com/ce/sdk",
                    "doc_version": "v1.0",
                    "effective_at": "2026-02-01",
                    "domain": "SDK",
                },
            ),
        ],
        "evidence management": [
            LlamaIndexNode(
                node_id="node-003",
                text=(
                    "证据管理是 Context Engineering 的核心能力之一。所有来源的信息"
                    "（RAG 检索、工具调用、LLM 输出、用户输入）统一落为 Evidence 对象，"
                    "带有 source、confidence、metadata 和 links 信息，实现统一溯源。"
                ),
                score=0.92,
                metadata={
                    "source_uri": "https://docs.example.com/ce/evidence",
                    "doc_version": "v2.0",
                    "effective_at": "2026-01-15",
                    "domain": "Evidence",
                },
            ),
        ],
    }

    for key, nodes in knowledge_base.items():
        if key in query.lower():
            return nodes
    return knowledge_base.get("context engineering", [])[:1]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )

    adapter = LlamaIndexCEAdapter(engine=engine, store=store, ingestor=ingestor)
    session_id = "llamaindex-demo-001"
    config = RuntimeConfig(summary=SummaryConfig(enabled=False))

    print("=" * 60)
    print("  Context Engineering SDK — LlamaIndex 集成示例")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 第 1 轮：RAG 检索 + 回答
    # ---------------------------------------------------------------
    print("\n--- 第 1 轮：什么是 Context Engineering ---")
    query1 = "什么是 Context Engineering？"
    print(f"[用户]: {query1}")

    result1 = await engine.prepare_turn(
        session_id,
        Message(role=Role.USER, content=query1),
        config,
    )

    # 模拟检索
    nodes = simulate_retrieval("context engineering")
    tc, evidences = await adapter.on_retrieval(
        session_id, query1, nodes,
        retriever_name="vector_retriever",
        duration_ms=45,
    )
    print(f"  [检索] 命中 {len(nodes)} 个文档节点:")
    for node in nodes:
        print(f"    - [{node.score:.2f}] {node.text[:50]}... (v={node.metadata.get('doc_version')})")

    # 模拟生成回复
    response1 = LlamaIndexResponse(
        response=(
            "Context Engineering（上下文工程）是一种将 AI 系统中的上下文信息"
            "进行结构化建模的方法论。SDK 提供了完整的流水线，包括证据入库、"
            "块派生、策略裁剪、渲染装配等核心组件。"
        ),
        source_nodes=nodes,
        metadata={"total_tokens": 280},
    )
    await adapter.on_synthesis(
        session_id, response1,
        source_evidence_ids=[ev.evidence_id for ev in evidences],
    )
    print(f"[助手]: {response1.response[:80]}...")

    # ---------------------------------------------------------------
    # 第 2 轮：深入询问证据管理
    # ---------------------------------------------------------------
    print("\n--- 第 2 轮：证据管理 ---")
    query2 = "SDK 是如何做 evidence management 的？"
    print(f"[用户]: {query2}")

    result2 = await engine.prepare_turn(
        session_id,
        Message(role=Role.USER, content=query2),
        config,
    )

    nodes2 = simulate_retrieval("evidence management")
    tc2, evidences2 = await adapter.on_retrieval(
        session_id, query2, nodes2,
        retriever_name="vector_retriever",
        duration_ms=38,
    )
    print(f"  [检索] 命中 {len(nodes2)} 个文档节点")

    response2 = LlamaIndexResponse(
        response=(
            "SDK 的证据管理采用统一的 Evidence 对象模型。所有来源的信息"
            "（RAG、工具、LLM、用户输入）都先落为证据再引用。"
            "每条证据带有 source、confidence、metadata 和 links 信息。"
        ),
        source_nodes=nodes2,
        metadata={"total_tokens": 200},
    )
    await adapter.on_synthesis(
        session_id, response2,
        source_evidence_ids=[ev.evidence_id for ev in evidences2],
    )
    print(f"[助手]: {response2.response[:80]}...")

    # ---------------------------------------------------------------
    # 第 3 轮：验证上下文积累（前两轮的证据参与装配）
    # ---------------------------------------------------------------
    print("\n--- 第 3 轮：验证上下文积累 ---")
    query3 = "总结一下我们讨论的内容"
    print(f"[用户]: {query3}")

    result3 = await engine.prepare_turn(
        session_id,
        Message(role=Role.USER, content=query3),
        config,
    )
    print(f"  [装配] 总 token 估算: {result3.assembled_input.total_tokens}")
    print(f"  [报告] 裁剪决策数: {len(result3.report.prune_decisions)}")

    # ---------------------------------------------------------------
    # 最终状态
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  最终状态汇总")
    print("=" * 60)

    doc = await store.get_session(session_id)
    all_evidences = await store.list_evidences(session_id)

    print(f"  消息数: {len(doc.session.messages)}")
    print(f"  证据总数: {len(all_evidences)}")

    # 按类型统计证据
    type_counts = {}
    for ev in all_evidences:
        type_counts[ev.type.value] = type_counts.get(ev.type.value, 0) + 1
    print("  证据类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # 展示证据的文档版本信息（时效治理）
    rag_evidences = [ev for ev in all_evidences if ev.type == EvidenceType.RAG_DOC]
    if rag_evidences:
        print("\n  RAG 文档版本治理:")
        for ev in rag_evidences:
            version = ev.metadata.get("doc_version", "unknown")
            effective = ev.metadata.get("effective_at", "unknown")
            score = ev.metadata.get("score", 0)
            domain = ev.metadata.get("domain", "unknown")
            print(f"    - [{domain}] v={version} effective_at={effective} "
                  f"score={score:.2f} content={ev.content[:40]}...")

    # 展示工具调用（检索记录）
    print(f"\n  检索记录 (tool_calls):")
    for tc in doc.session.tool_state.tool_calls:
        args = tc.args_digest
        ev_count = len(tc.result_evidence_ids)
        print(f"    - {tc.tool} query=\"{args.get('query', '?')}\" "
              f"top_k={args.get('top_k', '?')} -> {ev_count} evidences "
              f"({tc.duration_ms}ms)")

    print("\n  --- 以上展示了 LlamaIndex 检索流程如何被 CE SDK 结构化记录 ---")
    print("  --- RAG 文档带有版本和时效信息，支持版本治理与冲突检测 ---")


if __name__ == "__main__":
    asyncio.run(main())
