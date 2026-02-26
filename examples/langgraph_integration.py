"""
LangGraph + Context Engineering SDK 完整集成示例
=================================================

演示功能：
- LangGraph 状态图（StateGraph）与 CE SDK 的集成
- 基于图节点的 Router → Retriever → ToolExecutor → Generator 工作流
- 跨会话用户画像识别与长期记忆（MemoryManager）
- 多轮长对话摘要压缩
- 图节点执行过程的可观测性（EventBus）

场景说明：
  会话 1（6 轮）：用户与技术助手对话，图工作流根据意图自动路由到
  不同处理节点（检索 / 工具 / 直接回复），同时提取用户偏好。
  会话 2（3 轮）：新会话加载用户偏好与会话摘要，验证跨会话记忆。

运行方式：
    python examples/langgraph_integration.py
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum

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
from context_engineering_sdk.observability.event_bus import InMemoryEventBus
from context_engineering_sdk.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class SmartMockLlm:
    def __init__(self):
        self._call_count = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self._call_count += 1
        prompt_text = " ".join(p.content for p in request.messages)
        if "summarize" in prompt_text.lower() or "摘要" in prompt_text:
            return LlmResponse(
                content=(
                    "用户是一名熟悉 Python/Kubernetes 的全栈工程师，偏好中文简洁回答。"
                    "对话涵盖：K8s Pod 重启排查、Docker 网络配置、CI/CD 流水线设计。"
                    "用户对云原生技术栈有深入需求。"
                ),
                model="mock-gpt-4o",
                prompt_tokens=200, completion_tokens=80, total_tokens=280,
            )
        return LlmResponse(content="response", model="mock-gpt-4o",
                           prompt_tokens=100, completion_tokens=50, total_tokens=150)


# ---------------------------------------------------------------------------
# LangGraph-style State & Nodes
# ---------------------------------------------------------------------------

class RouteDecision(str, Enum):
    RETRIEVE = "retrieve"
    TOOL = "tool"
    DIRECT = "direct"


@dataclass
class GraphState:
    """Simulated LangGraph state shared across nodes."""
    user_query: str = ""
    route: RouteDecision = RouteDecision.DIRECT
    retrieved_docs: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    response: str = ""
    evidence_refs: list[str] = field(default_factory=list)


ROUTE_RULES = {
    "查": RouteDecision.TOOL,
    "查询": RouteDecision.TOOL,
    "执行": RouteDecision.TOOL,
    "运行": RouteDecision.TOOL,
    "什么是": RouteDecision.RETRIEVE,
    "如何": RouteDecision.RETRIEVE,
    "怎么": RouteDecision.RETRIEVE,
    "原理": RouteDecision.RETRIEVE,
    "解释": RouteDecision.RETRIEVE,
    "区别": RouteDecision.RETRIEVE,
}

KNOWLEDGE_BASE = {
    "kubernetes": "Kubernetes (K8s) 是一个开源容器编排平台。常见排错：kubectl describe pod 查看事件；kubectl logs 查看日志；OOMKilled 表示内存不足需调整 resources.limits。",
    "docker": "Docker 网络模式：bridge（默认）、host、none、overlay（多主机）。容器间通信推荐使用自定义 bridge 网络，通过容器名 DNS 解析。",
    "cicd": "CI/CD 流水线设计要点：1) 源码管理触发 2) 单元测试+lint 3) 构建容器镜像 4) 部署到 staging 5) 集成测试 6) 蓝绿/金丝雀发布到 production。",
}

TOOL_REGISTRY = {
    "kubectl_get_pods": lambda ns: json.dumps({
        "pods": [
            {"name": "api-server-7d8f9", "status": "Running", "restarts": 0},
            {"name": "worker-5c6d7", "status": "CrashLoopBackOff", "restarts": 15},
            {"name": "redis-0", "status": "Running", "restarts": 0},
        ]
    }, ensure_ascii=False),
    "docker_inspect": lambda cid: json.dumps({
        "container": cid, "network": "bridge",
        "ip": "172.17.0.5", "ports": {"8080/tcp": "0.0.0.0:8080"},
    }, ensure_ascii=False),
}

PREFERENCE_MAP = {
    "Python": ("tech_stack", "用户熟悉 Python"),
    "Kubernetes": ("tech_stack", "用户熟悉 Kubernetes"),
    "K8s": ("tech_stack", "用户熟悉 Kubernetes"),
    "Docker": ("tech_stack", "用户熟悉 Docker"),
    "全栈": ("domain", "用户是全栈工程师"),
    "简洁": ("style", "用户偏好简洁回答"),
    "中文": ("language", "用户偏好中文交流"),
}


# ---------------------------------------------------------------------------
# Graph Node Implementations
# ---------------------------------------------------------------------------

def router_node(state: GraphState) -> GraphState:
    """Route user query to the appropriate processing node."""
    for keyword, route in ROUTE_RULES.items():
        if keyword in state.user_query:
            state.route = route
            return state
    state.route = RouteDecision.DIRECT
    return state


def retriever_node(state: GraphState) -> GraphState:
    """Retrieve relevant documents from knowledge base."""
    query_lower = state.user_query.lower()
    for key, content in KNOWLEDGE_BASE.items():
        if key in query_lower or any(k in query_lower for k in key.split()):
            state.retrieved_docs.append({"source": key, "content": content})
    if not state.retrieved_docs:
        state.retrieved_docs.append({
            "source": "general",
            "content": "未找到精确匹配的文档，请提供更多上下文。",
        })
    return state


def tool_executor_node(state: GraphState) -> GraphState:
    """Execute tools based on user query."""
    query = state.user_query
    if "pod" in query.lower() or "k8s" in query.lower() or "kubernetes" in query.lower():
        result = TOOL_REGISTRY["kubectl_get_pods"]("default")
        state.tool_results.append({"tool": "kubectl_get_pods", "args": {"namespace": "default"}, "output": result})
    elif "docker" in query.lower() or "容器" in query.lower():
        result = TOOL_REGISTRY["docker_inspect"]("api-container")
        state.tool_results.append({"tool": "docker_inspect", "args": {"container_id": "api-container"}, "output": result})
    return state


def generator_node(state: GraphState, context_hint: str = "") -> GraphState:
    """Generate final response based on collected context."""
    parts = []
    if state.retrieved_docs:
        for doc in state.retrieved_docs:
            parts.append(doc["content"])
    if state.tool_results:
        for tr in state.tool_results:
            parts.append(f"[{tr['tool']}] {tr['output'][:100]}")
    if context_hint:
        parts.append(context_hint)
    state.response = " | ".join(parts) if parts else "好的，请告诉我更多细节。"
    return state


# ---------------------------------------------------------------------------
# LangGraph CE Adapter
# ---------------------------------------------------------------------------

class LangGraphCEAdapter:
    def __init__(self, engine, store, ingestor, memory_mgr, user_id):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._memory_mgr = memory_mgr
        self._user_id = user_id
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()

    async def run_graph(self, session_id, user_text, config):
        """Execute the full graph workflow with CE SDK integration."""
        memory_blocks = await self._memory_mgr.build_memory_blocks(self._user_id)
        config.extra_context_blocks = memory_blocks

        user_msg = Message(
            role=Role.USER, content=user_text,
            author=Author(kind=AuthorKind.USER, id=self._user_id),
        )
        prepare_result = await self._engine.prepare_turn(session_id, user_msg, config)

        for keyword, (cat, desc) in PREFERENCE_MAP.items():
            if keyword in user_text:
                existing = await self._memory_mgr.get_preferences(self._user_id)
                if not any(desc in p.content for p in existing):
                    await self._memory_mgr.add_preference(
                        self._user_id, cat, desc, session_id=session_id,
                    )

        state = GraphState(user_query=user_text)

        state = router_node(state)
        route_str = state.route.value
        print(f"      Router → {route_str}")

        evidence_refs = []
        if state.route == RouteDecision.RETRIEVE:
            state = retriever_node(state)
            for doc in state.retrieved_docs:
                ev, _ = await self._ingestor.ingest(
                    session_id=session_id,
                    content=doc["content"],
                    source=EvidenceSource(kind=SourceKind.RAG, name=doc["source"]),
                    evidence_type=EvidenceType.RAG_DOC,
                    options=IngestOptions(redact=False, dedup=True),
                )
                evidence_refs.append(Ref(evidence_id=ev.evidence_id))
            print(f"      Retriever → {len(state.retrieved_docs)} docs")

        elif state.route == RouteDecision.TOOL:
            state = tool_executor_node(state)
            for tr in state.tool_results:
                tc_id = self._id_gen.generate()
                ev, _ = await self._ingestor.ingest(
                    session_id=session_id,
                    content=tr["output"],
                    source=EvidenceSource(kind=SourceKind.TOOL, name=tr["tool"]),
                    evidence_type=EvidenceType.TOOL_RESULT,
                    links=EvidenceLinks(tool_call_id=tc_id),
                    options=IngestOptions(redact=False),
                )
                tc = ToolCall(
                    tool_call_id=tc_id, tool=tr["tool"],
                    provider=ToolProvider(kind=ProviderKind.BUILTIN, name="langgraph"),
                    called_at=self._clock.now_iso(),
                    args_digest=tr["args"],
                    status=ToolCallStatus.SUCCESS,
                    result_evidence_ids=[ev.evidence_id],
                )
                await self._engine.record_tool_call(session_id, tc)
                evidence_refs.append(Ref(evidence_id=ev.evidence_id))
            print(f"      ToolExecutor → {len(state.tool_results)} calls")

        state = generator_node(state)
        print(f"      Generator → response ({len(state.response)} chars)")

        await self._engine.commit_assistant_message(
            session_id,
            Message(role=Role.ASSISTANT, content=state.response,
                    author=Author(kind=AuthorKind.AGENT, id="langgraph-agent")),
            refs=evidence_refs or None,
        )

        return prepare_result, state


async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    llm_adapter = SmartMockLlm()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)
    user_memory_store = InMemoryUserMemoryStore()
    memory_mgr = MemoryManager(user_memory_store, id_generator=id_gen)
    event_bus = InMemoryEventBus()

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=llm_adapter,
    )
    engine._event_bus = event_bus

    user_id = "user-langgraph-demo"
    adapter = LangGraphCEAdapter(
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
    print("  LangGraph + CE SDK: StateGraph Workflow with Cross-Session Memory")
    print("=" * 70)

    # --- Session 1 ---
    session1 = "lg-session-001"
    print(f"\n{'─' * 70}")
    print(f"  SESSION 1: {session1} (6 turns with graph routing)")
    print(f"{'─' * 70}")

    session1_turns = [
        "你好，我是一名全栈工程师，熟悉 Python 和 Kubernetes，请用中文简洁回答",
        "什么是 Kubernetes 的 Pod 重启排查流程？",
        "帮我查一下当前 K8s 集群的 Pod 状态",
        "如何配置 Docker 容器间的网络通信？",
        "怎么设计一个完整的 CI/CD 流水线？",
        "好的，谢谢你的帮助！",
    ]

    for i, user_text in enumerate(session1_turns, 1):
        print(f"\n  [Turn {i}]")
        print(f"    User: {user_text}")
        result, state = await adapter.run_graph(session1, user_text, config)
        print(f"    Response: {state.response[:80]}{'...' if len(state.response) > 80 else ''}")
        print(f"    Token used: {result.report.token_used}")

    doc1 = await store.get_session(session1)
    evidences1 = await store.list_evidences(session1)
    prefs = await memory_mgr.get_preferences(user_id)
    has_summary = doc1.session.summary and doc1.session.summary.content

    print(f"\n  Session 1 Final State:")
    print(f"    Messages: {len(doc1.session.messages)}")
    print(f"    Evidences: {len(evidences1)}")
    print(f"    Tool calls: {len(doc1.session.tool_state.tool_calls)}")
    print(f"    Conversation compressed: {bool(has_summary)}")
    if has_summary:
        print(f"    Summary: {doc1.session.summary.content[:100]}...")
    print(f"    Preferences detected: {len(prefs)}")
    for p in prefs:
        print(f"      - [{p.category}] {p.content}")

    summary_text = doc1.session.summary.content if has_summary else (
        "用户是全栈工程师，讨论了 K8s 排错、Docker 网络、CI/CD 流水线设计。"
    )
    await memory_mgr.add_session_summary(user_id, session1, summary_text)

    # --- Session 2 ---
    session2 = "lg-session-002"
    config2 = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=8192, reserved_reply_tokens=1024),
        summary=SummaryConfig(enabled=False),
    )

    print(f"\n{'─' * 70}")
    print(f"  SESSION 2: {session2} (cross-session memory verification)")
    print(f"{'─' * 70}")

    cross_memories = await memory_mgr.get_all_memories(user_id)
    print(f"\n  Cross-session memories: {len(cross_memories)}")
    for m in cross_memories:
        print(f"    - [{m.category}] {m.content[:60]}...")

    session2_turns = [
        "你好，继续上次的话题",
        "如何监控 Kubernetes 集群的健康状态？",
        "谢谢，今天先到这里",
    ]

    for i, user_text in enumerate(session2_turns, 1):
        print(f"\n  [Turn {i}]")
        print(f"    User: {user_text}")
        result, state = await adapter.run_graph(session2, user_text, config2)
        print(f"    Response: {state.response[:80]}{'...' if len(state.response) > 80 else ''}")

        assembled = result.assembled_input.text or ""
        has_pref = "Preferences" in assembled or "偏好" in assembled
        has_hist = "Session Summaries" in assembled or "K8s" in assembled
        print(f"    Memory present: preferences={has_pref}, history={has_hist}")

    # Event summary
    event_types = {}
    for ev in event_bus.history:
        event_types[ev.event_type] = event_types.get(ev.event_type, 0) + 1

    print(f"\n{'─' * 70}")
    print(f"  Observability: {len(event_bus.history)} events emitted")
    for et, count in sorted(event_types.items()):
        print(f"    {et}: {count}")

    print(f"\n{'=' * 70}")
    print("  DEMO COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
