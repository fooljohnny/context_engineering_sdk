"""
流式对话与模型用量记录示例
==========================

演示 SDK 的流式回复支持和模型用量记录：
- commit_assistant_chunk() 逐 chunk 提交
- finalize_assistant_message() 合并 chunks 为完整消息
- record_model_usage() 记录 token 消耗与延迟
- 事件总线监控流式过程

运行方式:
    python examples/streaming_conversation.py
"""

import asyncio
import time

from context_engineering_sdk import (
    Message,
    Role,
    RuntimeConfig,
    BudgetConfig,
    SummaryConfig,
    create_context_engine,
)
from context_engineering_sdk.core.id_generator import UuidV4Generator
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
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.observability.event_bus import Event, InMemoryEventBus
from context_engineering_sdk.store.memory import MemoryStore


class MockLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="summary", model="mock")


class MockStreamingLlm:
    """模拟流式 LLM 输出。"""

    def __init__(self):
        self.provider_name = "mock-provider"
        self.model = "mock-gpt-4o"

    async def stream(self, assembled_input):
        """模拟流式返回 chunks。"""
        full_text = (
            "Context Engineering SDK 是一套用于结构化管理 AI 上下文的工具包。"
            "它的核心理念是将散乱的 prompt 文本转变为可裁剪、可追溯、可治理的结构化单元。"
            "主要包含以下组件：\n"
            "1. **证据入库（Ingestor）** - 统一落证据，支持去重与脱敏\n"
            "2. **块派生（Deriver）** - 从会话和证据派生上下文块\n"
            "3. **策略裁剪（Pruner）** - 在 token 预算内智能裁剪\n"
            "4. **渲染装配（Assembler）** - 生成最终模型输入"
        )
        # 按字符分 chunk（实际中 LLM 按 token 分）
        chunk_size = 20
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            await asyncio.sleep(0.02)  # 模拟网络延迟
            yield i // chunk_size, chunk

        self._last_response_text = full_text
        self._last_total_tokens = 350


async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    event_bus = InMemoryEventBus()
    streaming_llm = MockStreamingLlm()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    engine._event_bus = event_bus

    # 注册事件监听器
    chunk_events = []
    def on_chunk(event: Event):
        chunk_events.append(event)

    def on_finalize(event: Event):
        print(f"  [事件] {event.event_type}: "
              f"content_length={event.payload.get('content_length', '?')}")

    event_bus.on("AssistantChunkReceived", on_chunk)
    event_bus.on("AssistantMessageFinalized", on_finalize)

    session_id = "demo-streaming-001"
    config = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=4096, reserved_reply_tokens=1024),
        summary=SummaryConfig(enabled=False),
    )

    print("=" * 60)
    print("  Context Engineering SDK — 流式对话示例")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Phase A: 准备输入
    # ---------------------------------------------------------------
    user_msg = Message(role=Role.USER, content="请详细介绍一下 Context Engineering SDK")
    result = await engine.prepare_turn(session_id, user_msg, config)
    print(f"\n[用户]: {user_msg.content}")
    print(f"  [装配] 总 token 估算: {result.assembled_input.total_tokens}")

    # ---------------------------------------------------------------
    # Phase B: 流式输出 + chunk 提交
    # ---------------------------------------------------------------
    print(f"\n[助手]: ", end="", flush=True)
    start_time = time.monotonic()
    first_token_time = None
    chunk_count = 0

    async for chunk_index, chunk_text in streaming_llm.stream(result.assembled_input):
        if first_token_time is None:
            first_token_time = time.monotonic()
        print(chunk_text, end="", flush=True)
        await engine.commit_assistant_chunk(session_id, chunk_text, chunk_index)
        chunk_count += 1

    end_time = time.monotonic()
    print()  # newline after streaming output

    # ---------------------------------------------------------------
    # Phase C: 最终化消息
    # ---------------------------------------------------------------
    put_result = await engine.finalize_assistant_message(session_id)
    print(f"\n  [流式] Chunks 数量: {chunk_count}")
    print(f"  [流式] 首 token 延迟: {int((first_token_time - start_time) * 1000)}ms")
    print(f"  [流式] 总延迟: {int((end_time - start_time) * 1000)}ms")
    print(f"  [流式] 消息已持久化, version={put_result.version}")

    # ---------------------------------------------------------------
    # Phase D: 记录模型用量
    # ---------------------------------------------------------------
    usage_id = id_gen.generate()

    # 将 LLM 输出作为证据存储
    llm_output_ev, _ = await ingestor.ingest(
        session_id=session_id,
        content=streaming_llm._last_response_text,
        source=EvidenceSource(
            kind=SourceKind.LLM,
            name=streaming_llm.provider_name,
        ),
        evidence_type=EvidenceType.LLM_OUTPUT,
        links=EvidenceLinks(model_usage_id=usage_id),
        options=IngestOptions(redact=False),
    )

    model_usage = ModelUsage(
        model_usage_id=usage_id,
        provider=streaming_llm.provider_name,
        model=streaming_llm.model,
        stage=ModelUsageStage.ANSWER,
        prompt_tokens=result.assembled_input.total_tokens,
        completion_tokens=200,
        total_tokens=streaming_llm._last_total_tokens,
        first_token_latency_ms=int((first_token_time - start_time) * 1000),
        latency_ms=int((end_time - start_time) * 1000),
        status=ModelUsageStatus.SUCCESS,
    )
    await engine.record_model_usage(
        session_id, model_usage,
        llm_output_evidence_id=llm_output_ev.evidence_id,
    )
    print(f"\n  [用量] 已记录模型用量:")
    print(f"    Provider: {model_usage.provider}")
    print(f"    Model: {model_usage.model}")
    print(f"    Prompt tokens: {model_usage.prompt_tokens}")
    print(f"    Completion tokens: {model_usage.completion_tokens}")
    print(f"    Total tokens: {model_usage.total_tokens}")
    print(f"    First token latency: {model_usage.first_token_latency_ms}ms")
    print(f"    Total latency: {model_usage.latency_ms}ms")
    print(f"    LLM output evidence: {llm_output_ev.evidence_id[:8]}...")

    # ---------------------------------------------------------------
    # 查看事件历史
    # ---------------------------------------------------------------
    print(f"\n  [事件] 共产生 {len(event_bus.history)} 个事件:")
    event_types = {}
    for ev in event_bus.history:
        event_types[ev.event_type] = event_types.get(ev.event_type, 0) + 1
    for et, count in sorted(event_types.items()):
        print(f"    {et}: {count}")

    # 验证消息完整性
    doc = await store.get_session(session_id)
    assistant_msgs = [m for m in doc.session.messages if m.role == Role.ASSISTANT]
    print(f"\n  [验证] 助手消息数: {len(assistant_msgs)}")
    print(f"  [验证] 消息内容长度: {len(assistant_msgs[0].content)} 字符")
    print(f"  [验证] 模型用量记录数: {len(doc.session.model_usage)}")


if __name__ == "__main__":
    asyncio.run(main())
