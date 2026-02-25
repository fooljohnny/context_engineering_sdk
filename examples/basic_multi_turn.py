"""
基础多轮对话示例
================

演示如何使用 Context Engineering SDK 管理多轮对话，包括：
- 创建 ContextEngine 并初始化会话
- 多轮 prepare_turn -> commit_assistant_message 交互
- 通过 Report 观察每轮的 token 预算使用与裁剪决策
- 会话摘要自动触发（当消息数超过阈值时）

运行方式:
    python examples/basic_multi_turn.py
"""

import asyncio

from context_engineering_sdk import (
    Message,
    Role,
    Author,
    AuthorKind,
    PrepareResult,
    RuntimeConfig,
    BudgetConfig,
    SummaryConfig,
    create_context_engine,
)
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore
from context_engineering_sdk.observability.event_bus import InMemoryEventBus


# ---------------------------------------------------------------------------
# 1. 准备可替换的 LLM Adapter（此处使用 Mock 实现）
# ---------------------------------------------------------------------------

class MockLlmAdapter:
    """
    模拟 LLM 适配器。
    实际使用时替换为调用 OpenAI / Azure / 本地模型的实现。
    """

    RESPONSES = [
        "你好！我是你的 AI 助手，有什么可以帮你的吗？",
        "Context Engineering 是一种将 AI 系统中的上下文信息结构化、可裁剪、可追溯的方法论。"
        "它将散乱的 prompt 文本变成有序的证据和上下文块。",
        "SDK 的核心流程是：\n1. 证据入库（Ingest）\n2. 块派生（Derive）\n"
        "3. 策略裁剪（Prune）\n4. 渲染引用（Render）\n5. 装配输入（Assemble）",
        "多轮对话管理的要点：\n- 消息按时间序列存储在 session.messages 中\n"
        "- 旧消息会被滚动摘要压缩\n- 系统消息始终保留（priority=must）",
        "这是一段由 SDK 生成的摘要内容，覆盖了之前的对话主题。",
    ]

    def __init__(self):
        self._call_count = 0

    async def generate(self, request: LlmRequest) -> LlmResponse:
        idx = min(self._call_count, len(self.RESPONSES) - 1)
        content = self.RESPONSES[idx]
        self._call_count += 1
        return LlmResponse(
            content=content,
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------

async def main():
    # 初始化组件
    store = MemoryStore()
    token_estimator = CharBasedEstimator()
    llm_adapter = MockLlmAdapter()
    event_bus = InMemoryEventBus()

    engine = create_context_engine(
        store=store,
        token_estimator=token_estimator,
        llm_adapter=llm_adapter,
    )
    # 注入自定义 event_bus 以便观察事件
    engine._event_bus = event_bus

    session_id = "demo-multi-turn-001"

    # 运行时配置
    config = RuntimeConfig(
        model_hint="gpt-4o",
        budget=BudgetConfig(max_input_tokens=4096, reserved_reply_tokens=512),
        summary=SummaryConfig(
            enabled=True,
            trigger_message_count=6,   # 为演示设置较低阈值
            trigger_token_threshold=500,
            preserve_recent_messages=2,
        ),
    )

    # 模拟用户输入序列
    user_inputs = [
        "你好",
        "什么是 Context Engineering？",
        "SDK 的核心流程是怎样的？",
        "多轮对话管理有什么要点？",
    ]

    print("=" * 60)
    print("  Context Engineering SDK — 多轮对话示例")
    print("=" * 60)

    for i, user_text in enumerate(user_inputs, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"[用户]: {user_text}")

        # Phase A: 准备本轮输入
        user_msg = Message(
            role=Role.USER,
            content=user_text,
            author=Author(kind=AuthorKind.USER, id="user-001"),
        )
        result: PrepareResult = await engine.prepare_turn(
            session_id=session_id,
            user_message=user_msg,
            runtime_config=config,
        )

        # 输出报告
        report = result.report
        print(f"  [报告] Token 预算: {report.token_budget}, 已用: {report.token_used}")
        print(f"  [报告] 裁剪决策数: {len(report.prune_decisions)}")
        kept = sum(1 for d in report.prune_decisions if d.action == "kept")
        dropped = sum(1 for d in report.prune_decisions if d.action == "dropped")
        print(f"  [报告] 保留块: {kept}, 丢弃块: {dropped}")
        if report.degradations:
            print(f"  [报告] 降级: {report.degradations}")

        # 模拟模型生成回复
        llm_response = await llm_adapter.generate(
            LlmRequest(messages=result.assembled_input.parts)
        )
        assistant_text = llm_response.content
        print(f"[助手]: {assistant_text}")

        # Phase B: 提交 assistant 回复
        await engine.commit_assistant_message(
            session_id,
            Message(
                role=Role.ASSISTANT,
                content=assistant_text,
                author=Author(kind=AuthorKind.AGENT, id="agent-001"),
            ),
        )

    # 查看最终会话状态
    print("\n" + "=" * 60)
    print("  最终会话状态")
    print("=" * 60)
    doc = await store.get_session(session_id)
    print(f"  消息数: {len(doc.session.messages)}")
    print(f"  摘要: {'有' if doc.session.summary and doc.session.summary.content else '无'}")
    if doc.session.summary and doc.session.summary.content:
        print(f"  摘要内容: {doc.session.summary.content[:80]}...")

    # 输出事件历史
    print(f"\n  事件总数: {len(event_bus.history)}")
    event_types = {}
    for ev in event_bus.history:
        event_types[ev.event_type] = event_types.get(ev.event_type, 0) + 1
    for et, count in sorted(event_types.items()):
        print(f"    {et}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
