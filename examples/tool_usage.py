"""
工具调用与证据入库示例
======================

演示如何使用 Context Engineering SDK 记录工具调用，并将工具结果
作为证据（Evidence）入库，实现可追溯、可解释的上下文管理。

核心流程：
1. 用户提问 -> prepare_turn
2. 模拟工具调用（订单查询、天气查询等）
3. 将工具结果通过 EvidenceIngestor 落入证据库
4. 通过 record_tool_call 记录调用元数据
5. 下一轮 prepare_turn 时，证据自动参与上下文装配

运行方式:
    python examples/tool_usage.py
"""

import asyncio
import json
import time

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
    Ref,
    SourceKind,
    ToolCall,
    ToolCallStatus,
    ToolProvider,
    ToolType,
    ProviderKind,
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# Mock LLM Adapter
# ---------------------------------------------------------------------------

class MockLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="summary", model="mock-model")


# ---------------------------------------------------------------------------
# 模拟工具实现
# ---------------------------------------------------------------------------

class OrderService:
    """模拟订单查询服务"""

    @staticmethod
    def get_order(order_id: str) -> dict:
        orders = {
            "ORD-001": {
                "order_id": "ORD-001",
                "status": "shipped",
                "product": "机械键盘",
                "price": 599.00,
                "tracking": "SF1234567890",
                "estimated_delivery": "2026-03-01",
            },
            "ORD-002": {
                "order_id": "ORD-002",
                "status": "processing",
                "product": "无线鼠标",
                "price": 199.00,
                "tracking": None,
                "estimated_delivery": "2026-03-05",
            },
        }
        return orders.get(order_id, {"error": f"订单 {order_id} 不存在"})


class WeatherService:
    """模拟天气查询服务"""

    @staticmethod
    def get_weather(city: str) -> dict:
        return {
            "city": city,
            "temperature": 22,
            "condition": "晴",
            "humidity": 45,
            "wind": "东南风3级",
        }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main():
    store = MemoryStore()
    token_estimator = CharBasedEstimator()
    id_gen = UuidV4Generator()
    clock = SystemClock()

    engine = create_context_engine(
        store=store,
        token_estimator=token_estimator,
        llm_adapter=MockLlmAdapter(),
    )

    ingestor = DefaultEvidenceIngestor(store=store)
    session_id = "demo-tool-usage-001"

    config = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=4096, reserved_reply_tokens=512),
        summary=SummaryConfig(enabled=False),
    )

    print("=" * 60)
    print("  Context Engineering SDK — 工具调用与证据入库示例")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 第 1 轮：用户查询订单
    # ---------------------------------------------------------------
    print("\n--- 第 1 轮：订单查询 ---")
    user_msg = Message(
        role=Role.USER,
        content="帮我查一下订单 ORD-001 的物流状态",
        author=Author(kind=AuthorKind.USER, id="user-001"),
    )
    result = await engine.prepare_turn(session_id, user_msg, config)
    print(f"[用户]: {user_msg.content}")

    # 模拟调用订单工具
    tool_call_id = id_gen.generate()
    start_time = time.monotonic()
    order_result = OrderService.get_order("ORD-001")
    duration_ms = int((time.monotonic() - start_time) * 1000)

    print(f"[工具] 调用 getOrder(order_id='ORD-001')")
    print(f"[工具] 结果: {json.dumps(order_result, ensure_ascii=False, indent=2)}")

    # 将工具结果作为证据入库
    evidence, redaction = await ingestor.ingest(
        session_id=session_id,
        content=json.dumps(order_result, ensure_ascii=False),
        source=EvidenceSource(
            kind=SourceKind.TOOL,
            name="OrderService.getOrder",
            uri="https://api.example.com/orders",
        ),
        evidence_type=EvidenceType.TOOL_RESULT,
        links=EvidenceLinks(tool_call_id=tool_call_id),
        metadata={"order_id": "ORD-001"},
        options=IngestOptions(redact=False, dedup=True),
    )
    print(f"[证据] 入库成功, evidence_id={evidence.evidence_id}")

    # 记录工具调用元数据
    tc = ToolCall(
        tool_call_id=tool_call_id,
        tool="getOrder",
        provider=ToolProvider(
            kind=ProviderKind.BUILTIN,
            name="order-service",
            uri="https://api.example.com",
        ),
        type=ToolType.TOOL,
        called_at=clock.now_iso(),
        args_digest={"order_id": "ORD-001"},
        status=ToolCallStatus.SUCCESS,
        duration_ms=duration_ms,
        result_evidence_ids=[evidence.evidence_id],
    )
    await engine.record_tool_call(session_id, tc)
    print(f"[记录] 工具调用已记录, tool_call_id={tool_call_id}")

    # 提交助手回复（引用证据）
    assistant_text = (
        f"您的订单 ORD-001（{order_result['product']}）已发货，"
        f"快递单号为 {order_result['tracking']}，"
        f"预计 {order_result['estimated_delivery']} 送达。"
    )
    await engine.commit_assistant_message(
        session_id,
        Message(role=Role.ASSISTANT, content=assistant_text),
        refs=[Ref(evidence_id=evidence.evidence_id)],
    )
    print(f"[助手]: {assistant_text}")

    # ---------------------------------------------------------------
    # 第 2 轮：用户查询天气（使用 MCP 工具）
    # ---------------------------------------------------------------
    print("\n--- 第 2 轮：天气查询（MCP 工具） ---")
    user_msg2 = Message(
        role=Role.USER,
        content="北京今天天气怎么样？",
        author=Author(kind=AuthorKind.USER, id="user-001"),
    )
    result2 = await engine.prepare_turn(session_id, user_msg2, config)
    print(f"[用户]: {user_msg2.content}")

    # 模拟 MCP 工具调用
    tc2_id = id_gen.generate()
    weather_result = WeatherService.get_weather("北京")

    print(f"[工具] 调用 MCP:getWeather(city='北京')")
    print(f"[工具] 结果: {json.dumps(weather_result, ensure_ascii=False)}")

    # 入库天气证据
    ev2, _ = await ingestor.ingest(
        session_id=session_id,
        content=json.dumps(weather_result, ensure_ascii=False),
        source=EvidenceSource(
            kind=SourceKind.TOOL,
            name="weather-server",
            uri="mcp://weather-server/getWeather",
        ),
        evidence_type=EvidenceType.TOOL_RESULT,
        links=EvidenceLinks(tool_call_id=tc2_id),
        options=IngestOptions(redact=False),
    )

    # 记录 MCP 工具调用
    tc2 = ToolCall(
        tool_call_id=tc2_id,
        tool="getWeather",
        provider=ToolProvider(
            kind=ProviderKind.MCP,
            name="weather-server",
            uri="mcp://weather-server",
        ),
        type=ToolType.TOOL,
        called_at=clock.now_iso(),
        args_digest={"city": "北京"},
        status=ToolCallStatus.SUCCESS,
        duration_ms=15,
        result_evidence_ids=[ev2.evidence_id],
    )
    await engine.record_tool_call(session_id, tc2)

    assistant_text2 = (
        f"北京今天天气{weather_result['condition']}，"
        f"气温{weather_result['temperature']}°C，"
        f"湿度{weather_result['humidity']}%，{weather_result['wind']}。"
    )
    await engine.commit_assistant_message(
        session_id,
        Message(role=Role.ASSISTANT, content=assistant_text2),
        refs=[Ref(evidence_id=ev2.evidence_id)],
    )
    print(f"[助手]: {assistant_text2}")

    # ---------------------------------------------------------------
    # 第 3 轮：验证证据在上下文中的使用
    # ---------------------------------------------------------------
    print("\n--- 第 3 轮：综合查询（证据自动参与装配） ---")
    user_msg3 = Message(
        role=Role.USER,
        content="总结一下刚才查到的信息",
    )
    result3 = await engine.prepare_turn(session_id, user_msg3, config)
    print(f"[用户]: {user_msg3.content}")
    print(f"  [装配] 最终输入包含 {len(result3.assembled_input.parts)} 个部分")
    print(f"  [装配] 总 token 估算: {result3.assembled_input.total_tokens}")
    print(f"  [报告] 裁剪决策: {len(result3.report.prune_decisions)} 块")

    # 查看证据中包含了之前的工具结果
    assembled_text = result3.assembled_input.text or ""
    has_order = "ORD-001" in assembled_text or "shipped" in assembled_text
    has_weather = "北京" in assembled_text or "晴" in assembled_text
    print(f"  [验证] 上下文包含订单数据: {has_order}")
    print(f"  [验证] 上下文包含天气数据: {has_weather}")

    # ---------------------------------------------------------------
    # 查看最终状态
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  最终状态汇总")
    print("=" * 60)
    doc = await store.get_session(session_id)
    print(f"  消息数: {len(doc.session.messages)}")
    print(f"  证据数: {len(doc.evidences)}")
    print(f"  工具调用数: {len(doc.session.tool_state.tool_calls)}")

    print("\n  证据清单:")
    for eid, ev in doc.evidences.items():
        print(f"    - {eid[:8]}... type={ev.type.value}, source={ev.source.name}")
        if ev.links.tool_call_id:
            print(f"      链接到 tool_call_id={ev.links.tool_call_id[:8]}...")

    print("\n  工具调用清单:")
    for tc in doc.session.tool_state.tool_calls:
        provider_info = f"{tc.provider.kind.value}:{tc.provider.name}"
        print(f"    - {tc.tool} ({provider_info}) status={tc.status.value} "
              f"duration={tc.duration_ms}ms")


if __name__ == "__main__":
    asyncio.run(main())
