"""
AgentScope + Context Engineering SDK 完整集成示例
==================================================

演示功能：
- AgentScope 多 Agent 消息流与 CE SDK 的集成
- 基于消息广播/订阅模式的 Agent 协作（模拟 AgentScope 的 MsgHub）
- 跨会话用户画像识别与长期记忆（MemoryManager）
- 多轮长对话摘要压缩
- 每个 Agent 的消息可追溯（author 字段）
- 工具调用证据化与归因

场景说明：
  会话 1（多轮）：用户与 AgentScope 多 Agent 系统交互（Coordinator + DataAnalyst
  + ReportWriter），完成一个数据分析任务。SDK 记录所有 Agent 消息和工具调用。
  会话 2：新会话加载跨会话记忆，验证偏好和摘要的持久性。

运行方式：
    python examples/agentscope_integration.py
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
    Ref, SourceKind,
    Task, TaskStatus,
    ToolCall, ToolCallStatus, ToolProvider, ProviderKind,
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class SmartMockLlm:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        prompt_text = " ".join(p.content for p in request.messages)
        if "summarize" in prompt_text.lower() or "摘要" in prompt_text:
            return LlmResponse(
                content=(
                    "用户是一名数据分析师，偏好用 Python 和中文交流。"
                    "在会话中，多 Agent 协作完成了电商销售数据分析：DataAnalyst 执行了"
                    "数据查询和统计分析，ReportWriter 生成了中文分析报告。"
                    "关键发现：Q1 销售额同比增长 23%，电子品类表现最佳。"
                ),
                model="mock-gpt-4o",
                prompt_tokens=200, completion_tokens=100, total_tokens=300,
            )
        return LlmResponse(content="ok", model="mock-gpt-4o",
                           prompt_tokens=80, completion_tokens=30, total_tokens=110)


# ---------------------------------------------------------------------------
# Simulated AgentScope Agent & MsgHub
# ---------------------------------------------------------------------------

@dataclass
class ASAgent:
    """Simulates an AgentScope Agent."""
    name: str
    agent_id: str
    sys_prompt: str
    tools: list[str] = field(default_factory=list)


@dataclass
class ASMessage:
    """Simulates an AgentScope Msg."""
    name: str
    role: str
    content: str
    url: str | None = None
    metadata: dict = field(default_factory=dict)


class MsgHub:
    """Simulates AgentScope's msghub for multi-agent communication."""

    def __init__(self):
        self._participants: list[ASAgent] = []
        self._history: list[ASMessage] = []

    def add(self, agent: ASAgent):
        self._participants.append(agent)

    def broadcast(self, msg: ASMessage):
        self._history.append(msg)

    @property
    def history(self):
        return list(self._history)


COORDINATOR = ASAgent(
    name="Coordinator", agent_id="coordinator-001",
    sys_prompt="你是协调 Agent，负责分配任务和汇总结果。",
)
DATA_ANALYST = ASAgent(
    name="DataAnalyst", agent_id="analyst-001",
    sys_prompt="你是数据分析 Agent，负责执行数据查询和统计分析。",
    tools=["sql_query", "data_stats"],
)
REPORT_WRITER = ASAgent(
    name="ReportWriter", agent_id="writer-001",
    sys_prompt="你是报告撰写 Agent，负责生成分析报告。",
    tools=["format_report"],
)

PREFERENCE_MAP = {
    "数据": ("domain", "用户从事数据分析工作"),
    "Python": ("tech_stack", "用户熟悉 Python"),
    "pandas": ("tech_stack", "用户熟悉 pandas 数据分析库"),
    "中文": ("language", "用户偏好中文交流"),
    "简洁": ("style", "用户偏好简洁风格"),
    "图表": ("output_pref", "用户偏好数据可视化和图表展示"),
}


# ---------------------------------------------------------------------------
# AgentScope CE Adapter
# ---------------------------------------------------------------------------

class AgentScopeCEAdapter:
    def __init__(self, engine, store, ingestor, memory_mgr, user_id):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._memory_mgr = memory_mgr
        self._user_id = user_id
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()
        self._hub = MsgHub()

    @property
    def hub(self):
        return self._hub

    def setup_agents(self, agents: list[ASAgent]):
        for agent in agents:
            self._hub.add(agent)

    async def user_input(self, session_id, content, config):
        memory_blocks = await self._memory_mgr.build_memory_blocks(self._user_id)
        config.extra_context_blocks = memory_blocks

        msg = Message(
            role=Role.USER, content=content,
            author=Author(kind=AuthorKind.USER, id=self._user_id),
        )
        result = await self._engine.prepare_turn(session_id, msg, config)

        self._hub.broadcast(ASMessage(
            name="User", role="user", content=content,
        ))

        for kw, (cat, desc) in PREFERENCE_MAP.items():
            if kw in content:
                existing = await self._memory_mgr.get_preferences(self._user_id)
                if not any(desc in p.content for p in existing):
                    await self._memory_mgr.add_preference(
                        self._user_id, cat, desc, session_id=session_id,
                    )
        return result

    async def agent_speak(self, session_id, agent: ASAgent, content):
        msg = Message(
            role=Role.ASSISTANT, content=content,
            author=Author(kind=AuthorKind.AGENT, id=agent.agent_id),
            at=self._clock.now_iso(),
        )
        await self._store.append_messages(session_id, [msg])

        self._hub.broadcast(ASMessage(
            name=agent.name, role="assistant", content=content,
            metadata={"agent_id": agent.agent_id},
        ))

    async def agent_tool_call(self, session_id, agent: ASAgent,
                               tool_name, args, result, task_id=None):
        tc_id = self._id_gen.generate()
        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=result,
            source=EvidenceSource(kind=SourceKind.TOOL, name=tool_name),
            evidence_type=EvidenceType.TOOL_RESULT,
            links=EvidenceLinks(tool_call_id=tc_id),
            metadata={"agent_id": agent.agent_id, "agent_name": agent.name},
            options=IngestOptions(redact=False),
        )
        tc = ToolCall(
            tool_call_id=tc_id, tool=tool_name,
            provider=ToolProvider(kind=ProviderKind.BUILTIN,
                                  name=f"agentscope:{agent.name}"),
            called_at=self._clock.now_iso(),
            args_digest=args,
            status=ToolCallStatus.SUCCESS,
            result_evidence_ids=[ev.evidence_id],
            task_id=task_id,
        )
        await self._engine.record_tool_call(session_id, tc)

        self._hub.broadcast(ASMessage(
            name=agent.name, role="tool",
            content=f"[{tool_name}] {result[:80]}...",
            metadata={"tool_call_id": tc_id},
        ))
        return tc, ev

    async def register_tasks(self, session_id, tasks):
        await self._store.upsert_tasks(session_id, tasks)

    async def update_task(self, session_id, task_id, status, evidence_ids=None):
        doc = await self._store.get_session(session_id)
        if doc:
            for t in doc.session.task_state.todo_list.tasks:
                if t.task_id == task_id:
                    t.status = status
                    if evidence_ids:
                        t.result_evidence_ids.extend(evidence_ids)
                    break
            await self._store.upsert_tasks(
                session_id, doc.session.task_state.todo_list.tasks,
            )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

async def run_session_1(adapter, session_id, config):
    """Session 1: Multi-agent data analysis workflow."""

    print("\n  [Step 1] User request")
    result = await adapter.user_input(
        session_id,
        "我是数据分析师，熟悉 Python 和 pandas，请用中文帮我分析今年 Q1 的电商销售数据，"
        "生成一份简洁的分析报告，最好带图表说明",
        config,
    )
    print(f"    Token used: {result.report.token_used}")

    # Coordinator assigns tasks
    print("\n  [Step 2] Coordinator plans")
    tasks = [
        Task(task_id="t1", name="查询 Q1 销售数据", status=TaskStatus.PENDING),
        Task(task_id="t2", name="统计分析与趋势计算",
             status=TaskStatus.PENDING, depends_on=["t1"]),
        Task(task_id="t3", name="生成分析报告",
             status=TaskStatus.PENDING, depends_on=["t2"]),
    ]
    await adapter.register_tasks(session_id, tasks)
    await adapter.agent_speak(
        session_id, COORDINATOR,
        "收到任务，已分配：DataAnalyst 负责数据查询和分析，ReportWriter 负责报告生成。",
    )
    print(f"    [{COORDINATOR.name}]: Task planned, 3 subtasks created")

    # DataAnalyst queries data
    print("\n  [Step 3] DataAnalyst queries data")
    await adapter.update_task(session_id, "t1", TaskStatus.IN_PROGRESS)

    sql_result = json.dumps({
        "total_sales": 1250000,
        "categories": [
            {"name": "电子产品", "sales": 580000, "growth": "28%"},
            {"name": "服装", "sales": 320000, "growth": "15%"},
            {"name": "食品", "sales": 200000, "growth": "22%"},
            {"name": "家居", "sales": 150000, "growth": "10%"},
        ],
        "monthly": [
            {"month": "1月", "sales": 380000},
            {"month": "2月", "sales": 420000},
            {"month": "3月", "sales": 450000},
        ],
    }, ensure_ascii=False)

    tc1, ev1 = await adapter.agent_tool_call(
        session_id, DATA_ANALYST, "sql_query",
        args={"query": "SELECT * FROM sales WHERE quarter='Q1' GROUP BY category"},
        result=sql_result, task_id="t1",
    )
    await adapter.agent_speak(session_id, DATA_ANALYST, "Q1 数据查询完成，共 4 个品类")
    await adapter.update_task(session_id, "t1", TaskStatus.COMPLETED, [ev1.evidence_id])
    print(f"    [{DATA_ANALYST.name}] sql_query → 4 categories, total=1,250,000")

    # DataAnalyst performs statistics
    print("\n  [Step 4] DataAnalyst performs analysis")
    await adapter.update_task(session_id, "t2", TaskStatus.IN_PROGRESS)

    stats_result = json.dumps({
        "yoy_growth": "23%",
        "top_category": "电子产品 (28% growth)",
        "trend": "逐月递增，3月环比增长7.1%",
        "avg_monthly": 416667,
        "insights": [
            "电子产品占总销售额 46.4%，增速最快",
            "食品品类增长 22%，潜力较大",
            "家居品类增速仅 10%，需关注",
        ],
    }, ensure_ascii=False)

    tc2, ev2 = await adapter.agent_tool_call(
        session_id, DATA_ANALYST, "data_stats",
        args={"metrics": ["yoy_growth", "category_share", "monthly_trend"]},
        result=stats_result, task_id="t2",
    )
    await adapter.agent_speak(session_id, DATA_ANALYST,
                               "分析完成：Q1 同比增长 23%，电子产品表现最佳")
    await adapter.update_task(session_id, "t2", TaskStatus.COMPLETED, [ev2.evidence_id])
    print(f"    [{DATA_ANALYST.name}] data_stats → YoY +23%, top: electronics")

    # User asks follow-up
    print("\n  [Step 5] User follow-up")
    result2 = await adapter.user_input(
        session_id,
        "电子产品具体卖了哪些品类？能细分一下吗",
        config,
    )

    sub_category_result = json.dumps({
        "sub_categories": [
            {"name": "智能手机", "sales": 250000, "share": "43.1%"},
            {"name": "笔记本电脑", "sales": 180000, "share": "31.0%"},
            {"name": "智能穿戴", "sales": 100000, "share": "17.2%"},
            {"name": "配件", "sales": 50000, "share": "8.6%"},
        ]
    }, ensure_ascii=False)

    tc_sub, ev_sub = await adapter.agent_tool_call(
        session_id, DATA_ANALYST, "sql_query",
        args={"query": "SELECT * FROM sales WHERE category='电子产品' GROUP BY sub_category"},
        result=sub_category_result,
    )
    await adapter.agent_speak(session_id, DATA_ANALYST,
                               "细分数据：智能手机占 43.1%，笔记本 31.0%，穿戴 17.2%")
    print(f"    [{DATA_ANALYST.name}] sql_query → 4 sub-categories")

    # ReportWriter generates report
    print("\n  [Step 6] ReportWriter generates report")
    await adapter.update_task(session_id, "t3", TaskStatus.IN_PROGRESS)

    report_content = """# Q1 电商销售分析报告

## 概览
- 总销售额：125 万元，同比增长 23%
- 月度趋势：逐月递增（38万 → 42万 → 45万）

## 品类表现
| 品类 | 销售额 | 增长率 | 占比 |
|------|--------|--------|------|
| 电子产品 | 58万 | 28% | 46.4% |
| 服装 | 32万 | 15% | 25.6% |
| 食品 | 20万 | 22% | 16.0% |
| 家居 | 15万 | 10% | 12.0% |

## 关键洞察
1. 电子产品是增长引擎，智能手机贡献最大（43.1%）
2. 食品品类增长潜力大（22%），建议加大投入
3. 家居品类增速最慢（10%），需分析原因

## 建议
- 扩展智能穿戴品类，增速有空间
- 食品领域尝试直播带货模式
"""

    tc3, ev3 = await adapter.agent_tool_call(
        session_id, REPORT_WRITER, "format_report",
        args={"format": "markdown", "language": "zh-CN"},
        result=report_content, task_id="t3",
    )
    await adapter.agent_speak(session_id, REPORT_WRITER, "分析报告已生成，包含概览、品类分析和建议")
    await adapter.update_task(session_id, "t3", TaskStatus.COMPLETED, [ev3.evidence_id])
    print(f"    [{REPORT_WRITER.name}] format_report → markdown report generated")

    # Coordinator wraps up
    await adapter.agent_speak(
        session_id, COORDINATOR,
        "所有任务已完成：数据查询 ✓ 统计分析 ✓ 报告生成 ✓",
    )

    # User final turn
    print("\n  [Step 7] User final")
    await adapter.user_input(session_id, "非常好，报告很清晰！谢谢", config)
    await adapter.agent_speak(session_id, COORDINATOR, "感谢使用，有需要随时联系！")


async def run_session_2(adapter, session_id, config):
    """Session 2: cross-session memory verification."""

    print("\n  [Turn 1] Verify memory")
    result = await adapter.user_input(session_id, "你好，上次帮我做了什么分析？", config)
    assembled = result.assembled_input.text or ""
    has_pref = "Preferences" in assembled or "数据" in assembled
    has_hist = "Session Summaries" in assembled or "Q1" in assembled
    print(f"    Memory: preferences={has_pref}, history={has_hist}")
    print(f"    Memory blocks: {len(config.extra_context_blocks)}")
    for blk in config.extra_context_blocks:
        print(f"      - [{blk.priority.value}] {blk.content[:60]}...")
    await adapter.agent_speak(
        session_id, COORDINATOR,
        "根据记忆，上次我们分析了 Q1 电商销售数据，总销售额 125 万，同比增长 23%。",
    )

    print("\n  [Turn 2] Follow-up request")
    await adapter.user_input(session_id, "这次帮我分析 Q2 的数据，对比 Q1", config)
    await adapter.agent_speak(session_id, COORDINATOR,
                               "好的，我来安排 DataAnalyst 查询 Q2 数据并与 Q1 对比分析。")

    print("\n  [Turn 3] Wrap up")
    await adapter.user_input(session_id, "好的，下次继续", config)
    await adapter.agent_speak(session_id, COORDINATOR, "已记录，下次继续 Q2 分析！")


async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    llm_adapter = SmartMockLlm()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)
    user_memory_store = InMemoryUserMemoryStore()
    memory_mgr = MemoryManager(user_memory_store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=llm_adapter,
    )

    user_id = "user-agentscope-demo"
    adapter = AgentScopeCEAdapter(
        engine=engine, store=store, ingestor=ingestor,
        memory_mgr=memory_mgr, user_id=user_id,
    )
    adapter.setup_agents([COORDINATOR, DATA_ANALYST, REPORT_WRITER])

    summary_config = SummaryConfig(
        enabled=True,
        trigger_message_count=8,
        trigger_token_threshold=300,
        preserve_recent_messages=3,
    )
    config = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=8192, reserved_reply_tokens=1024),
        summary=summary_config,
    )

    print("=" * 70)
    print("  AgentScope + CE SDK: Multi-Agent Data Analysis with Memory")
    print("=" * 70)

    # --- Session 1 ---
    session1 = "as-session-001"
    print(f"\n{'─' * 70}")
    print(f"  SESSION 1: {session1} (multi-agent data analysis)")
    print(f"{'─' * 70}")

    await engine.prepare_turn(
        session1,
        Message(role=Role.SYSTEM, content="AgentScope Multi-Agent System"),
        RuntimeConfig(summary=SummaryConfig(enabled=False)),
    )
    await run_session_1(adapter, session1, config)

    doc1 = await store.get_session(session1)
    evidences1 = await store.list_evidences(session1)
    prefs = await memory_mgr.get_preferences(user_id)
    has_summary = doc1.session.summary and doc1.session.summary.content

    print(f"\n  Session 1 Final State:")
    print(f"    Messages: {len(doc1.session.messages)}")
    print(f"    Evidences: {len(evidences1)}")
    print(f"    Tool calls: {len(doc1.session.tool_state.tool_calls)}")
    print(f"    Compressed: {bool(has_summary)}")
    if has_summary:
        print(f"    Summary: {doc1.session.summary.content[:100]}...")

    print(f"\n    Tasks:")
    for t in doc1.session.task_state.todo_list.tasks:
        ev_count = len(t.result_evidence_ids)
        print(f"      [{t.status.value}] {t.name} → {ev_count} evidence(s)")

    print(f"\n    Preferences ({len(prefs)}):")
    for p in prefs:
        print(f"      - [{p.category}] {p.content}")

    print(f"\n    MsgHub history: {len(adapter.hub.history)} messages")
    for msg in adapter.hub.history[:5]:
        print(f"      [{msg.name}] {msg.content[:50]}...")
    if len(adapter.hub.history) > 5:
        print(f"      ... and {len(adapter.hub.history) - 5} more")

    summary_text = doc1.session.summary.content if has_summary else (
        "多 Agent 分析 Q1 电商销售数据，总销售额 125 万，同比增长 23%。"
    )
    await memory_mgr.add_session_summary(user_id, session1, summary_text)

    # --- Session 2 ---
    session2 = "as-session-002"
    config2 = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=8192, reserved_reply_tokens=1024),
        summary=SummaryConfig(enabled=False),
    )

    print(f"\n{'─' * 70}")
    print(f"  SESSION 2: {session2} (cross-session memory)")
    print(f"{'─' * 70}")

    all_memories = await memory_mgr.get_all_memories(user_id)
    print(f"\n  Cross-session memories: {len(all_memories)}")
    for m in all_memories:
        print(f"    - [{m.category}] {m.content[:60]}...")

    await engine.prepare_turn(
        session2,
        Message(role=Role.SYSTEM, content="AgentScope Multi-Agent System"),
        RuntimeConfig(summary=SummaryConfig(enabled=False)),
    )
    await run_session_2(adapter, session2, config2)

    doc2 = await store.get_session(session2)
    print(f"\n  Session 2 Final: {len(doc2.session.messages)} messages")

    print(f"\n{'=' * 70}")
    print("  DEMO COMPLETE")
    print(f"  - Agents: Coordinator, DataAnalyst, ReportWriter")
    print(f"  - Tool calls: {len(doc1.session.tool_state.tool_calls)}")
    print(f"  - Preferences: {len(prefs)}")
    print(f"  - Cross-session summaries: {len(await memory_mgr.get_session_summaries(user_id))}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
