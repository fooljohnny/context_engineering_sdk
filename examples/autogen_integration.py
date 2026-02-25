"""
AutoGen 多 Agent 集成示例
=========================

演示如何将 Context Engineering SDK 与 AutoGen 框架集成：
- 多个 Agent 之间的消息传递通过 author 字段追踪
- 每个 Agent 的工具调用和模型用量分别记录
- 任务分解与结果归因（task_id 关联工具调用和证据）
- 多 Agent 场景下的完整可审计链路

注意：此示例使用模拟的 AutoGen 接口来演示集成模式。
实际使用时需安装 autogen/ag2 包并替换为真实实现。

运行方式:
    python examples/autogen_integration.py
"""

import asyncio
import json
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
    Ref,
    SourceKind,
    Task,
    TaskStatus,
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
# 模拟 AutoGen Agent 定义
# ---------------------------------------------------------------------------

@dataclass
class AutoGenAgent:
    """模拟 AutoGen 的 Agent"""
    name: str
    agent_id: str
    system_message: str
    tools: list[str] = field(default_factory=list)


@dataclass
class AutoGenMessage:
    """模拟 AutoGen 的消息"""
    sender: str
    receiver: str
    content: str
    tool_calls: list[dict] | None = None


# ---------------------------------------------------------------------------
# AutoGen <-> CE SDK 适配器
# ---------------------------------------------------------------------------

class AutoGenCEAdapter:
    """
    将 AutoGen 多 Agent 的生命周期事件映射到 CE SDK。

    核心映射：
    - AutoGen Agent Message -> CE Message (with author.kind=agent)
    - AutoGen Tool/Function Call -> CE ToolCall + Evidence
    - AutoGen 多 Agent 轮次 -> CE session.messages 按时间顺序记录
    """

    def __init__(self, engine, store, ingestor):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()

    def _agent_author(self, agent: AutoGenAgent) -> Author:
        return Author(kind=AuthorKind.AGENT, id=agent.agent_id)

    async def on_agent_message(
        self, session_id: str, agent: AutoGenAgent, content: str,
        is_user_proxy: bool = False,
    ):
        """记录 Agent 发送的消息。"""
        if is_user_proxy:
            role = Role.USER
            author = Author(kind=AuthorKind.USER, id=agent.agent_id)
        else:
            role = Role.ASSISTANT
            author = self._agent_author(agent)

        msg = Message(
            role=role,
            content=content,
            author=author,
            at=self._clock.now_iso(),
        )
        await self._store.append_messages(session_id, [msg])
        return msg

    async def on_tool_execution(
        self,
        session_id: str,
        agent: AutoGenAgent,
        tool_name: str,
        args: dict,
        result: str,
        task_id: str | None = None,
    ) -> tuple[ToolCall, Evidence]:
        """记录 Agent 的工具调用。"""
        tc_id = self._id_gen.generate()

        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=result,
            source=EvidenceSource(
                kind=SourceKind.TOOL, name=tool_name,
            ),
            evidence_type=EvidenceType.TOOL_RESULT,
            links=EvidenceLinks(tool_call_id=tc_id),
            metadata={"agent_id": agent.agent_id, "agent_name": agent.name},
            options=IngestOptions(redact=False),
        )

        tc = ToolCall(
            tool_call_id=tc_id,
            tool=tool_name,
            provider=ToolProvider(kind=ProviderKind.BUILTIN, name=f"autogen:{agent.name}"),
            called_at=self._clock.now_iso(),
            args_digest=args,
            status=ToolCallStatus.SUCCESS,
            result_evidence_ids=[ev.evidence_id],
            task_id=task_id,
        )
        await self._engine.record_tool_call(session_id, tc)
        return tc, ev

    async def register_tasks(
        self, session_id: str, tasks: list[Task]
    ):
        """注册任务计划到 CE SDK。"""
        await self._store.upsert_tasks(session_id, tasks)

    async def update_task_status(
        self, session_id: str, task_id: str, status: TaskStatus,
        result_evidence_ids: list[str] | None = None,
    ):
        """更新任务状态。"""
        doc = await self._store.get_session(session_id)
        if doc:
            for task in doc.session.task_state.todo_list.tasks:
                if task.task_id == task_id:
                    task.status = status
                    if result_evidence_ids:
                        task.result_evidence_ids.extend(result_evidence_ids)
                    break
            await self._store.upsert_tasks(
                session_id,
                doc.session.task_state.todo_list.tasks,
            )


# ---------------------------------------------------------------------------
# 模拟 AutoGen 多 Agent 协作
# ---------------------------------------------------------------------------

async def simulate_autogen_multi_agent(
    adapter: AutoGenCEAdapter, session_id: str
):
    """模拟一个多 Agent 协作场景：用户请求 -> Planner -> Researcher -> Coder"""

    # 定义 Agent
    user_proxy = AutoGenAgent(
        name="UserProxy",
        agent_id="user-proxy-001",
        system_message="",
    )

    planner = AutoGenAgent(
        name="Planner",
        agent_id="planner-001",
        system_message="你是一个任务规划 Agent，负责分解复杂任务。",
    )

    researcher = AutoGenAgent(
        name="Researcher",
        agent_id="researcher-001",
        system_message="你是一个信息检索 Agent，负责查找资料。",
        tools=["web_search", "doc_lookup"],
    )

    coder = AutoGenAgent(
        name="Coder",
        agent_id="coder-001",
        system_message="你是一个代码编写 Agent，负责实现代码。",
        tools=["code_executor"],
    )

    store = adapter._store

    # 初始化会话
    engine = adapter._engine
    config = RuntimeConfig(summary=SummaryConfig(enabled=False))
    await engine.prepare_turn(
        session_id,
        Message(role=Role.SYSTEM, content="AutoGen Multi-Agent System"),
        config,
    )

    # ---------------------------------------------------------------
    # Step 1: 用户输入
    # ---------------------------------------------------------------
    print("\n--- Step 1: 用户输入 ---")
    await adapter.on_agent_message(
        session_id, user_proxy,
        "帮我用 Python 写一个计算斐波那契数列的函数，需要支持缓存优化。",
        is_user_proxy=True,
    )
    print(f"[{user_proxy.name}]: 帮我用 Python 写一个计算斐波那契数列的函数，需要支持缓存优化。")

    # ---------------------------------------------------------------
    # Step 2: Planner 分解任务
    # ---------------------------------------------------------------
    print("\n--- Step 2: Planner 分解任务 ---")
    tasks = [
        Task(task_id="task-1", name="调研斐波那契优化方案", status=TaskStatus.PENDING),
        Task(task_id="task-2", name="编写带缓存的实现代码",
             status=TaskStatus.PENDING, depends_on=["task-1"]),
        Task(task_id="task-3", name="测试并验证正确性",
             status=TaskStatus.PENDING, depends_on=["task-2"]),
    ]
    await adapter.register_tasks(session_id, tasks)

    planner_msg = (
        "我已将任务分解为 3 个子任务：\n"
        "1. 调研斐波那契优化方案\n"
        "2. 编写带缓存的实现代码\n"
        "3. 测试并验证正确性"
    )
    await adapter.on_agent_message(session_id, planner, planner_msg)
    print(f"[{planner.name}]: {planner_msg}")

    # ---------------------------------------------------------------
    # Step 3: Researcher 执行调研
    # ---------------------------------------------------------------
    print("\n--- Step 3: Researcher 调研 ---")
    await adapter.update_task_status(session_id, "task-1", TaskStatus.IN_PROGRESS)

    research_result = json.dumps({
        "methods": [
            {"name": "递归 + functools.lru_cache", "complexity": "O(n)", "space": "O(n)"},
            {"name": "迭代法", "complexity": "O(n)", "space": "O(1)"},
            {"name": "矩阵快速幂", "complexity": "O(log n)", "space": "O(1)"},
        ],
        "recommendation": "对于通用场景，推荐 lru_cache 装饰器方案",
    }, ensure_ascii=False)

    tc1, ev1 = await adapter.on_tool_execution(
        session_id, researcher, "doc_lookup",
        args={"query": "fibonacci optimization python"},
        result=research_result,
        task_id="task-1",
    )
    print(f"[{researcher.name}] 工具调用: doc_lookup -> 找到 3 种优化方案")

    await adapter.on_agent_message(
        session_id, researcher,
        f"调研完成，推荐使用 functools.lru_cache 装饰器方案。\n详情: {research_result[:80]}...",
    )
    await adapter.update_task_status(
        session_id, "task-1", TaskStatus.COMPLETED,
        result_evidence_ids=[ev1.evidence_id],
    )
    print(f"[{researcher.name}]: 调研完成，推荐 lru_cache 方案")

    # ---------------------------------------------------------------
    # Step 4: Coder 编写代码
    # ---------------------------------------------------------------
    print("\n--- Step 4: Coder 编写代码 ---")
    await adapter.update_task_status(session_id, "task-2", TaskStatus.IN_PROGRESS)

    code = '''from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """计算第 n 个斐波那契数（带缓存优化）"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''

    tc2, ev2 = await adapter.on_tool_execution(
        session_id, coder, "code_executor",
        args={"action": "write", "filename": "fibonacci.py"},
        result=code,
        task_id="task-2",
    )
    print(f"[{coder.name}] 工具调用: code_executor -> 编写了 fibonacci.py")

    await adapter.on_agent_message(session_id, coder, f"代码已编写完成：\n```python\n{code}```")
    await adapter.update_task_status(
        session_id, "task-2", TaskStatus.COMPLETED,
        result_evidence_ids=[ev2.evidence_id],
    )
    print(f"[{coder.name}]: 代码编写完成")

    # ---------------------------------------------------------------
    # Step 5: Coder 测试代码
    # ---------------------------------------------------------------
    print("\n--- Step 5: Coder 测试 ---")
    await adapter.update_task_status(session_id, "task-3", TaskStatus.IN_PROGRESS)

    test_result = json.dumps({
        "tests_run": 5,
        "tests_passed": 5,
        "results": {
            "fibonacci(0)": 0, "fibonacci(1)": 1,
            "fibonacci(10)": 55, "fibonacci(20)": 6765,
            "fibonacci(30)": 832040,
        }
    }, ensure_ascii=False)

    tc3, ev3 = await adapter.on_tool_execution(
        session_id, coder, "code_executor",
        args={"action": "test", "filename": "fibonacci.py"},
        result=test_result,
        task_id="task-3",
    )
    print(f"[{coder.name}] 工具调用: code_executor(test) -> 5/5 测试通过")

    await adapter.on_agent_message(session_id, coder, "所有测试通过！")
    await adapter.update_task_status(
        session_id, "task-3", TaskStatus.COMPLETED,
        result_evidence_ids=[ev3.evidence_id],
    )
    print(f"[{coder.name}]: 所有测试通过")


async def main():
    store = MemoryStore()
    id_gen = UuidV4Generator()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )

    adapter = AutoGenCEAdapter(engine=engine, store=store, ingestor=ingestor)
    session_id = "autogen-demo-001"

    print("=" * 60)
    print("  Context Engineering SDK — AutoGen 多 Agent 集成示例")
    print("=" * 60)

    await simulate_autogen_multi_agent(adapter, session_id)

    # ---------------------------------------------------------------
    # 最终状态
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  最终状态汇总")
    print("=" * 60)

    doc = await store.get_session(session_id)
    evidences = await store.list_evidences(session_id)

    print(f"  消息数: {len(doc.session.messages)}")
    print(f"  证据数: {len(evidences)}")
    print(f"  工具调用数: {len(doc.session.tool_state.tool_calls)}")

    # 展示 Agent 消息归属
    print("\n  消息归属:")
    for msg in doc.session.messages:
        if msg.author:
            agent_name = msg.author.id
            print(f"    [{msg.role.value}] author={agent_name}: {msg.content[:50]}...")
        else:
            print(f"    [{msg.role.value}]: {msg.content[:50]}...")

    # 展示任务完成状态
    print("\n  任务状态:")
    for task in doc.session.task_state.todo_list.tasks:
        evidence_refs = ", ".join(eid[:8] + "..." for eid in task.result_evidence_ids)
        print(f"    [{task.status.value}] {task.name}")
        if evidence_refs:
            print(f"      -> 产出证据: {evidence_refs}")

    # 展示工具调用归因
    print("\n  工具调用归因:")
    for tc in doc.session.tool_state.tool_calls:
        task_ref = f"task={tc.task_id}" if tc.task_id else "no task"
        ev_refs = ", ".join(eid[:8] + "..." for eid in tc.result_evidence_ids)
        print(f"    {tc.tool} ({tc.provider.name}) [{task_ref}] -> {ev_refs}")

    print("\n  --- 以上展示了 AutoGen 多 Agent 协作的完整可审计链路 ---")
    print("  --- 每个 Agent 的消息、工具调用、产出证据均有归属追踪 ---")


if __name__ == "__main__":
    asyncio.run(main())
