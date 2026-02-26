"""
AutoGen + Context Engineering SDK 完整集成示例
===============================================

演示功能：
- AutoGen 多 Agent 协作与 CE SDK 的集成
- UserProxy → Planner → Researcher → Coder 多 Agent 工作流
- 跨会话用户画像与偏好持久化（MemoryManager）
- 多轮长对话摘要压缩
- 任务分解与结果归因（task_id 关联证据和工具调用）
- 每个 Agent 的消息归属追踪（author 字段）

场景说明：
  会话 1（多轮多 Agent 协作）：用户请求一个编程任务，多 Agent 分工协作，
  SDK 记录每个 Agent 的消息、工具调用和产出，同时提取用户偏好。
  会话 2：新会话加载用户偏好和历史会话摘要，验证跨会话记忆。

运行方式：
    python examples/autogen_integration.py
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
                    "多 Agent 协作完成了一个 Python Web API 开发任务。用户是一名偏好"
                    "使用 Python 和 FastAPI 的后端开发者。Planner 将任务分为 API 设计、"
                    "代码实现和测试验证三步。Researcher 调研了 FastAPI 最佳实践，"
                    "Coder 实现了 REST API 并通过了全部测试。"
                ),
                model="mock-gpt-4o",
                prompt_tokens=200, completion_tokens=100, total_tokens=300,
            )
        return LlmResponse(content="ok", model="mock-gpt-4o",
                           prompt_tokens=80, completion_tokens=30, total_tokens=110)


# ---------------------------------------------------------------------------
# Simulated AutoGen Agent definitions
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    name: str
    agent_id: str
    system_message: str
    tools: list[str] = field(default_factory=list)


USER_PROXY = Agent(name="UserProxy", agent_id="user-proxy", system_message="")
PLANNER = Agent(name="Planner", agent_id="planner-001",
                system_message="你是任务规划 Agent，负责分解复杂任务。")
RESEARCHER = Agent(name="Researcher", agent_id="researcher-001",
                   system_message="你是调研 Agent，负责查找技术资料。",
                   tools=["web_search", "doc_lookup"])
CODER = Agent(name="Coder", agent_id="coder-001",
              system_message="你是编码 Agent，负责实现代码。",
              tools=["code_executor", "test_runner"])


PREFERENCE_MAP = {
    "Python": ("tech_stack", "用户熟悉 Python"),
    "FastAPI": ("framework", "用户偏好使用 FastAPI 框架"),
    "Go": ("tech_stack", "用户熟悉 Go"),
    "简洁": ("style", "用户偏好简洁回答"),
    "中文": ("language", "用户偏好中文交流"),
    "后端": ("domain", "用户是后端开发者"),
    "REST": ("api_style", "用户偏好 REST API 风格"),
}


# ---------------------------------------------------------------------------
# AutoGen CE Adapter
# ---------------------------------------------------------------------------

class AutoGenCEAdapter:
    def __init__(self, engine, store, ingestor, memory_mgr, user_id):
        self._engine = engine
        self._store = store
        self._ingestor = ingestor
        self._memory_mgr = memory_mgr
        self._user_id = user_id
        self._id_gen = UuidV4Generator()
        self._clock = SystemClock()

    async def start_session(self, session_id, config):
        memory_blocks = await self._memory_mgr.build_memory_blocks(self._user_id)
        config.extra_context_blocks = memory_blocks
        sys_msg = Message(role=Role.SYSTEM, content="AutoGen Multi-Agent System")
        await self._engine.prepare_turn(session_id, sys_msg, config)
        return config

    async def user_message(self, session_id, content, config):
        memory_blocks = await self._memory_mgr.build_memory_blocks(self._user_id)
        config.extra_context_blocks = memory_blocks
        msg = Message(
            role=Role.USER, content=content,
            author=Author(kind=AuthorKind.USER, id=self._user_id),
        )
        result = await self._engine.prepare_turn(session_id, msg, config)

        for kw, (cat, desc) in PREFERENCE_MAP.items():
            if kw in content:
                existing = await self._memory_mgr.get_preferences(self._user_id)
                if not any(desc in p.content for p in existing):
                    await self._memory_mgr.add_preference(
                        self._user_id, cat, desc, session_id=session_id,
                    )
        return result

    async def agent_message(self, session_id, agent: Agent, content):
        msg = Message(
            role=Role.ASSISTANT, content=content,
            author=Author(kind=AuthorKind.AGENT, id=agent.agent_id),
            at=self._clock.now_iso(),
        )
        await self._store.append_messages(session_id, [msg])

    async def tool_execution(self, session_id, agent: Agent, tool_name, args, result, task_id=None):
        tc_id = self._id_gen.generate()
        ev, _ = await self._ingestor.ingest(
            session_id=session_id,
            content=result,
            source=EvidenceSource(kind=SourceKind.TOOL, name=tool_name),
            evidence_type=EvidenceType.TOOL_RESULT,
            links=EvidenceLinks(tool_call_id=tc_id),
            metadata={"agent_id": agent.agent_id},
            options=IngestOptions(redact=False),
        )
        tc = ToolCall(
            tool_call_id=tc_id, tool=tool_name,
            provider=ToolProvider(kind=ProviderKind.BUILTIN, name=f"autogen:{agent.name}"),
            called_at=self._clock.now_iso(),
            args_digest=args,
            status=ToolCallStatus.SUCCESS,
            result_evidence_ids=[ev.evidence_id],
            task_id=task_id,
        )
        await self._engine.record_tool_call(session_id, tc)
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
# Multi-Agent Collaboration Simulation
# ---------------------------------------------------------------------------

async def run_session_1(adapter, session_id, config):
    """Session 1: Multi-agent collaboration to build a REST API."""

    print("\n  [Step 1] User request")
    result = await adapter.user_message(
        session_id,
        "我是后端开发者，熟悉 Python 和 FastAPI，请帮我用 FastAPI 写一个用户管理的 REST API，"
        "包含 CRUD 操作，要简洁，用中文回复",
        config,
    )
    print(f"    Token used: {result.report.token_used}")

    # Planner decomposes task
    print("\n  [Step 2] Planner decomposes task")
    tasks = [
        Task(task_id="t1", name="调研 FastAPI CRUD 最佳实践", status=TaskStatus.PENDING),
        Task(task_id="t2", name="实现用户管理 REST API",
             status=TaskStatus.PENDING, depends_on=["t1"]),
        Task(task_id="t3", name="编写单元测试并验证",
             status=TaskStatus.PENDING, depends_on=["t2"]),
    ]
    await adapter.register_tasks(session_id, tasks)
    planner_msg = "任务已分解：1) 调研 FastAPI 最佳实践 2) 实现 CRUD API 3) 测试验证"
    await adapter.agent_message(session_id, PLANNER, planner_msg)
    print(f"    [{PLANNER.name}]: {planner_msg}")

    # Researcher does research
    print("\n  [Step 3] Researcher investigates")
    await adapter.update_task(session_id, "t1", TaskStatus.IN_PROGRESS)

    research_result = json.dumps({
        "best_practices": [
            "使用 Pydantic 模型做请求/响应验证",
            "使用依赖注入管理数据库连接",
            "使用 HTTPException 统一错误处理",
            "使用 APIRouter 组织路由",
        ],
        "reference": "https://fastapi.tiangolo.com/tutorial/",
    }, ensure_ascii=False)

    tc1, ev1 = await adapter.tool_execution(
        session_id, RESEARCHER, "doc_lookup",
        args={"query": "FastAPI CRUD best practices"},
        result=research_result, task_id="t1",
    )
    research_msg = "调研完成：推荐使用 Pydantic 模型 + APIRouter + 依赖注入模式"
    await adapter.agent_message(session_id, RESEARCHER, research_msg)
    await adapter.update_task(session_id, "t1", TaskStatus.COMPLETED, [ev1.evidence_id])
    print(f"    [{RESEARCHER.name}] Tool: doc_lookup → best practices found")
    print(f"    [{RESEARCHER.name}]: {research_msg}")

    # User asks a follow-up
    print("\n  [Step 4] User follow-up")
    result2 = await adapter.user_message(
        session_id,
        "数据库用 SQLite 就行，先不考虑鉴权",
        config,
    )
    await adapter.agent_message(session_id, PLANNER, "收到，更新方案：SQLite + 无鉴权简化版")
    print(f"    Token used: {result2.report.token_used}")

    # Coder implements
    print("\n  [Step 5] Coder implements")
    await adapter.update_task(session_id, "t2", TaskStatus.IN_PROGRESS)

    code = '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
users_db: dict[int, dict] = {}
next_id = 1

class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    global next_id
    user_data = {"id": next_id, **user.dict()}
    users_db[next_id] = user_data
    next_id += 1
    return user_data

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UserCreate):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    users_db[user_id] = {"id": user_id, **user.dict()}
    return users_db[user_id]

@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[user_id]
    return {"message": "User deleted"}
'''
    tc2, ev2 = await adapter.tool_execution(
        session_id, CODER, "code_executor",
        args={"action": "write", "filename": "main.py"},
        result=code, task_id="t2",
    )
    await adapter.agent_message(session_id, CODER, f"代码已完成，包含 4 个 CRUD 端点")
    await adapter.update_task(session_id, "t2", TaskStatus.COMPLETED, [ev2.evidence_id])
    print(f"    [{CODER.name}] Tool: code_executor → main.py written")

    # Coder runs tests
    print("\n  [Step 6] Coder tests")
    await adapter.update_task(session_id, "t3", TaskStatus.IN_PROGRESS)

    test_result = json.dumps({
        "tests_run": 6, "passed": 6, "failed": 0,
        "details": [
            "test_create_user: PASS", "test_get_user: PASS",
            "test_update_user: PASS", "test_delete_user: PASS",
            "test_get_nonexistent: PASS", "test_delete_nonexistent: PASS",
        ],
    }, ensure_ascii=False)

    tc3, ev3 = await adapter.tool_execution(
        session_id, CODER, "test_runner",
        args={"action": "test", "filename": "main.py"},
        result=test_result, task_id="t3",
    )
    await adapter.agent_message(session_id, CODER, "全部 6 个测试通过！")
    await adapter.update_task(session_id, "t3", TaskStatus.COMPLETED, [ev3.evidence_id])
    print(f"    [{CODER.name}] Tool: test_runner → 6/6 passed")

    # User asks another question (triggers more turns for compression)
    print("\n  [Step 7] User asks about deployment")
    result3 = await adapter.user_message(
        session_id,
        "部署到生产环境有什么建议？",
        config,
    )
    await adapter.agent_message(
        session_id, RESEARCHER,
        "建议：1) 使用 uvicorn + gunicorn 2) Docker 容器化 3) Nginx 反向代理 4) 配置 CORS",
    )
    print(f"    Token used: {result3.report.token_used}")

    print("\n  [Step 8] User wraps up")
    result4 = await adapter.user_message(session_id, "好的，非常感谢！任务完成了", config)
    await adapter.agent_message(session_id, PLANNER, "所有任务已完成，感谢使用多 Agent 协作系统！")


async def run_session_2(adapter, session_id, config):
    """Session 2: cross-session memory verification."""

    print("\n  [Turn 1]")
    result = await adapter.user_message(session_id, "你好，上次帮我做了什么？", config)
    assembled = result.assembled_input.text or ""
    has_pref = "Preferences" in assembled or "FastAPI" in assembled
    has_hist = "Session Summaries" in assembled or "REST" in assembled
    print(f"    Memory: preferences={has_pref}, history={has_hist}")
    print(f"    Memory blocks injected: {len(config.extra_context_blocks)}")
    for blk in config.extra_context_blocks:
        print(f"      - [{blk.priority.value}] {blk.content[:60]}...")
    await adapter.agent_message(
        session_id, PLANNER,
        "根据记忆，上次我们用 FastAPI 完成了用户管理 REST API，全部测试通过。",
    )

    print("\n  [Turn 2]")
    await adapter.user_message(session_id, "这次帮我加上 JWT 鉴权功能", config)
    await adapter.agent_message(session_id, PLANNER, "好的，我来规划 JWT 鉴权的实现步骤...")

    print("\n  [Turn 3]")
    await adapter.user_message(session_id, "先到这里，下次继续", config)
    await adapter.agent_message(session_id, PLANNER, "好的，已记录进度，下次继续！")


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

    user_id = "user-autogen-demo"
    adapter = AutoGenCEAdapter(
        engine=engine, store=store, ingestor=ingestor,
        memory_mgr=memory_mgr, user_id=user_id,
    )

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
    print("  AutoGen + CE SDK: Multi-Agent with Cross-Session Memory")
    print("=" * 70)

    # --- Session 1 ---
    session1 = "ag-session-001"
    print(f"\n{'─' * 70}")
    print(f"  SESSION 1: {session1} (multi-agent collaboration)")
    print(f"{'─' * 70}")

    config1 = RuntimeConfig(
        budget=BudgetConfig(max_input_tokens=8192, reserved_reply_tokens=1024),
        summary=summary_config,
    )
    await adapter.start_session(session1, config1)
    await run_session_1(adapter, session1, config1)

    doc1 = await store.get_session(session1)
    evidences1 = await store.list_evidences(session1)
    prefs = await memory_mgr.get_preferences(user_id)
    has_summary = doc1.session.summary and doc1.session.summary.content

    print(f"\n  Session 1 Final State:")
    print(f"    Messages: {len(doc1.session.messages)}")
    print(f"    Evidences: {len(evidences1)}")
    print(f"    Tool calls: {len(doc1.session.tool_state.tool_calls)}")
    print(f"    Tasks: {len(doc1.session.task_state.todo_list.tasks)}")
    print(f"    Compressed: {bool(has_summary)}")
    if has_summary:
        print(f"    Summary: {doc1.session.summary.content[:100]}...")

    print(f"\n    Task status:")
    for t in doc1.session.task_state.todo_list.tasks:
        ev_refs = ", ".join(eid[:8] + "..." for eid in t.result_evidence_ids)
        print(f"      [{t.status.value}] {t.name} → {ev_refs or 'no evidence'}")

    print(f"\n    Preferences ({len(prefs)}):")
    for p in prefs:
        print(f"      - [{p.category}] {p.content}")

    summary_text = doc1.session.summary.content if has_summary else (
        "多 Agent 协作完成 FastAPI 用户管理 REST API，包含 CRUD + 测试。"
    )
    await memory_mgr.add_session_summary(user_id, session1, summary_text)

    # --- Session 2 ---
    session2 = "ag-session-002"
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

    await adapter.start_session(session2, config2)
    await run_session_2(adapter, session2, config2)

    doc2 = await store.get_session(session2)
    print(f"\n  Session 2 Final: {len(doc2.session.messages)} messages")

    print(f"\n{'=' * 70}")
    print("  DEMO COMPLETE")
    print(f"  - Multi-agent: {len(doc1.session.tool_state.tool_calls)} tool calls across agents")
    print(f"  - Preferences persisted: {len(prefs)}")
    print(f"  - Cross-session summaries: {len(await memory_mgr.get_session_summaries(user_id))}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
