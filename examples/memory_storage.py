"""
持久化记忆存储示例
==================

演示如何使用 FileStore 实现持久化的会话记忆存储：
- 使用 FileStore 将会话数据以 JSON 文件持久化到磁盘
- 跨进程恢复会话状态
- 证据与上下文块的持久化存储与检索
- 对比 MemoryStore（内存）和 FileStore（文件）两种存储方式

运行方式:
    python examples/memory_storage.py
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

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
    EvidenceSource,
    EvidenceType,
    SourceKind,
    Task,
    TaskStatus,
)
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor, IngestOptions
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse
from context_engineering_sdk.store.memory import MemoryStore
from context_engineering_sdk.store.file import FileStore


class MockLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(content="summary", model="mock")


async def demo_memory_store():
    """演示 MemoryStore：纯内存，进程结束后数据丢失。"""
    print("\n--- MemoryStore 演示 ---")
    store = MemoryStore()

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    config = RuntimeConfig(summary=SummaryConfig(enabled=False))

    await engine.prepare_turn(
        "mem-session",
        Message(role=Role.USER, content="这条消息存在内存中"),
        config,
    )
    await engine.commit_assistant_message(
        "mem-session",
        Message(role=Role.ASSISTANT, content="收到，但我在进程结束后会消失。"),
    )

    doc = await store.get_session("mem-session")
    print(f"  消息数: {len(doc.session.messages)}")
    print(f"  存储位置: 内存（进程退出后丢失）")
    return store


async def demo_file_store(base_dir: str):
    """演示 FileStore：持久化到磁盘 JSON 文件。"""
    print("\n--- FileStore 演示 ---")
    store = FileStore(base_dir)
    id_gen = UuidV4Generator()
    ingestor = DefaultEvidenceIngestor(store=store, id_generator=id_gen)

    engine = create_context_engine(
        store=store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    config = RuntimeConfig(summary=SummaryConfig(enabled=False))
    session_id = "file-session-001"

    # 第 1 轮：写入基本对话
    print("\n  [写入] 第 1 轮对话...")
    await engine.prepare_turn(
        session_id,
        Message(
            role=Role.USER,
            content="你好，请记住我的名字是小明",
            author=Author(kind=AuthorKind.USER, id="user-xiaoming"),
        ),
        config,
    )
    await engine.commit_assistant_message(
        session_id,
        Message(role=Role.ASSISTANT, content="好的小明，我已经记住了！"),
    )

    # 写入一些证据
    print("  [写入] 存储用户偏好证据...")
    ev1, _ = await ingestor.ingest(
        session_id=session_id,
        content="用户名字是小明，偏好中文交流",
        source=EvidenceSource(kind=SourceKind.USER, name="user_profile"),
        evidence_type=EvidenceType.USER_INPUT,
        metadata={"category": "profile"},
        options=IngestOptions(redact=False),
    )

    ev2, _ = await ingestor.ingest(
        session_id=session_id,
        content="用户的技术栈: Python, Go, Kubernetes",
        source=EvidenceSource(kind=SourceKind.USER, name="user_skills"),
        evidence_type=EvidenceType.USER_INPUT,
        metadata={"category": "skills"},
        options=IngestOptions(redact=False),
    )

    # 写入任务
    print("  [写入] 存储任务状态...")
    await store.upsert_tasks(session_id, [
        Task(task_id="t1", name="学习 Context Engineering", status=TaskStatus.IN_PROGRESS),
        Task(task_id="t2", name="集成到项目中", status=TaskStatus.PENDING),
    ])

    # 第 2 轮
    print("  [写入] 第 2 轮对话...")
    await engine.prepare_turn(
        session_id,
        Message(role=Role.USER, content="我正在学习 Context Engineering"),
        config,
    )
    await engine.commit_assistant_message(
        session_id,
        Message(role=Role.ASSISTANT, content="很好！Context Engineering 的核心是将上下文结构化..."),
    )

    # 查看持久化文件
    file_path = Path(base_dir) / f"{session_id}.json"
    print(f"\n  [存储] 文件路径: {file_path}")
    print(f"  [存储] 文件大小: {file_path.stat().st_size} bytes")

    return store, session_id, base_dir


async def demo_restore_from_file(base_dir: str, session_id: str):
    """演示从磁盘恢复会话状态（模拟新进程启动）。"""
    print("\n--- 从磁盘恢复会话 ---")

    # 创建一个全新的 FileStore 实例（模拟新进程）
    restored_store = FileStore(base_dir)

    # 从文件恢复
    doc = await restored_store.get_session(session_id)
    if doc is None:
        print("  [错误] 会话未找到！")
        return

    print(f"  [恢复] 会话 ID: {doc.session.session_id}")
    print(f"  [恢复] Schema 版本: {doc.schema_version}")
    print(f"  [恢复] 消息数: {len(doc.session.messages)}")
    print(f"  [恢复] 证据数: {len(doc.evidences)}")
    print(f"  [恢复] 任务数: {len(doc.session.task_state.todo_list.tasks)}")

    print("\n  消息列表:")
    for msg in doc.session.messages:
        role_str = msg.role.value
        content_preview = msg.content[:50] + ("..." if len(msg.content) > 50 else "")
        print(f"    [{role_str}]: {content_preview}")

    print("\n  证据列表:")
    for eid, ev in doc.evidences.items():
        print(f"    - [{ev.type.value}] {ev.content[:50]}...")

    print("\n  任务列表:")
    for task in doc.session.task_state.todo_list.tasks:
        print(f"    - [{task.status.value}] {task.name}")

    # 在恢复的会话上继续对话
    engine = create_context_engine(
        store=restored_store,
        token_estimator=CharBasedEstimator(),
        llm_adapter=MockLlmAdapter(),
    )
    config = RuntimeConfig(summary=SummaryConfig(enabled=False))

    print("\n  [续写] 在恢复的会话上继续第 3 轮对话...")
    result = await engine.prepare_turn(
        session_id,
        Message(role=Role.USER, content="上次我们聊到哪里了？"),
        config,
    )
    print(f"  [续写] 装配的输入包含 {len(result.assembled_input.parts)} 个部分")
    print(f"  [续写] 总 token 估算: {result.assembled_input.total_tokens}")

    await engine.commit_assistant_message(
        session_id,
        Message(role=Role.ASSISTANT, content="小明你好！上次我们聊到了 Context Engineering..."),
    )

    doc_after = await restored_store.get_session(session_id)
    print(f"  [续写] 续写后消息数: {len(doc_after.session.messages)}")


async def demo_compare_stores():
    """对比两种存储方式的特性。"""
    print("\n" + "=" * 60)
    print("  MemoryStore vs FileStore 特性对比")
    print("=" * 60)

    comparison = [
        ("持久化", "否（进程退出后丢失）", "是（JSON 文件）"),
        ("适用场景", "测试、原型、临时会话", "单机持久化、开发环境"),
        ("并发安全", "单线程安全", "单进程安全"),
        ("性能", "最快（纯内存）", "较快（本地文件 IO）"),
        ("数据可读性", "不可直接查看", "JSON 文件，可直接查看/编辑"),
        ("扩展方向", "—", "可进一步扩展为 SqlStore / RedisStore"),
    ]

    print(f"\n  {'特性':<12} {'MemoryStore':<24} {'FileStore'}")
    print(f"  {'-' * 12} {'-' * 24} {'-' * 24}")
    for feature, mem, file in comparison:
        print(f"  {feature:<12} {mem:<24} {file}")


async def main():
    print("=" * 60)
    print("  Context Engineering SDK — 记忆存储示例")
    print("=" * 60)

    # MemoryStore 演示
    await demo_memory_store()

    # FileStore 演示
    with tempfile.TemporaryDirectory() as tmpdir:
        store_dir = os.path.join(tmpdir, "ce_sessions")
        _, session_id, base_dir = await demo_file_store(store_dir)

        # 从磁盘恢复
        await demo_restore_from_file(base_dir, session_id)

        # 展示持久化文件内容
        file_path = Path(store_dir) / f"{session_id}.json"
        print("\n--- 持久化 JSON 文件内容（截取前 500 字符）---")
        content = file_path.read_text(encoding="utf-8")
        print(content[:500] + ("..." if len(content) > 500 else ""))

    # 对比两种存储
    await demo_compare_stores()


if __name__ == "__main__":
    asyncio.run(main())
