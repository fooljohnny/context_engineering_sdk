# Context Engineering SDK 示例

本目录包含 Context Engineering SDK 的完整示例集，涵盖基础用法和主流 Agent 框架集成。

## 示例列表

### 基础功能示例

| 示例 | 文件 | 描述 |
|------|------|------|
| 多轮对话 | `basic_multi_turn.py` | 多轮 `prepare_turn` -> `commit_assistant_message` 交互，观察 token 预算与裁剪决策 |
| 工具调用与证据 | `tool_usage.py` | 工具调用记录、MCP 工具调用、证据入库与溯源 |
| 个性化记忆 | `personalized_memory.py` | 用户偏好存储、跨会话记忆携带、Memory Block 注入 |
| 持久化存储 | `memory_storage.py` | FileStore 磁盘持久化、跨进程恢复、MemoryStore vs FileStore 对比 |
| 流式对话 | `streaming_conversation.py` | 流式 chunk 提交、消息合并、模型用量记录、事件监控 |

### Agent 框架集成示例

| 示例 | 文件 | 描述 |
|------|------|------|
| LangChain | `langchain_integration.py` | Callback 接入、RAG 检索证据化、工具调用记录、多轮 RAG 对话 |
| AutoGen | `autogen_integration.py` | 多 Agent 消息归属、任务分解与归因、工具调用溯源 |
| LlamaIndex | `llamaindex_integration.py` | Node/Document 证据化、检索记录、文档版本治理 |

## 运行方式

```bash
# 安装 SDK
pip install -e .

# 运行任意示例
python examples/basic_multi_turn.py
python examples/tool_usage.py
python examples/personalized_memory.py
python examples/memory_storage.py
python examples/streaming_conversation.py
python examples/langchain_integration.py
python examples/autogen_integration.py
python examples/llamaindex_integration.py
```

所有示例使用 Mock 实现，无需外部依赖（无需 API Key），可直接运行。

## 示例要点

### 多轮对话（basic_multi_turn.py）

```python
engine = create_context_engine(store, token_estimator, llm_adapter)

# 每轮：prepare -> 调用 LLM -> commit
result = await engine.prepare_turn(session_id, user_message, config)
# ... 调用 LLM 获取回复 ...
await engine.commit_assistant_message(session_id, assistant_message)
```

核心流程：`prepare_turn()` 内部自动完成消息追加、摘要检查、证据加载、块派生、裁剪和装配。

### 工具调用（tool_usage.py）

```python
# 1. 工具结果作为证据入库
evidence, _ = await ingestor.ingest(
    session_id, content=tool_output,
    source=EvidenceSource(kind=SourceKind.TOOL, name="getOrder"),
    evidence_type=EvidenceType.TOOL_RESULT,
    links=EvidenceLinks(tool_call_id=tc_id),
)

# 2. 记录工具调用元数据
await engine.record_tool_call(session_id, tool_call)
```

### 个性化记忆（personalized_memory.py）

```python
# 偏好作为证据存储
pref = await ingestor.ingest(session_id, "用户偏好中文", ...)

# 构建 Memory Block 注入上下文
memory_block = ContextBlock(
    block_type=BlockType.MEMORY,
    priority=Priority.HIGH,
    content="用户偏好...",
    refs=[Ref(evidence_id=pref.evidence_id)],
)
```

### 流式对话（streaming_conversation.py）

```python
# 逐 chunk 提交
async for chunk in llm.stream(assembled_input):
    await engine.commit_assistant_chunk(session_id, chunk.text, chunk.index)

# 合并为完整消息
await engine.finalize_assistant_message(session_id)
```

### 框架集成模式

所有框架集成遵循相同的适配模式：

1. **Mapper** — 将框架对象映射为 CE SDK 类型
2. **Recorder** — 在框架回调/钩子中调用 SDK 记录
3. **Injector** — 将 `assembled_input` 注入框架的 LLM 调用

```
Framework Lifecycle ──┐
                      ├──→ CE Adapter ──→ CE SDK Store
                      │       ↑
                      │       │ assembled_input
                      └───────┘
```
