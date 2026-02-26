# Context Engineering SDK 示例

本目录包含 Context Engineering SDK 的完整示例集，涵盖基础用法和主流 AI Agent 框架集成。

## 示例列表

### 基础功能示例

| 示例 | 文件 | 描述 |
|------|------|------|
| 多轮对话 | `basic_multi_turn.py` | 多轮 `prepare_turn` -> `commit_assistant_message` 交互，观察 token 预算与裁剪决策 |
| 工具调用与证据 | `tool_usage.py` | 工具调用记录、MCP 工具调用、证据入库与溯源 |
| 个性化记忆 | `personalized_memory.py` | 用户偏好存储、跨会话记忆携带、Memory Block 注入 |
| 持久化存储 | `memory_storage.py` | FileStore 磁盘持久化、跨进程恢复、MemoryStore vs FileStore 对比 |
| 流式对话 | `streaming_conversation.py` | 流式 chunk 提交、消息合并、模型用量记录、事件监控 |

### Agent 框架集成示例（含跨会话记忆 + 长对话压缩）

每个框架示例均包含以下完整功能：
- **跨会话用户画像**：自动识别并持久化用户偏好（语言、风格、技术栈等）
- **长期记忆**：通过 `MemoryManager` 在新会话中注入历史偏好和会话摘要
- **长对话压缩**：当消息数超过阈值时自动触发 `Summarizer` 进行滚动摘要
- **工具调用与证据溯源**：完整记录工具调用、RAG 检索、模型用量

| 示例 | 文件 | 场景 |
|------|------|------|
| LangChain | `langchain_integration.py` | 智能客服：8 轮对话（RAG + 工具 + 偏好检测 + 压缩）→ 跨会话验证 |
| LangGraph | `langgraph_integration.py` | 技术助手：StateGraph 路由工作流（Router→Retriever→Tool→Generator）→ 跨会话验证 |
| AutoGen | `autogen_integration.py` | 编程任务：多 Agent 协作（Planner→Researcher→Coder）+ 任务归因 → 跨会话验证 |
| AgentScope | `agentscope_integration.py` | 数据分析：MsgHub 多 Agent（Coordinator→DataAnalyst→ReportWriter）→ 跨会话验证 |

## 运行方式

```bash
# 安装 SDK
pip install -e .

# 运行基础示例
python examples/basic_multi_turn.py
python examples/tool_usage.py
python examples/personalized_memory.py
python examples/memory_storage.py
python examples/streaming_conversation.py

# 运行框架集成示例（含跨会话记忆 + 长对话压缩）
python examples/langchain_integration.py
python examples/langgraph_integration.py
python examples/autogen_integration.py
python examples/agentscope_integration.py
```

所有示例使用 Mock 实现，无需外部依赖（无需 API Key），可直接运行。

## 跨会话记忆架构

框架集成示例使用 SDK 新增的 `memory/` 模块实现跨会话记忆：

```
MemoryManager
├── UserMemoryStore (Protocol)
│   ├── InMemoryUserMemoryStore (测试用)
│   └── FileUserMemoryStore (持久化)
├── add_preference(user_id, category, content)    # 保存用户偏好
├── add_user_fact(user_id, content)                # 保存用户画像
├── add_session_summary(user_id, session_id, ...)  # 保存会话摘要
└── build_memory_blocks(user_id)                   # 构建 ContextBlock 注入上下文
```

注入方式：通过 `RuntimeConfig.extra_context_blocks` 字段，在 `prepare_turn` 时
将记忆块注入 Pipeline，无需修改 Engine 核心逻辑。

## 每个框架示例的核心流程

```
Session 1 (多轮长对话)
  ├── 用户消息中检测偏好 → MemoryManager.add_preference()
  ├── 消息数超阈值 → Summarizer 自动压缩
  ├── 工具调用 / RAG 检索 → Evidence 存证
  └── 会话结束 → MemoryManager.add_session_summary()

Session 2 (跨会话验证)
  ├── MemoryManager.build_memory_blocks() → extra_context_blocks
  ├── prepare_turn 注入记忆块到上下文
  └── 验证：偏好 + 历史摘要 存在于 assembled_input 中
```

## 框架适配模式

所有框架集成遵循统一的适配模式：

1. **Adapter 类** — 将框架的生命周期事件映射为 CE SDK 操作
2. **prepare_turn** — 注入跨会话记忆 + 偏好检测
3. **工具/检索回调** — 通过 `EvidenceIngestor` 存证 + `record_tool_call` 记录
4. **commit** — 提交助手回复并关联 evidence refs
5. **Session 结束** — 保存会话摘要到 `MemoryManager`

```
Framework Lifecycle ──► CE Adapter ──► CE SDK Engine
                            │                │
                            ├─ MemoryManager ─┤ (cross-session)
                            ├─ Ingestor ──────┤ (evidence)
                            └─ EventBus ──────┘ (observability)
```
