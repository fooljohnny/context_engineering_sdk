# Context Engineering SDK 架构设计文档

> 版本：0.1.0 | 更新日期：2026-02-25

---

## 目录

1. [项目概述](#1-项目概述)
2. [业务架构](#2-业务架构)
3. [代码架构](#3-代码架构)
4. [部署架构](#4-部署架构)
5. [信息架构](#5-信息架构)
6. [核心时序图](#6-核心时序图)

---

## 1. 项目概述

Context Engineering SDK 是一个面向 AI Agent 的上下文工程 Python SDK，提供结构化上下文建模、证据管理和上下文组装能力。SDK 将传统的"Prompt 拼接"升级为一套可裁剪、可复用、可解释、可治理的上下文建设流水线（Pipeline），核心流程为：

**Ingest → Derive → Prune → Render → Assemble**

### 核心价值

| 能力 | 说明 |
|------|------|
| 可裁剪 | 按 priority / TTL / token 预算策略精细裁剪上下文块 |
| 可复用 | 证据与上下文块可跨 turn 引用，避免重复装载 |
| 可解释 | 每次输出可追溯到 evidence_id，链接工具调用与模型使用 |
| 可治理 | 支持缓存、A/B 测试、回放评测、归因分析 |

---

## 2. 业务架构

业务架构描述 SDK 在 AI Agent 系统中的定位以及它所服务的业务能力域。

```mermaid
graph TB
    subgraph 用户层["用户层"]
        U[终端用户 / 开发者]
    end

    subgraph 应用层["应用层 — AI Agent 应用"]
        A1[对话管理]
        A2[任务规划]
        A3[工具编排]
        A4[知识检索 RAG]
    end

    subgraph SDK层["能力层 — Context Engineering SDK"]
        direction TB
        B1["会话管理<br/>Session Lifecycle"]
        B2["证据采集<br/>Evidence Ingestion"]
        B3["上下文派生<br/>Block Derivation"]
        B4["预算裁剪<br/>Token Pruning"]
        B5["渲染 & 组装<br/>Render & Assemble"]
        B6["可观测性<br/>Event Observability"]
    end

    subgraph 基础设施层["基础设施层"]
        C1["存储后端<br/>Memory / File / Custom"]
        C2["LLM 服务<br/>LlmAdapter"]
        C3["外部工具<br/>MCP / API / RAG"]
    end

    U -->|自然语言| A1
    A1 -->|prepare_turn| B1
    A2 -->|task_state| B3
    A3 -->|tool_result| B2
    A4 -->|rag_doc| B2

    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 -->|AssembledInput| A1
    B1 --> B6

    B1 --- C1
    B5 --- C2
    B2 --- C3
```

### 业务能力域说明

| 能力域 | 职责 | SDK 入口 |
|--------|------|----------|
| 会话管理 | 创建、加载、持久化会话文档，管理消息追加与版本控制 | `ContextEngine.prepare_turn()` |
| 证据采集 | 将外部信息（RAG 文档、工具结果、LLM 输出）统一存证，支持去重与脱敏 | `EvidenceIngestor.ingest()` |
| 上下文派生 | 从会话消息、任务状态、证据库派生出带优先级的上下文块 | `BlockDeriver.derive()` |
| 预算裁剪 | 按 must/high/medium/low 优先级和 token 预算进行贪心裁剪 | `Pruner.prune()` |
| 渲染 & 组装 | 解析 refs、渲染证据片段、按类型分组组装为最终模型输入 | `Renderer` + `Assembler` |
| 可观测性 | 通过 EventBus 发布结构化事件，支持审计、回放与排障 | `EventBus.emit()` |

---

## 3. 代码架构

### 3.1 包结构总览

```mermaid
graph TB
    subgraph PKG["context_engineering_sdk"]
        direction TB

        INIT["__init__.py<br/>公共 API 导出"]
        ENGINE["engine.py<br/>DefaultContextEngine<br/>create_context_engine()"]
        CONFIG["config.py<br/>RuntimeConfig<br/>BudgetConfig / PruneConfig / ..."]

        subgraph CORE["core/"]
            direction LR
            TYPES["types.py<br/>枚举 & 数据类"]
            ERRORS["errors.py<br/>异常层次"]
            ID["id_generator.py<br/>UuidV4Generator"]
            CLK["clock.py<br/>SystemClock"]
            HASH["hasher.py<br/>Sha256Hasher"]
            REDACT["redactor.py<br/>RegexRedactor"]
            REF["ref_selector.py<br/>RefSelector"]
            TOK["token_estimator.py<br/>CharBasedEstimator"]
            MIG["migrator.py<br/>Schema 迁移"]
        end

        subgraph BUILDER["builder/"]
            direction LR
            ING["ingestor.py<br/>DefaultEvidenceIngestor"]
            DER["deriver.py<br/>DefaultBlockDeriver"]
            PRU["pruner.py<br/>GreedyPriorityPruner"]
            REN["renderer.py<br/>DefaultRenderer"]
            ASM["assembler.py<br/>DefaultAssembler"]
            SUM["summarizer.py<br/>DefaultSummarizer"]
        end

        subgraph STORE["store/"]
            direction LR
            BASE["base.py<br/>Store Protocol"]
            MEM["memory.py<br/>MemoryStore"]
            FILE["file.py<br/>FileStore"]
        end

        subgraph OBS["observability/"]
            EB["event_bus.py<br/>InMemoryEventBus"]
        end

        subgraph INT["integrations/"]
            PLACEHOLDER["__init__.py<br/>框架适配器（预留）"]
        end
    end

    ENGINE --> BUILDER
    ENGINE --> STORE
    ENGINE --> OBS
    ENGINE --> CONFIG
    BUILDER --> CORE
    STORE --> CORE
```

### 3.2 核心类依赖关系

```mermaid
classDiagram
    class ContextEngine {
        <<Protocol>>
        +prepare_turn(session_id, user_message, config) PrepareResult
        +commit_assistant_message(session_id, message, refs) PutResult
        +commit_assistant_chunk(session_id, chunk, index) void
        +finalize_assistant_message(session_id, refs) PutResult
        +record_tool_call(session_id, tool_call, evidence_ids) PutResult
        +record_model_usage(session_id, usage, evidence_id) PutResult
    }

    class DefaultContextEngine {
        -Store _store
        -TokenEstimator _tok
        -LlmAdapter _llm
        -EvidenceIngestor _ingestor
        -BlockDeriver _deriver
        -Pruner _pruner
        -Renderer _renderer
        -Assembler _assembler
        -Summarizer _summarizer
        -EventBus _event_bus
        -IdGenerator _id_gen
        -Clock _clock
        -Hasher _hasher
        -Redactor _redactor
        -RefSelector _ref_selector
    }

    class Store {
        <<Protocol>>
        +get_session(session_id) SessionDocument
        +put_session(session_id, doc) PutResult
        +append_messages(session_id, messages) PutResult
        +put_evidence(session_id, evidence) str
        +list_evidences(session_id, filter) list~Evidence~
        +put_context_blocks(session_id, blocks) PutResult
    }

    class EvidenceIngestor {
        <<Protocol>>
        +ingest(session_id, content, source, type, ...) tuple
    }

    class BlockDeriver {
        <<Protocol>>
        +derive(session_doc, evidences, options) list~ContextBlock~
    }

    class Pruner {
        <<Protocol>>
        +prune(blocks, budget, rules) PruneResult
    }

    class Renderer {
        <<Protocol>>
        +render_block(block, resolver, hints) RenderedBlock
    }

    class Assembler {
        <<Protocol>>
        +assemble(messages, rendered_blocks, model_hint) AssembledInput
    }

    class Summarizer {
        <<Protocol>>
        +should_summarize(session, config) bool
        +summarize(messages, summary, config) Summary
    }

    class EventBus {
        <<Protocol>>
        +emit(event) void
        +on(event_type, handler) void
        +off(event_type, handler) void
        +on_all(handler) void
    }

    class TokenEstimator {
        <<Protocol>>
        +estimate_text(text, model_hint) int
        +estimate_message(role, content) int
        +estimate_messages(messages) int
    }

    class LlmAdapter {
        <<Protocol>>
        +generate(request) LlmResponse
    }

    ContextEngine <|.. DefaultContextEngine : implements
    DefaultContextEngine --> Store
    DefaultContextEngine --> EvidenceIngestor
    DefaultContextEngine --> BlockDeriver
    DefaultContextEngine --> Pruner
    DefaultContextEngine --> Renderer
    DefaultContextEngine --> Assembler
    DefaultContextEngine --> Summarizer
    DefaultContextEngine --> EventBus
    DefaultContextEngine --> TokenEstimator
    DefaultContextEngine --> LlmAdapter

    Store <|.. MemoryStore : implements
    Store <|.. FileStore : implements

    class MemoryStore {
        -dict _sessions
    }
    class FileStore {
        -Path _base_dir
        -dict _versions
    }
```

### 3.3 Pipeline 组件职责

| 组件 | Protocol | 默认实现 | 职责 |
|------|----------|----------|------|
| Ingestor | `EvidenceIngestor` | `DefaultEvidenceIngestor` | 接收外部数据，去重（hash）、脱敏（regex）后存为 Evidence |
| Deriver | `BlockDeriver` | `DefaultBlockDeriver` | 从 Session 的 messages/task_state/evidences 派生 ContextBlock |
| Pruner | `Pruner` | `GreedyPriorityPruner` | 按优先级排序，在 token 预算内贪心选取 block |
| Renderer | `Renderer` | `DefaultRenderer` | 解析 block 上的 refs，通过 EvidenceResolver 加载内容片段 |
| Assembler | `Assembler` | `DefaultAssembler` | 将 rendered blocks 按类型分组，与 messages 合并为 AssembledInput |
| Summarizer | `Summarizer` | `DefaultSummarizer` | 判断是否需要摘要压缩，调用 LLM 生成滚动摘要 |

---

## 4. 部署架构

Context Engineering SDK 是一个嵌入式 Python 库（Library），不作为独立服务部署。以下展示 SDK 在典型 AI Agent 系统中的部署拓扑。

### 4.1 单体部署模式

```mermaid
graph TB
    subgraph Client["客户端"]
        WEB["Web / CLI / API Client"]
    end

    subgraph AppServer["应用服务器"]
        subgraph AgentProcess["Agent 进程"]
            APP["Agent 应用逻辑"]
            SDK["Context Engineering SDK"]
            subgraph SDKInternal["SDK 内部"]
                ENG["ContextEngine"]
                PIPE["Pipeline<br/>Ingest→Derive→Prune→Render→Assemble"]
                MSTORE["MemoryStore"]
                EBUS["InMemoryEventBus"]
            end
        end
    end

    subgraph External["外部服务"]
        LLM["LLM API<br/>(OpenAI / 自部署)"]
        TOOLS["工具服务<br/>(MCP / REST API)"]
        RAG["RAG 服务<br/>(向量数据库)"]
    end

    WEB -->|HTTP / WebSocket| APP
    APP --> SDK
    SDK --> LLM
    SDK --> TOOLS
    SDK --> RAG
```

### 4.2 生产部署模式（带持久化）

```mermaid
graph TB
    subgraph LB["负载均衡"]
        NGINX["Nginx / ALB"]
    end

    subgraph Cluster["应用集群"]
        subgraph Node1["Node 1"]
            A1["Agent + SDK"]
        end
        subgraph Node2["Node 2"]
            A2["Agent + SDK"]
        end
        subgraph NodeN["Node N"]
            AN["Agent + SDK"]
        end
    end

    subgraph Storage["持久化层"]
        FS["FileStore<br/>(本地 / NFS / S3)"]
        DB["自定义 Store<br/>(Redis / PostgreSQL / ...)"]
    end

    subgraph ExternalSvc["外部服务集群"]
        LLM["LLM 网关"]
        MCP["MCP Server 集群"]
        VDB["向量数据库"]
    end

    subgraph Monitor["可观测性"]
        LOG["日志平台<br/>(EventBus → Logger)"]
        METRICS["指标系统<br/>(Prometheus / Grafana)"]
        TRACE["链路追踪<br/>(Jaeger / OpenTelemetry)"]
    end

    NGINX --> Node1
    NGINX --> Node2
    NGINX --> NodeN

    Node1 --> FS
    Node1 --> DB
    Node2 --> FS
    Node2 --> DB
    NodeN --> FS
    NodeN --> DB

    Node1 --> LLM
    Node1 --> MCP
    Node1 --> VDB

    Node1 -.->|Event| LOG
    Node1 -.->|Metrics| METRICS
    Node1 -.->|Trace| TRACE
```

### 4.3 部署要求

| 维度 | 要求 |
|------|------|
| Python 版本 | >= 3.11 |
| 依赖 | 零外部运行时依赖（仅标准库） |
| 开发依赖 | pytest >= 7.0, pytest-asyncio >= 0.23 |
| 存储 | 内置 MemoryStore（测试用）和 FileStore（本地 JSON），生产建议实现自定义 Store |
| LLM | 用户自行实现 `LlmAdapter` Protocol，对接任意 LLM 服务 |

---

## 5. 信息架构

信息架构展示系统中核心数据实体的结构与关联关系。

### 5.1 数据模型全景

```mermaid
erDiagram
    SessionDocument {
        string schema_version
    }

    Session {
        string session_id PK
    }

    Meta {
        string locale
        string created_at
        string updated_at
    }

    Actor {
        string user_id
        string user_role
    }

    AgentInfo {
        string agent_id PK
        string name
        string version
    }

    Message {
        enum role "system|user|assistant|tool"
        string content
        string at
    }

    Author {
        enum kind "user|agent|tool|system"
        string id
    }

    Ref {
        string evidence_id FK
        string selector
    }

    Summary {
        string content
        string updated_at
    }

    MessageIndexRange {
        int from_index
        int to_index
    }

    TaskState {
    }

    TodoList {
    }

    Task {
        string task_id PK
        string name
        enum status "pending|in_progress|completed|failed|cancelled"
        string error
    }

    ToolState {
    }

    ToolCall {
        string tool_call_id PK
        string tool
        enum type "tool|skill|function_call"
        string called_at
        enum status "success|timeout|forbidden|not_found|error"
        int duration_ms
        string task_id FK
    }

    ToolProvider {
        enum kind "builtin|mcp|other"
        string name
        string uri
    }

    ModelUsage {
        string model_usage_id PK
        string provider
        string model
        enum stage "route|plan|tool_call|answer|summarize|other"
        int prompt_tokens
        int completion_tokens
        int total_tokens
        int latency_ms
        enum status "success|error"
        string task_id FK
    }

    Evidence {
        string evidence_id PK
        enum type "rag_doc|tool_result|skill_output|llm_output|user_input|other"
        string content
        float confidence
    }

    EvidenceSource {
        enum kind "rag|tool|skill|llm|user|system"
        string name
        string uri
    }

    EvidenceLinks {
        string model_usage_id FK
        string tool_call_id FK
    }

    ContextBlock {
        string block_id PK
        enum block_type "instruction|conversation|state|plan|evidence|memory"
        enum priority "must|high|medium|low"
        int token_estimate
        string content
    }

    SessionDocument ||--|| Session : contains
    SessionDocument ||--o| Meta : contains
    SessionDocument ||--o{ Evidence : "evidences (dict)"
    SessionDocument ||--o{ ContextBlock : "context_blocks (list)"

    Meta ||--o| Actor : has
    Actor ||--o| AgentInfo : has

    Session ||--o{ Message : messages
    Session ||--o| Summary : summary
    Session ||--|| TaskState : task_state
    Session ||--|| ToolState : tool_state
    Session ||--o{ ModelUsage : model_usage

    Message ||--o| Author : author
    Message ||--o{ Ref : refs

    Summary ||--|| MessageIndexRange : message_index_range

    TaskState ||--|| TodoList : todo_list
    TodoList ||--o{ Task : tasks
    Task ||--o{ Evidence : "result_evidence_ids"

    ToolState ||--o{ ToolCall : tool_calls
    ToolCall ||--|| ToolProvider : provider
    ToolCall ||--o{ Evidence : "result_evidence_ids"

    Evidence ||--|| EvidenceSource : source
    Evidence ||--|| EvidenceLinks : links

    ContextBlock ||--o{ Ref : refs
    Ref }o--|| Evidence : references
```

### 5.2 核心数据流

```mermaid
flowchart LR
    subgraph Input["输入"]
        RAW["原始数据<br/>(RAG/Tool/LLM/User)"]
    end

    subgraph Ingestion["证据采集"]
        DEDUP["去重<br/>(Hash)"]
        REDACT["脱敏<br/>(Regex)"]
        STORE_EV["存储 Evidence"]
    end

    subgraph Derivation["上下文派生"]
        CONV["对话块<br/>Conversation"]
        STATE["状态块<br/>State"]
        EVBLK["证据块<br/>Evidence"]
    end

    subgraph Assembly["裁剪 & 组装"]
        PRUNE["优先级裁剪"]
        RENDER["Ref 解析渲染"]
        ASM["分类组装"]
    end

    subgraph Output["输出"]
        AI["AssembledInput<br/>(parts + total_tokens)"]
    end

    RAW --> DEDUP --> REDACT --> STORE_EV
    STORE_EV --> EVBLK
    CONV --> PRUNE
    STATE --> PRUNE
    EVBLK --> PRUNE
    PRUNE --> RENDER --> ASM --> AI
```

### 5.3 Block 优先级与裁剪规则

| 优先级 | Rank | 裁剪行为 | 典型来源 |
|--------|------|----------|----------|
| `must` | 0 | 永不丢弃，超预算则抛 `BudgetExceededError` | system message |
| `high` | 1 | 优先保留 | 最近 3 条消息、摘要、任务状态、高置信证据 |
| `medium` | 2 | 预算允许时保留 | 最近 4-10 条消息、中等置信证据 |
| `low` | 3 | 预算紧张时最先丢弃 | 早期历史消息、低置信证据 |

---

## 6. 核心时序图

### 6.1 prepare_turn 完整流程

此时序图展示 SDK 的核心方法 `prepare_turn()` 的完整执行流程。

```mermaid
sequenceDiagram
    autonumber
    participant App as Agent 应用
    participant Engine as DefaultContextEngine
    participant Store as Store
    participant Summarizer as Summarizer
    participant Deriver as BlockDeriver
    participant Pruner as Pruner
    participant Renderer as Renderer
    participant Assembler as Assembler
    participant EventBus as EventBus
    participant LLM as LlmAdapter

    App->>Engine: prepare_turn(session_id, user_message, config)
    activate Engine

    Engine->>Store: get_session(session_id)
    alt Session 不存在
        Store-->>Engine: None
        Engine->>Store: put_session(session_id, new_doc)
        Engine->>EventBus: emit("SessionLoaded", created=True)
    else Session 已存在
        Store-->>Engine: SessionDocument
        Engine->>EventBus: emit("SessionLoaded", created=False)
    end

    Engine->>Store: append_messages(session_id, [user_message])
    Engine->>Store: get_session(session_id)
    Store-->>Engine: updated SessionDocument
    Engine->>EventBus: emit("MessageAppended")

    alt 摘要功能已启用
        Engine->>Summarizer: should_summarize(session, config)
        Summarizer-->>Engine: true / false
        opt 需要摘要
            Engine->>Summarizer: summarize(messages, existing_summary, config)
            Summarizer->>LLM: generate(LlmRequest)
            LLM-->>Summarizer: LlmResponse
            Summarizer-->>Engine: Summary
            Engine->>Store: patch_session(session_id, {summary})
            Engine->>EventBus: emit("SummaryGenerated")
        end
    end

    Engine->>Store: list_evidences(session_id)
    Store-->>Engine: list[Evidence]

    Engine->>Deriver: derive(session_doc, evidences)
    Deriver-->>Engine: list[ContextBlock]
    Engine->>EventBus: emit("BlocksDerived", count=N)

    Engine->>Pruner: prune(blocks, budget, rules)
    Pruner-->>Engine: PruneResult(kept, dropped, decisions)
    Engine->>EventBus: emit("PruneCompleted", kept=K, dropped=D)

    Engine->>Store: put_context_blocks(session_id, kept_blocks)

    loop 每个 kept block
        Engine->>Renderer: render_block(block, resolver)
        Renderer->>Store: get_evidence(session_id, evidence_id)
        Store-->>Renderer: Evidence
        Renderer-->>Engine: RenderedBlock
    end

    Engine->>Assembler: assemble(messages, rendered_blocks, model_hint)
    Assembler-->>Engine: AssembledInput
    Engine->>EventBus: emit("Assembled", total_tokens=T)

    Engine-->>App: PrepareResult(assembled_input, report)
    deactivate Engine
```

### 6.2 多轮对话完整生命周期

此时序图展示一次完整的多轮对话流程，包括 prepare → LLM 调用 → commit 三个阶段。

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant App as Agent 应用
    participant Engine as ContextEngine
    participant LLM as LLM 服务

    User->>App: 发送消息

    rect rgb(230, 245, 255)
        note right of App: Phase A — prepare_turn
        App->>Engine: prepare_turn(session_id, user_msg, config)
        Engine-->>App: PrepareResult { assembled_input, report }
    end

    rect rgb(255, 245, 230)
        note right of App: LLM 调用（SDK 外部）
        App->>LLM: 调用 LLM(assembled_input)
        LLM-->>App: assistant_response
    end

    rect rgb(230, 255, 230)
        note right of App: Phase B — commit
        App->>Engine: commit_assistant_message(session_id, assistant_msg)
        Engine-->>App: PutResult
    end

    App-->>User: 返回响应
```

### 6.3 流式输出生命周期

```mermaid
sequenceDiagram
    autonumber
    participant User as 用户
    participant App as Agent 应用
    participant Engine as ContextEngine
    participant LLM as LLM 服务

    User->>App: 发送消息

    App->>Engine: prepare_turn(session_id, user_msg, config)
    Engine-->>App: PrepareResult

    App->>LLM: stream(assembled_input)

    loop 流式 chunk
        LLM-->>App: chunk[i]
        App->>Engine: commit_assistant_chunk(session_id, chunk, i)
        App-->>User: 实时推送 chunk
    end

    App->>Engine: finalize_assistant_message(session_id, refs)
    Engine-->>App: PutResult
```

### 6.4 证据采集与工具调用记录

```mermaid
sequenceDiagram
    autonumber
    participant App as Agent 应用
    participant Engine as ContextEngine
    participant Ingestor as EvidenceIngestor
    participant Hasher as Hasher
    participant Redactor as Redactor
    participant Store as Store
    participant EventBus as EventBus

    App->>Engine: record_tool_call(session_id, tool_call, evidence_ids)
    Engine->>Store: append_tool_calls(session_id, [tool_call])
    Engine->>EventBus: emit("ToolCallCompleted")

    note over App: 工具返回结果后, 采集证据

    App->>Ingestor: ingest(session_id, content, source, type)
    activate Ingestor

    opt 去重检查
        Ingestor->>Hasher: digest(content)
        Hasher-->>Ingestor: content_hash
        Ingestor->>Ingestor: 检查 dedup_cache
    end

    opt 脱敏处理
        Ingestor->>Redactor: redact(content)
        Redactor-->>Ingestor: RedactResult
    end

    Ingestor->>Store: put_evidence(session_id, evidence)
    Store-->>Ingestor: evidence_id
    Ingestor-->>App: (Evidence, RedactionRecord)
    deactivate Ingestor
```

### 6.5 EventBus 事件流

以下列出 `prepare_turn` 过程中发布的全部事件类型及顺序：

```mermaid
graph LR
    E1["SessionLoaded"] --> E2["MessageAppended"]
    E2 --> E3["SummaryGenerated<br/>(可选)"]
    E3 --> E4["BlocksDerived"]
    E4 --> E5["PruneCompleted"]
    E5 --> E6["Assembled"]

    style E1 fill:#e1f5fe
    style E2 fill:#e1f5fe
    style E3 fill:#fff3e0
    style E4 fill:#e8f5e9
    style E5 fill:#e8f5e9
    style E6 fill:#f3e5f5
```

| 事件类型 | 触发时机 | payload 字段 |
|----------|----------|-------------|
| `SessionLoaded` | 会话加载/创建后 | `{created: bool}` |
| `MessageAppended` | 用户消息追加后 | `{role: str}` |
| `SummaryGenerated` | 摘要生成成功后 | — |
| `Degraded` | 摘要失败降级时 | `{reason: str}` |
| `BlocksDerived` | 上下文块派生完成 | `{count: int}` |
| `PruneCompleted` | 裁剪完成后 | `{kept: int, dropped: int}` |
| `Error` | must 块超预算时 | `{reason: str}` |
| `Assembled` | 最终组装完成 | `{total_tokens: int}` |
| `AssistantChunkReceived` | 流式 chunk 接收 | `{chunk_index, chunk_length}` |
| `AssistantMessageFinalized` | 助手消息最终化 | `{content_length: int}` |
| `ToolCallCompleted` | 工具调用记录 | `{tool, status, duration_ms}` |
| `ModelUsageRecorded` | 模型用量记录 | `{model, total_tokens}` |
