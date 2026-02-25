# 上下文结构化建模

**1 上下文结构化建模的目标**

上下文结构化建模的目标，是把“要塞进 prompt 的所有信息”从散乱文本，变成可裁剪、可复用、可解释的结构化单元。

* 可裁剪：可按 priority、ttl、token 预算等策略精细裁剪，保留重要信息。
* 可复用：把通用规则、工具结果、检索结果沉淀成可复用单元或证据单元，跨节点复用（引用），只更新一处。
* 可解释：每次输出能指到用过的 evidence\_ids、状态与冲突决策，便于排障与复盘。
* 可治理：能做缓存、A/B、回放评测、坏例归因（到底是缺证据还是裁剪策略错）。

**2 上下文结构化建模定义（简化版）**

这版模型把“会话状态（session）”作为唯一核心；把“来源证据（evidences）”作为统一溯源入口；把“可裁剪上下文块（context_blocks）”作为最终可装配单元。

- **命名统一**：不再在 `meta` 里放 `conversation_id`；会话唯一标识放在 `session.session_id`（或 `session.id`，二选一即可）。
- **去掉运行态组装字段**：不再需要 `meta.policy`、`execution/runtime`、`prompt_layers`、`snapshots` 这类“为了拼 system prompt/运行时开关”的结构。
- **用量与调用历史进入会话**：`model_usage`、工具/skills 调用、以及它们产出的证据，都以“会话历史记录”的方式沉淀在 `session` + `evidences` 中。

推荐结构如下（JSON 仅示意）：

```json
{
  "schema_version": "1.0",
  "meta": {
    "locale": "zh-CN",
    "created_at": "2026-01-20T10:00:00Z",
    "updated_at": "2026-01-20T10:05:00Z",
    "actor": {
      "user_id": "xxxx",
      "user_role": "developer",
      "agent": { "agent_id": "agent-001", "name": "my-agent", "version": "2026-02-10" }
    }
  },
  "session": {
    "session_id": "uuid",
    "messages": [
      { "role": "system", "author": { "kind": "agent", "id": "agent-001" }, "content": "你是一个XXX专家…", "at": "2026-02-10T10:05:00Z" },
      { "role": "user", "author": { "kind": "user", "id": "xxxx" }, "content": "包周期续订的接口是？", "at": "2026-02-10T10:06:00Z" },
      {
        "role": "assistant",
        "author": { "kind": "agent", "id": "agent-001" },
        "content": "…（本轮最终回答）",
        "at": "2026-02-10T10:07:00Z",
        "refs": [{ "evidence_id": "uuid", "selector": "lines:12-18,chars:120-260" }]
      }
    ],
    "summary": {
      "content": "截至目前：…（会话级滚动摘要，用于压缩）",
      "updated_at": "2026-02-10T10:07:00Z",
      "message_index_range": { "from": 1, "to": 18 }
    },
    "task_state": {
      "todo_list": {
        "tasks": [
          {
            "task_id": "uuid",
            "name": "…",
            "depends_on": [],
            "status": "pending",
            "result_evidence_ids": [],
            "error": ""
          }
        ]
      }
    },
    "tool_state": {
      "tool_calls": [
        {
          "task_id": "uuid",
          "tool_call_id": "uuid",
          "tool": "…",
          "provider": {
            "kind": "builtin|mcp|other",
            "name": "mcp_server_name_or_registry",
            "uri": "可选：mcp://server 或 https://..."
          },
          "type": "tool|skill|function_call",
          "called_at": "…",
          "args_digest": {},
          "status": "success|timeout|forbidden|not_found|error",
          "duration_ms": 0,
          "result_evidence_ids": []
        }
      ]
    },
    "model_usage": [
      {
        "model_usage_id": "uuid",
        "task_id": "uuid",
        "stage": "route|plan|tool_call|answer|other",
        "provider": "pangu|qwen|deepseek|...",
        "model": "pangu-V-7B",
        "params": { "temperature": 0.2 },
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "first_token_latency_ms": 0,
        "latency_ms": 0,
        "status": "success|error",
        "error": ""
      }
    ]
  },
  "evidences": {
    "uuid": {
      "evidence_id": "uuid",
      "type": "rag_doc|tool_result|skill_output|llm_output|user_input|other",
      "source": {
        "kind": "rag|tool|skill|llm|user|system",
        "name": "wiki|getOrder|mySkill|pangu|...",
        "uri": "https://..."
      },
      "content": "可引用片段或结构化摘要（已脱敏）",
      "confidence": 0.92,
      "metadata": { "dataField": [], "knowledgeType": [], "tags": [] },
      "links": {
        "model_usage_id": "uuid",
        "tool_call_id": "uuid"
      }
    }
  },
  "context_blocks": [
    {
      "block_id": "uuid",
      "block_type": "instruction|conversation|state|plan|evidence|memory",
      "priority": "must|high|medium|low",
      "token_estimate": 350,
      "content": "（可选）派生后的可直接装配内容",
      "refs": [
        { "evidence_id": "uuid", "selector": "可选：指向 evidence.content 的片段/字段路径" }
      ]
    }
  ]
}
```

字段说明（只保留核心内聚能力）：

- **`schema_version`**：上下文模型结构版本号。用于演进兼容、迁移与回放解析（不同版本可按策略降级/升级解析）。
- **`meta.actor.agent`**：本次会话默认 Agent 标识（可选）。用于审计、回放、A/B 与跨框架迁移时的“产出归属”（建议至少包含 `agent_id`，可选 `name/version`）。
- **`session.messages`**：把 system/user/assistant/tool 统一记录在会话里（用于回放与审计）。其中 `role=assistant` 的 message 就是**每一轮对用户的最终回答**。如需可解释/可引用，可在 message 上增加可选 `refs`（指向 `evidences`），或把引用关系放到 `context_blocks[].refs`。
- **`session.messages[].author`**：消息作者标识（可选）。建议形如 `{kind: user|agent|tool|system, id: "..."}`，用于多 Agent/多参与方场景下的可追溯与跨框架映射。
- **`session.summary`**：会话级滚动摘要（压缩锚点）。用于在长对话中替代较早的 messages，降低 token 消耗；建议记录摘要覆盖的消息范围，便于回放与增量更新。
- **`session.task_state.todo_list.tasks`**：ToDoList 是一组 task，task 的产出通过 `result_evidence_ids` 指向 `evidences`。
- **`session.tool_state.tool_calls`**：外部能力调用历史（工具/skills/检索服务RAG等）仍保留；建议在每次调用记录中写入 `task_id` 用于归因（一次调用由哪个 task 触发）。为便于还原 MCP 调用链路，建议增加 `provider` 字段（如 `provider.kind=mcp`，并记录 server/name/uri）；调用产物用 `result_evidence_ids` 统一落证据。
- **`session.model_usage`**：模型调用历史作为会话级历史数据；建议在每条调用记录中增加可选 `task_id` 用于把 token/latency 等用量归因到具体任务；可选增加 `first_token_latency_ms` 记录首 token 延迟（TTFB/first-token latency），与 `latency_ms`（整次调用端到端耗时）配套用于性能分析；模型产出的文本/结构结果应以 `evidences[type=llm_output]` 形式存证，并可通过 `links.model_usage_id` 回链。
- **`evidences`（dict）**：所有来源（RAG/工具/skills/LLM/用户输入）统一承载为“可引用证据”。
- **`context_blocks.refs`**：合并了原来的 `content_ref` + `evidence_ids`：一个 block 既能指向来源证据（溯源），也能表达“从证据的哪一部分取用”（selector）。

**3 上下文建模在Agent各环节的应用**

![](data:image/png;base64...)

* 用户输入

作用点：把原始自然语言作为“输入层”保留原样，避免后续多次改写造成语义漂移。

价值：可回放、可审计；后续所有派生（意图、任务、证据）都能追溯到这条 raw\_query。

* 意图识别

作用点：把“要做什么”从文本里抽出来，作为后续规划、检索、工具路由的统一入口。

价值：减少模型在长对话里“猜任务”；同一意图可复用同一套策略与模板。

说明：如果不保留 `dialog_state`，这些“意图/槽位/待确认问题”等派生信息可以用两种方式承载：

- 以 `context_blocks[block_type=state|plan]` 的形式承载（可裁剪、可装配）。
- 或以 `evidences[type=llm_output|other]` 的形式存证（便于审计与回放），再由 `context_blocks.refs` 选择性引用。

* 策略/元信息

作用点：用 meta.actor 等元信息约束“说给谁听”。（口径/合规策略建议作为固定系统提示词或产品侧配置，不放入上下文模型。）

价值：输出风格/合规口径稳定；同一问题对内对外可走不同答复与脱敏策略。

* 规划

作用点：把“怎么做”结构化为 `session.task_state.todo_list.tasks`（步骤、依赖、完成态），并明确哪些步骤需要检索/工具。

价值：复杂任务可控、可拆解、可回放；减少直接生成导致的漏步骤和逻辑跳跃。

* 检索

作用点：把检索（RAG）视为一次外部调用，记录到 `session.tool_state.tool_calls`（如 `tool=rag_search`，并写入 `task_id/args_digest/status/duration_ms/result_evidence_ids`）；命中的文档片段落为 `evidences[type=rag_doc]`。如需做版本/时效/领域标签等治理字段，可写入 `evidences[evidence_id].metadata`（或由上层派生生成可裁剪的 `context_blocks`）。

价值：检索调用可审计可观测（便于排障、重试、缓存与成本治理）；回答可“基于证据”；支持版本/时效治理与冲突检测，而不是把一堆文档片段混在一起。

* 工具调用

作用点：把外部能力调用（工具/skills/检索RAG等）变成可审计的结构日志（入参摘要、状态、耗时、错误、证据指针、归因到 `task_id`）。如需区分调用由哪种接入层提供（例如 MCP），在 `tool_calls[].provider` 中记录 `kind/name/uri`，即可从日志中还原出 MCP server 与工具来源。

价值：可重试/回退；可定位问题在“模型推理”还是“工具数据”；可沉淀为可复用证据。

* 证据库沉淀

作用点：把检索/工具/skills/LLM 输出统一落到 `evidences`（dict），并带 `source/confidence/metadata` 与回链信息（如 `model_usage_id`、`tool_call_id`）。

价值：实现“证据统一入口”；后续任何上下文块都用 evidence\_id 引用，天然可解释、可去重、可复用。

* 生成候选上下文块（context\_blocks）

作用点：把可用信息切成 `context_blocks[]`（带 `priority/token_estimate` 等最少治理字段）。

价值：上下文变成“可裁剪的积木”；能按预算选择、按敏感分级过滤、按时效剔除、按优先级保底。

* 绑定证据依赖与派生定位（refs）

作用点：用 `context_blocks[].refs[]` 同时表达“基于哪些证据”以及“取证据的哪一段/哪一字段”（selector）。

价值：避免重复塞长文本；同一证据可用不同渲染策略（摘要/表格/时间线）；输出可解释到证据级别。

* 组装最终输入

作用点：把 `session.messages`（含 system/user 等）与筛选后的 `context_blocks` 装配为最终输入（并做去重、裁剪与渲染）。

价值：把“提示词拼接”升级为“上下文装配流水线”；质量稳定、成本可控。

* 对话输出（本轮最终回答）

作用点：将本轮对用户可见的最终回答写入 `session.messages[]`（`role=assistant`）。如需“可解释/可追溯”，可以在该条 message 上附带 `refs` 指向 `evidences`，或在 `context_blocks` 中记录引用关系。

价值：对话回放只依赖 messages；排障/审计再按 refs 追溯到证据与工具/模型调用。

* 模型生成与用量记录

作用点：将模型调用历史记录在 `session.model_usage[]`（可选记录 `task_id` 便于归因；可选记录 `first_token_latency_ms` 便于区分“首包慢”还是“生成慢”）；将模型输出落为 `evidences[type=llm_output]` 并回链到对应 `model_usage_id`。

价值：可做成本优化、回归评测、容量规划；badcase 可定位是“上下文供给”还是“模型能力/参数”。

* 对外输出

作用点：按系统固定策略/产品侧配置做最终口径与格式约束（是否引用证据、是否脱敏、是否给出操作步骤/免责声明等）。

价值：同一内部推理可输出多种外显形态；减少“内部信息外泄”与口径不一致。

**4 渐进式应用——最小可用（MVP）**

确保能达成可组装、可裁剪、可复用、可解释的基本目的，初步落地最小模型时建议只实现三件事：

- **会话状态**：`session.messages` + `session.task_state.todo_list.tasks`
- **证据统一入口**：`evidences`（dict），所有外部信息/模型输出都先落证据再引用
- **可裁剪上下文块**：`context_blocks`，只带 `priority/token_estimate/refs`

#### 4.1 MVP 字段“是否必须（Required）”标识

说明：这里的“必须”指 **为了达成 MVP 的三件事（会话状态/证据/可裁剪上下文块）**，该字段 **是否必须存在于存储结构中**。  
其中“可空”表示字段必须存在，但允许为空值（如 `[]`/`{}`）；“存在时必须”表示该对象一旦出现，其内部字段必须满足最低约束。

| 字段路径 | MVP 是否必须 | 备注 |
|---|---:|---|
| `schema_version` | 是 | 上下文模型结构版本号；用于兼容/迁移/回放解析 |
| `meta` | 否 | 可选：本地化、审计元信息；不影响 MVP 运转 |
| `meta.actor.agent` | 否 | 可选：默认 Agent 标识（多 Agent/审计/回放/A-B/迁移时很有用） |
| `meta.actor.agent.agent_id` | 是（存在时必须） | Agent 唯一标识（建议稳定且可跨系统映射） |
| `session` | 是（可空对象不允许） | MVP 的会话承载体 |
| `session.session_id` | 是 | 会话唯一标识，用于回放/索引/缓存 |
| `session.messages` | 是（可空：否） | 至少应包含本轮 `user` 输入；通常也会有固定 `system` 与 `assistant` 输出 |
| `session.messages[].role` | 是（存在时必须） | `system/user/assistant/tool` 等统一口径（可扩展但需可映射） |
| `session.messages[].content` | 是（存在时必须） | 原始输入/最终输出/工具可见文本等 |
| `session.messages[].author` | 否 | 可选：消息作者（user/agent/tool/system），用于多 Agent 与框架映射 |
| `session.messages[].at` | 否 | 建议保留：用于审计、压缩覆盖范围计算 |
| `session.summary` | 否 | 压缩能力用；不属于 MVP 必需字段 |
| `session.task_state` | 是（可空对象不允许） | MVP 的“规划/任务”承载体（可先全空实现） |
| `session.task_state.todo_list` | 是 | 任务列表容器 |
| `session.task_state.todo_list.tasks` | 是（可空：是） | MVP 可先为空数组，后续逐步接入规划 |
| `session.tool_state` | 否 | 可选：需要可观测/可重放时再接入 |
| `session.model_usage` | 否 | 可选：需要成本/性能治理时再接入 |
| `evidences` | 是（可空：是） | 证据统一入口，允许 `{}` |
| `evidences.{evidence_id}` | 否（存在时必须满足约束） | 只有当你开始落证据时才会出现条目 |
| `evidences.{evidence_id}.evidence_id` | 是（存在时必须） | 建议与 key 保持一致，便于一致性校验 |
| `evidences.{evidence_id}.type` | 是（存在时必须） | `rag_doc/tool_result/skill_output/llm_output/user_input/other` |
| `evidences.{evidence_id}.source.kind` | 是（存在时必须） | `rag/tool/skill/llm/user/system` |
| `evidences.{evidence_id}.source.name` | 否 | 建议保留：如 `wiki|getOrder|mySkill|pangu` |
| `evidences.{evidence_id}.content` | 否 | 允许不存 `content`，仅保留 `source` 用于“按来源取回”（但会降低可复现性与可解释性） |
| `context_blocks` | 是（可空：是） | 最终可装配单元，允许 `[]` |
| `context_blocks[].block_id` | 是（存在时必须） | 块唯一标识 |
| `context_blocks[].block_type` | 是（存在时必须） | `instruction/conversation/state/plan/evidence/memory` |
| `context_blocks[].priority` | 是（存在时必须） | `must/high/medium/low` |
| `context_blocks[].token_estimate` | 否 | 建议保留：用于预算裁剪 |
| `context_blocks[].content` | 否 | 可选：也可完全由 `refs` 派生渲染 |
| `context_blocks[].refs` | 否 | 可选：用于溯源与 selector 取片段 |

最小可用结构示例：

```json
{
  "schema_version": "1.0",
  "meta": {
    "actor": {
      "user_id": "xxxx",
      "user_role": "developer",
      "agent": { "agent_id": "agent-001", "name": "my-agent", "version": "2026-02-10" }
    }
  },
  "session": {
    "session_id": "uuid",
    "messages": [
      { "role": "system", "author": { "kind": "agent", "id": "agent-001" }, "content": "固定系统提示词版本号=2026-02-10", "at": "…" },
      { "role": "user", "author": { "kind": "user", "id": "xxxx" }, "content": "包周期续订的接口是？", "at": "…" },
      { "role": "assistant", "author": { "kind": "agent", "id": "agent-001" }, "content": "…（本轮最终回答）", "at": "…" }
    ],
    "summary": { "content": "", "updated_at": "", "message_index_range": { "from": 0, "to": 0 } },
    "task_state": { "todo_list": { "tasks": [] } },
    "tool_state": { "tool_calls": [] },
    "model_usage": []
  },
  "evidences": {},
  "context_blocks": []
}
```
