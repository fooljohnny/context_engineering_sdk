"""Core data types aligned with the context engineering schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AuthorKind(str, Enum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


class EvidenceType(str, Enum):
    RAG_DOC = "rag_doc"
    TOOL_RESULT = "tool_result"
    SKILL_OUTPUT = "skill_output"
    LLM_OUTPUT = "llm_output"
    USER_INPUT = "user_input"
    OTHER = "other"


class SourceKind(str, Enum):
    RAG = "rag"
    TOOL = "tool"
    SKILL = "skill"
    LLM = "llm"
    USER = "user"
    SYSTEM = "system"


class BlockType(str, Enum):
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    STATE = "state"
    PLAN = "plan"
    EVIDENCE = "evidence"
    MEMORY = "memory"


class Priority(str, Enum):
    MUST = "must"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


PRIORITY_RANK: dict[Priority, int] = {
    Priority.MUST: 0,
    Priority.HIGH: 1,
    Priority.MEDIUM: 2,
    Priority.LOW: 3,
}


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    ERROR = "error"


class ProviderKind(str, Enum):
    BUILTIN = "builtin"
    MCP = "mcp"
    OTHER = "other"


class ToolType(str, Enum):
    TOOL = "tool"
    SKILL = "skill"
    FUNCTION_CALL = "function_call"


class ModelUsageStage(str, Enum):
    ROUTE = "route"
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    ANSWER = "answer"
    SUMMARIZE = "summarize"
    OTHER = "other"


class ModelUsageStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class Author:
    kind: AuthorKind
    id: str


@dataclass
class Ref:
    evidence_id: str
    selector: str | None = None


@dataclass
class Message:
    role: Role
    content: str
    author: Author | None = None
    at: str | None = None
    refs: list[Ref] | None = None


@dataclass
class MessageIndexRange:
    from_index: int
    to_index: int


@dataclass
class Summary:
    content: str
    updated_at: str
    message_index_range: MessageIndexRange


@dataclass
class Task:
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    result_evidence_ids: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class TodoList:
    tasks: list[Task] = field(default_factory=list)


@dataclass
class TaskState:
    todo_list: TodoList = field(default_factory=TodoList)


@dataclass
class ToolProvider:
    kind: ProviderKind
    name: str = ""
    uri: str = ""


@dataclass
class ToolCall:
    tool_call_id: str
    tool: str
    provider: ToolProvider
    type: ToolType = ToolType.TOOL
    called_at: str = ""
    args_digest: dict = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    duration_ms: int = 0
    result_evidence_ids: list[str] = field(default_factory=list)
    task_id: str | None = None


@dataclass
class ToolState:
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ModelUsage:
    model_usage_id: str
    provider: str
    model: str
    stage: ModelUsageStage = ModelUsageStage.OTHER
    params: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    first_token_latency_ms: int | None = None
    latency_ms: int = 0
    status: ModelUsageStatus = ModelUsageStatus.SUCCESS
    error: str = ""
    task_id: str | None = None


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    summary: Summary | None = None
    task_state: TaskState = field(default_factory=TaskState)
    tool_state: ToolState = field(default_factory=ToolState)
    model_usage: list[ModelUsage] = field(default_factory=list)


@dataclass
class EvidenceSource:
    kind: SourceKind
    name: str = ""
    uri: str = ""


@dataclass
class EvidenceLinks:
    model_usage_id: str | None = None
    tool_call_id: str | None = None


@dataclass
class Evidence:
    evidence_id: str
    type: EvidenceType
    source: EvidenceSource
    content: str = ""
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)
    links: EvidenceLinks = field(default_factory=EvidenceLinks)


@dataclass
class ContextBlock:
    block_id: str
    block_type: BlockType
    priority: Priority
    token_estimate: int = 0
    content: str = ""
    refs: list[Ref] = field(default_factory=list)


@dataclass
class AgentInfo:
    agent_id: str
    name: str = ""
    version: str = ""


@dataclass
class Actor:
    user_id: str = ""
    user_role: str = ""
    agent: AgentInfo | None = None


@dataclass
class Meta:
    locale: str = ""
    created_at: str = ""
    updated_at: str = ""
    actor: Actor | None = None


# ---------------------------------------------------------------------------
# Top-level document & result types
# ---------------------------------------------------------------------------

@dataclass
class SessionDocument:
    schema_version: str
    session: Session
    evidences: dict[str, Evidence] = field(default_factory=dict)
    context_blocks: list[ContextBlock] = field(default_factory=list)
    meta: Meta | None = None


@dataclass
class PutResult:
    success: bool
    version: int
    error: str | None = None


@dataclass
class RenderedBlock:
    block_id: str
    block_type: BlockType
    priority: Priority
    rendered_content: str
    token_estimate: int


@dataclass
class MessagePart:
    role: Role
    content: str


@dataclass
class AssembledInput:
    parts: list[MessagePart]
    text: str | None = None
    total_tokens: int = 0


@dataclass
class PruneDecision:
    block_id: str
    action: str  # "kept" | "dropped" | "degraded"
    reason: str
    token_estimate: int


@dataclass
class ConflictRecord:
    evidence_ids: list[str]
    field_path: str
    resolution: str
    detail: str = ""


@dataclass
class RedactionRecord:
    evidence_id: str
    rules_applied: list[str]
    fields_redacted: int


@dataclass
class Report:
    turn_id: str
    new_evidence_ids: list[str]
    new_block_ids: list[str]
    prune_decisions: list[PruneDecision]
    conflicts: list[ConflictRecord]
    redactions: list[RedactionRecord]
    token_budget: int
    token_used: int
    degradations: list[str]
    errors: list[str]


@dataclass
class PrepareResult:
    assembled_input: AssembledInput
    report: Report
    session_version: int | None = None
