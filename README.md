# Context Engineering SDK

Python SDK for structured context modeling, evidence management, and context assembly for AI agents.

The SDK provides a reusable, observable, and governable context building pipeline based on the structured context engineering model (`context_engineering_model.md`), enabling:

- **Standardized data structures**: Typed `session`, `evidences`, and `context_blocks` with validation
- **Context assembly pipeline**: Ingest evidence -> derive blocks -> prune by budget -> render -> assemble final model input
- **Evidence traceability**: Every output can trace back to `evidence_id` via `refs`, linking to tool calls and model usage
- **Governance & observability**: Token costs, pruning decisions, conflict detection, and all key actions emit structured events
- **Extensibility**: Storage, tokenizer, pruning strategy, rendering, and more are pluggable via protocols

## Quick Start

```python
import asyncio
from context_engineering_sdk import (
    create_context_engine, RuntimeConfig, Message, Role,
)
from context_engineering_sdk.store import MemoryStore
from context_engineering_sdk.core.token_estimator import CharBasedEstimator
from context_engineering_sdk.builder.summarizer import LlmRequest, LlmResponse

# Provide your own LLM adapter
class MyLlmAdapter:
    async def generate(self, request: LlmRequest) -> LlmResponse:
        # Call your LLM here
        return LlmResponse(content="summary", model="my-model")

async def main():
    engine = create_context_engine(
        store=MemoryStore(),
        token_estimator=CharBasedEstimator(),
        llm_adapter=MyLlmAdapter(),
    )

    # 1. Prepare turn: builds assembled input from user message + context
    result = await engine.prepare_turn(
        session_id="session-1",
        user_message=Message(role=Role.USER, content="What is context engineering?"),
        runtime_config=RuntimeConfig(),
    )

    # 2. Use assembled_input to call your LLM
    print(f"Total tokens: {result.assembled_input.total_tokens}")
    print(f"Parts: {len(result.assembled_input.parts)}")

    # 3. Commit assistant response
    await engine.commit_assistant_message(
        "session-1",
        Message(role=Role.ASSISTANT, content="Context engineering is..."),
    )

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  ContextEngine (orchestration entry point)          │
│  prepare_turn -> commit -> record                   │
├─────────────────────────────────────────────────────┤
│  Pipeline:                                          │
│  Ingest -> Derive -> Prune -> Render -> Assemble    │
├──────────────┬──────────────┬───────────────────────┤
│  Store       │  Policies    │  Observability        │
│  (Memory/    │  (Budget/    │  (EventBus/           │
│   File)      │   Priority/  │   Events/             │
│              │   TTL)       │   Replay)             │
├──────────────┴──────────────┴───────────────────────┤
│  Core: types / errors / id / clock / hash / redact  │
│        ref_selector / token_estimator / migrator     │
└─────────────────────────────────────────────────────┘
```

## Package Structure

```
context_engineering_sdk/
├── __init__.py                 # Public API exports
├── config.py                   # RuntimeConfig and sub-configs
├── engine.py                   # DefaultContextEngine + create_context_engine
├── core/
│   ├── types.py                # All enums and dataclasses (schema-aligned)
│   ├── errors.py               # Exception hierarchy
│   ├── id_generator.py         # IdGenerator protocol + UuidV4Generator
│   ├── clock.py                # Clock protocol + SystemClock
│   ├── hasher.py               # Hasher protocol + Sha256Hasher
│   ├── redactor.py             # Redactor protocol + RegexRedactor
│   ├── ref_selector.py         # RefSelector (lines/chars/json/regex)
│   ├── token_estimator.py      # TokenEstimator protocol + CharBasedEstimator
│   └── migrator.py             # Schema version migration
├── store/
│   ├── base.py                 # Store protocol + filter types
│   ├── memory.py               # MemoryStore (in-memory, for testing)
│   └── file.py                 # FileStore (local JSON files)
├── builder/
│   ├── ingestor.py             # EvidenceIngestor (dedup + redaction)
│   ├── deriver.py              # BlockDeriver (conversation/state/evidence)
│   ├── pruner.py               # Pruner + GreedyPriorityPruner
│   ├── renderer.py             # Renderer + EvidenceResolver
│   ├── assembler.py            # Assembler (messages + blocks -> input)
│   └── summarizer.py           # Summarizer + LlmAdapter protocol
├── observability/
│   └── event_bus.py            # EventBus protocol + InMemoryEventBus
└── integrations/               # Framework adapters (future)
```

## Key Concepts

### Two-Phase Turn API

The SDK uses a two-phase approach for each conversation turn:

1. **Phase A: `prepare_turn()`** - Loads session, appends user message, optionally summarizes, derives context blocks, prunes within budget, renders refs, and assembles final input.
2. **Phase B: `commit_assistant_message()` / `finalize_assistant_message()`** - Records the assistant's response back to the session store.

### Streaming Support

```python
# Streaming workflow
result = await engine.prepare_turn(session_id, user_msg, config)

# Feed assembled_input to your LLM
async for chunk in llm.stream(result.assembled_input):
    await engine.commit_assistant_chunk(session_id, chunk.text, chunk.index)

# Finalize when stream ends
await engine.finalize_assistant_message(session_id, refs=[])
```

### Evidence Ingestion

All external information (tool results, RAG docs, LLM outputs) flows through evidence:

```python
from context_engineering_sdk.builder.ingestor import DefaultEvidenceIngestor

ingestor = DefaultEvidenceIngestor(store=store)
evidence, redaction = await ingestor.ingest(
    session_id="s1",
    content='{"order_id": "123", "status": "shipped"}',
    source=EvidenceSource(kind=SourceKind.TOOL, name="getOrder"),
    evidence_type=EvidenceType.TOOL_RESULT,
    links=EvidenceLinks(tool_call_id="tc1"),
)
```

### Pruning Strategy

The `GreedyPriorityPruner` ensures:
- `must`-priority blocks are never dropped
- Remaining blocks are sorted by priority and recency
- Total token estimate stays within budget
- Every block gets a `PruneDecision` explaining why it was kept or dropped

### Pluggable Components

All components follow Protocol-based interfaces and can be replaced:

| Component | Protocol | Default Implementation |
|-----------|----------|----------------------|
| Store | `Store` | `MemoryStore`, `FileStore` |
| Token Estimator | `TokenEstimator` | `CharBasedEstimator` |
| Pruner | `Pruner` | `GreedyPriorityPruner` |
| Redactor | `Redactor` | `RegexRedactor` |
| EventBus | `EventBus` | `InMemoryEventBus` |
| LLM Adapter | `LlmAdapter` | (user-provided) |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT
