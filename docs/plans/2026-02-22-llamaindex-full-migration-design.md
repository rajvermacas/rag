# LlamaIndex-Centered RAG Migration Design (OpenRouter + OpenAI + Azure OpenAI)

**Date:** 2026-02-22  
**Status:** Approved

## Objective
Migrate the existing native RAG implementation to a LlamaIndex-centered architecture using battle-tested libraries while preserving all current UI workflows.

## Required Outcomes
- Keep OpenRouter support (required) for embeddings and chat.
- Use OpenAI and Azure OpenAI libraries through LlamaIndex-backed integrations.
- Breaking internal/backend changes are acceptable if they simplify migration.
- Preserve end-to-end UI workflows currently in place (`upload`, `chat`, `stream`, `battleground`, model listing).
- Keep strict fail-fast behavior for configuration and runtime validation.

## Key Decisions
1. Use a **full LlamaIndex-centered architecture** for ingestion, indexing, retrieval, and response generation.
2. Retain **OpenRouter** for both embeddings and chat via OpenAI-compatible integration.
3. Support three backend provider types:
- `openrouter`
- `openai`
- `azure_openai`
4. Keep route-level UX behavior and workflow parity; internal service contracts may change.

## Target Architecture
1. **LLM Provider Layer**
- Provider registry maps `backend_id -> configured LlamaIndex LLM`.
- OpenRouter configured as OpenAI-compatible base URL (`https://openrouter.ai/api/v1`).
- OpenAI configured via official OpenAI integration.
- Azure OpenAI configured via official Azure integration (endpoint, deployment/api version settings).

2. **Indexing Layer**
- Use LlamaIndex readers for supported file types (`pdf`, `docx`, `txt`).
- Use LlamaIndex text splitting and ingestion pipeline.
- Store vectors in Chroma through LlamaIndex Chroma adapter.

3. **Query Layer**
- Build per-request query execution from selected backend/model.
- Retrieval and synthesis handled through LlamaIndex retriever/query engine.
- Maintain current document-first assistant behavior via centralized prompt templates and light response post-processing.

4. **API/UI Layer**
- FastAPI routes remain mapped to current UI workflows.
- Response shapes/events remain compatible with existing frontend behavior for chat and battleground streaming.

## Component Plan
### `app/config.py`
- Continue strict env parsing with explicit errors for missing values.
- Extend backend profile schema for provider types: `openrouter`, `openai`, `azure_openai`.
- Keep per-backend model allowlists.

### `app/services/llm_registry.py` (new)
- Build and expose provider/model-aware LlamaIndex LLM instances.
- Validate provider-specific required fields and raise explicit exceptions.

### `app/services/indexing.py` (new)
- Replace native parser/chunk/embed flow with LlamaIndex ingestion pipeline.
- Persist to existing Chroma directory/collection using LlamaIndex adapter.
- Maintain metadata needed by document list/delete UI workflows.

### `app/services/query_engine.py` (new)
- Encapsulate retrieval + generation per backend/model.
- Support non-streaming and streaming response generation.
- Enforce backend/model allowlist validation.

### `app/main.py`
- Keep endpoints used by UI.
- Replace native service wiring with new LlamaIndex-based services.

## Workflow Data Flow
### Upload
1. Validate file metadata and size.
2. Parse with LlamaIndex readers.
3. Split/chunk through LlamaIndex transforms.
4. Embed via OpenRouter integration.
5. Persist vectors + metadata to Chroma.
6. Return indexed document metadata to UI.

### Chat
1. Validate request payload and non-empty fields.
2. Resolve `backend_id` and enforce allowed `model`.
3. Execute retrieval + synthesis through LlamaIndex query flow.
4. Return answer payload compatible with current UI handling.

### Streaming Chat
1. Resolve provider/model as in chat.
2. Use LlamaIndex streaming response path.
3. Emit SSE in format expected by existing frontend handlers.

### Battleground
1. Resolve and validate two backend/model selections.
2. Run two query streams in parallel.
3. Emit side-tagged streaming events for both panels.

## Error Handling and Observability
- Keep fail-fast startup config validation.
- Normalize provider and LlamaIndex exceptions to clear API-facing errors.
- Keep structured logs for:
  - backend/provider/model selection
  - retrieval counts and thresholds
  - streaming start/completion
  - ingestion/indexing lifecycle
- Reject unknown backend IDs and disallowed models at server boundary.

## Constraints
- No silent defaults or fallback behavior.
- Preserve detailed logging.
- Keep files under 800 lines and functions under 80 lines by splitting modules.

## Test Strategy
1. Replace native provider transport tests with LlamaIndex provider integration contract tests.
2. Keep and adapt API tests for:
- `/upload`
- `/documents`
- `/chat`
- `/chat/stream`
- `/models/chat`
- `/models/battleground`
- `/battleground/compare/stream`
3. Add startup smoke tests for each configured provider type.
4. Add streaming behavior tests for chat and battleground parity.

## Rollout Plan
1. Introduce new LlamaIndex service modules alongside current routes.
2. Switch chat and battleground internals to new services.
3. Switch ingestion and retrieval internals to new pipeline.
4. Remove deprecated native modules (manual HTTP clients, manual retrieval/chunking/parsing where redundant).
5. Update `.env.example` and `README.md` with new provider schema and examples.

## Success Criteria
- All UI workflows operate end-to-end as they do today.
- OpenRouter, OpenAI, and Azure OpenAI are supported in provider selection.
- No native raw HTTP LLM provider wrappers remain in active code paths.
- Strict validation and detailed logging remain in place.
- Test suite passes with migrated architecture.
