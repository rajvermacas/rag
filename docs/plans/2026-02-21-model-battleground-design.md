# Chat UX + Model Battleground Design

Date: 2026-02-21
Status: Approved brainstorm design
Scope: Existing FastAPI + Tailwind single-page app

## 1. Locked Decisions

- New chat default assistant message must be exactly: `Hello! How can I assist you today?`
- Add a new UI tab/section: `Model Battleground`.
- Battleground mode is RAG for both model responses.
- Model selection is user-selectable per side (`Model A`, `Model B`).
- Responses stream in parallel side by side.
- End-user judges manually; no winner vote or auto-scoring in v1.
- Battleground model list is from required config allowlist, not hardcoded.

## 2. Goals and Non-Goals

Goals:
- Improve first-run chat UX with a concise default greeting.
- Enable direct side-by-side comparison between two selected LLMs under identical RAG context.
- Preserve fail-fast behavior: no defaults for required config, explicit errors for invalid inputs.

Non-goals:
- Automatic evaluation, ranking, or score aggregation.
- Multi-user battleground history/account persistence.
- Dynamic model discovery from provider APIs in v1.

## 3. UI/UX Design

### 3.1 Navigation

- Add top-level tabs to the existing page shell:
  - `Chat` (existing functionality)
  - `Model Battleground` (new)
- Keep existing upload/documents panel available in the same workspace.

### 3.2 Chat Tab Behavior

- On initial load when no persisted chat state exists, create a new session and append the exact assistant message:
  - `Hello! How can I assist you today?`
- On `New Chat` action, new session should receive the same default assistant message.

### 3.3 Model Battleground Layout

- Prompt input (single message) and submit action.
- Two required dropdowns:
  - `Model A`
  - `Model B`
- Two response panes:
  - left pane for `Model A`
  - right pane for `Model B`
- Render both outputs with the same markdown + sanitizer pipeline used in chat.
- Manual visual comparison only.

### 3.4 Responsiveness

- Desktop: two response panes side by side.
- Mobile: panes stack vertically (`A` then `B`) while keeping parallel streaming behavior.

## 4. Backend Design

### 4.1 New API Endpoints

- `GET /models/battleground`
  - Returns configured allowlist for battleground model dropdowns.
- `POST /battleground/compare/stream`
  - Streams multiplexed events for A/B responses.

### 4.2 Compare Request Contract

Required request fields:
- `message: str`
- `model_a: str`
- `model_b: str`

Validation rules (fail-fast):
- Reject empty `message`.
- Reject missing/empty `model_a` or `model_b`.
- Reject model IDs not present in configured allowlist.
- Reject `model_a == model_b` with explicit 400 error.

### 4.3 Fairness and Data Flow

1. Build retrieval query once from prompt.
2. Retrieve relevant chunks once.
3. Build one shared system/user prompt context.
4. Launch two model chat streams concurrently using identical context and different model IDs.
5. Emit stream events tagged by source (`A` or `B`).

This ensures equal evidence and prompt context for both sides.

### 4.4 Streaming Format

- Use structured newline-delimited JSON events, one object per line:
  - `{ "side": "A", "chunk": "..." }`
  - `{ "side": "B", "chunk": "..." }`
- Optional terminal events can include done/error per side:
  - `{ "side": "A", "done": true }`
  - `{ "side": "B", "error": "..." }`

## 5. Service and Client Changes

### 5.1 Service Layer

- Add `BattlegroundService` to orchestrate:
  - shared retrieval
  - parallel model streaming
  - event multiplexing
- Keep `ChatService` focused on existing chat behavior.

### 5.2 OpenRouter Client

- Add explicit model-override stream method:
  - `stream_chat_response_with_model(model: str, system_prompt: str, user_prompt: str)`
- No fallback model selection when override is absent; raise explicit exception on invalid input.

### 5.3 Frontend Refactor Requirement

- `app/templates/index.html` is near the 800-line limit.
- Split frontend JS into dedicated files to stay under limit and improve SRP:
  - `app/static/js/common.js`
  - `app/static/js/chat.js`
  - `app/static/js/battleground.js`

## 6. Configuration Contract

Add required setting:
- `OPENROUTER_BATTLEGROUND_MODELS`

Requirements:
- Must be present.
- Must parse to a non-empty list.
- No empty entries.
- Startup fails with explicit exception on any invalid condition.

## 7. Logging and Observability

Detailed logs for battleground flow:
- compare request started (model IDs, prompt length)
- retrieval completed (chunk count, timing)
- stream started per side
- stream completed per side (chunk count, timing)
- stream error per side with clear error reason

## 8. Testing Strategy

- UI tests:
  - tab existence and IDs
  - default greeting exact string for new chat
- API tests:
  - `/models/battleground` response and validation behavior
  - compare endpoint request validation failures
- Service tests:
  - single retrieval used for both sides
  - multiplexed stream tagging correctness
  - side-specific error handling behavior
- Integration tests:
  - battleground prompt triggers two side-by-side streamed outputs

## 9. Risks and Mitigations

- Risk: One model stream fails while the other continues.
  - Mitigation: emit side-specific error event and continue best-effort for the other side.
- Risk: UI complexity growth in single template file.
  - Mitigation: split JS into dedicated files now.
- Risk: Cost/latency increase from dual generation.
  - Mitigation: explicit model selection and detailed logging for observability.

## 10. Success Criteria

- New chat always starts with exact greeting text.
- Battleground tab allows selecting two distinct configured models.
- One prompt produces parallel streamed answers in both panes.
- Both answers are generated from identical RAG evidence context.
- Invalid config or invalid request inputs fail fast with explicit errors.
