# Multi-Backend Chat Support Design (OpenAI-Compatible + Azure OpenAI)

Date: 2026-02-22
Status: Approved brainstorm design
Scope: Existing FastAPI + Tailwind single-page app

## 1. Locked Decisions

- Add chat backend support beyond OpenRouter using two provider types in v1:
  - OpenAI-compatible endpoint (`base_url` + `api_key` + model)
  - Azure OpenAI (`endpoint` + `api_key` + `api_version` + deployment mapping)
- Use server-managed backend profiles only. Clients do not submit raw endpoint credentials.
- Keep embeddings fixed globally for now (chat backend selection only).
- Model dropdowns must show provider/source identity, not model name alone.
- No fallback/default behavior for missing/invalid backend configuration; fail fast with explicit exceptions.

## 2. Goals and Non-Goals

Goals:
- Support chat calls to self-hosted/private OpenAI-compatible endpoints (local GPU or private network).
- Add first-class Azure OpenAI support with explicit deployment mapping.
- Preserve strict validation, explicit runtime errors, and detailed logging.
- Eliminate model identity ambiguity by requiring `backend_id` + `model` in requests.

Non-goals:
- Dynamic provider discovery from remote APIs in v1.
- Request-time ad hoc backend credentials.
- Embedding backend multiplexing in v1.

## 3. Architecture

### 3.1 Provider Abstraction

Add a chat provider abstraction with explicit model override methods:
- `generate_chat_response_with_model(model, system_prompt, user_prompt)`
- `stream_chat_response_with_model(model, system_prompt, user_prompt)`

Implementations:
- `OpenAICompatibleChatProvider`
- `AzureOpenAIChatProvider`

### 3.2 Routing Layer

Add a backend router/registry that resolves configured `backend_id` to a concrete provider client instance.

Fail-fast behaviors:
- Unknown backend ID: reject request with explicit 400.
- Unknown model for backend: reject request with explicit 400.
- Missing provider-specific config at startup: raise explicit `ValueError`.

### 3.3 Embeddings Strategy

Retain existing embedding client path and settings as globally fixed for v1 rollout.

## 4. Configuration Contract

Replace chat-only OpenRouter model config with backend profile catalog while preserving existing embedding variables.

### 4.1 Proposed `.env.example`

```dotenv
# Fixed embedding backend (phase 1)
OPENROUTER_API_KEY=
OPENROUTER_EMBED_MODEL=sentence-transformers/all-minilm-l6-v2

# Chat backend catalog
CHAT_BACKEND_IDS=lab_vllm,azure_prod

# Backend: lab_vllm (OpenAI-compatible)
CHAT_BACKEND_LAB_VLLM_PROVIDER=openai_compatible
CHAT_BACKEND_LAB_VLLM_BASE_URL=http://10.0.0.42:8000/v1
CHAT_BACKEND_LAB_VLLM_API_KEY=
CHAT_BACKEND_LAB_VLLM_MODELS=meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3

# Backend: azure_prod (Azure OpenAI)
CHAT_BACKEND_AZURE_PROD_PROVIDER=azure_openai
CHAT_BACKEND_AZURE_PROD_ENDPOINT=https://<resource>.openai.azure.com
CHAT_BACKEND_AZURE_PROD_API_KEY=
CHAT_BACKEND_AZURE_PROD_API_VERSION=2024-10-21
CHAT_BACKEND_AZURE_PROD_MODELS=gpt-4o-mini,gpt-4.1-mini
CHAT_BACKEND_AZURE_PROD_DEPLOYMENTS=gpt-4o-mini:chat-gpt4o-mini,gpt-4.1-mini:chat-gpt41-mini

CHROMA_PERSIST_DIR=/workspaces/rag/chroma
CHROMA_COLLECTION_NAME=rag_docs
MAX_UPLOAD_MB=25
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=6
MIN_RELEVANCE_SCORE=0.3
APP_LOG_LEVEL=INFO
```

### 4.2 Validation Rules

Startup must enforce:
- `CHAT_BACKEND_IDS` exists and is non-empty.
- Every backend ID has a declared provider from supported set.
- `openai_compatible` requires `BASE_URL`, `API_KEY`, and non-empty model list.
- `azure_openai` requires `ENDPOINT`, `API_KEY`, `API_VERSION`, model list, and complete deployment mapping.
- Duplicate model IDs within a backend are rejected.
- Azure deployment mapping must cover every declared Azure model.

## 5. API Contract Changes

### 5.1 Chat Model List Endpoints

Return rich model options containing provider/source identity.

`GET /models/chat` and `GET /models/battleground` response item shape:

```json
{
  "backend_id": "azure_prod",
  "provider": "azure_openai",
  "model": "gpt-4o-mini",
  "label": "azure_prod (azure_openai) · gpt-4o-mini"
}
```

### 5.2 Chat Request Body

`POST /chat` and `POST /chat/stream` must require:
- `message`
- `history`
- `backend_id`
- `model`

### 5.3 Battleground Compare Request Body

`POST /battleground/compare/stream` must require:
- `message`
- `history`
- `model_a_backend_id`
- `model_a`
- `model_b_backend_id`
- `model_b`

Validation:
- each side must use allowed model for its selected backend
- reject identical `(backend_id, model)` pair across sides

## 6. UI/UX Changes

- Chat and battleground dropdowns display provider-aware labels:
  - `azure_prod (azure_openai) · gpt-4o-mini`
  - `lab_vllm (openai_compatible) · meta-llama/Llama-3.1-8B-Instruct`
- UI sends both `backend_id` and `model` in request payload.
- Model selection remains required; empty selection is explicit client-side error.

## 7. Runtime Behavior and Errors

- No implicit provider fallback.
- No implicit model fallback.
- Upstream HTTP failures from provider clients raise explicit runtime errors with status/body context.
- Request validation remains 400 with clear field-specific detail.

## 8. Logging and Observability

Add/retain detailed structured logs for:
- backend resolution (`backend_id`, `provider`, `model`)
- request start/finish for chat and stream routes
- provider outbound call start/finish, status code, stream completion
- error events with backend/model context

## 9. Testing Strategy

- Config tests:
  - backend catalog parsing and validation
  - Azure deployment completeness and duplicates
- Provider tests:
  - OpenAI-compatible request/stream payload shape and error handling
  - Azure request path/query/body construction and stream parsing
- API tests:
  - required `backend_id` and `model`
  - allowlist validation per backend
  - battleground pair validation
- UI tests:
  - provider-aware labels rendered
  - payload includes backend fields for chat and battleground
- Integration tests:
  - end-to-end chat stream using non-OpenRouter OpenAI-compatible backend profile

## 10. Migration and Rollout

- Deprecate `OPENROUTER_CHAT_MODEL` and `OPENROUTER_BATTLEGROUND_MODELS` for chat selection.
- Keep embedding variables unchanged in v1.
- Update `.env.example`, README, and endpoint request/response docs.
- Provide migration notes with exact env var rename/add/remove mapping.

## 11. Success Criteria

- Chat works with private OpenAI-compatible endpoints by selecting backend profile + model.
- Chat works with Azure OpenAI using configured deployment mapping.
- Chat and battleground model selectors clearly show provider/source.
- Invalid config and invalid selection fail fast with explicit errors.
- Existing embedding pipeline remains functional without chat-provider regressions.
