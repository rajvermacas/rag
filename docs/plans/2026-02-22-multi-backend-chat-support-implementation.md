# Multi-Backend Chat Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add server-managed chat backend profiles that support both arbitrary OpenAI-compatible endpoints and Azure OpenAI, while keeping embeddings fixed to current OpenRouter settings.

**Architecture:** Replace single-provider chat model settings with a validated backend catalog (`backend_id -> provider config + allowed models`). Introduce provider-specific chat clients behind a shared router so chat and battleground flows accept explicit `backend_id` + `model` pairs. Update API and UI contracts to expose provider-aware model options and send backend identity in every chat/battleground request.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, httpx async streaming, vanilla JavaScript, pytest.

---

### Task 1: Introduce chat backend profile config schema and validation

**Files:**
- Modify: `app/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`
- Modify: `.env.example`

**Step 1: Write failing config tests first**
Add test cases for:
- missing `CHAT_BACKEND_IDS`
- unknown provider value for a backend
- missing required provider-specific fields
- duplicate models in a backend
- missing Azure deployment mapping entries
- successful parse of two backends (`openai_compatible` and `azure_openai`)

Example target assertions:
```python
settings = Settings.from_env()
assert settings.chat_backend_profiles["lab_vllm"].provider == "openai_compatible"
assert settings.chat_backend_profiles["azure_prod"].azure_deployments["gpt-4o-mini"] == "chat-gpt4o-mini"
```

**Step 2: Run targeted tests to verify failure**
Run: `pytest tests/test_config.py -q`
Expected: FAIL because new backend config contract is not implemented.

**Step 3: Implement minimal config model and parser**
Add typed immutable config structures and strict parsing helpers, for example:
```python
@dataclass(frozen=True)
class ChatBackendProfile:
    backend_id: str
    provider: str
    models: tuple[str, ...]
    base_url: str | None
    api_key: str
    azure_endpoint: str | None
    azure_api_version: str | None
    azure_deployments: Mapping[str, str]
```

Implementation requirements:
- parse `CHAT_BACKEND_IDS` as required CSV with non-empty unique IDs
- resolve each backend namespace from uppercased backend ID token
- enforce provider-specific required env vars
- enforce model list non-empty, trimmed, unique
- enforce Azure deployment map coverage for all listed models
- preserve existing fail-fast explicit exception style
- keep embedding env vars untouched in this task

**Step 4: Update shared pytest env fixture**
In `tests/conftest.py`, replace deprecated chat env vars with new backend catalog vars used by the test suite.

**Step 5: Re-run targeted tests**
Run: `pytest tests/test_config.py -q`
Expected: PASS.

**Step 6: Commit**
```bash
git add app/config.py tests/conftest.py tests/test_config.py .env.example
git commit -m "feat(config): add multi-backend chat profile parsing"
```

### Task 2: Add provider clients and backend router for chat generation

**Files:**
- Create: `app/services/chat_provider_models.py`
- Create: `app/services/chat_provider_router.py`
- Create: `app/services/openai_compatible_chat_provider.py`
- Create: `app/services/azure_openai_chat_provider.py`
- Modify: `app/services/__init__.py`
- Create: `tests/services/test_chat_provider_router.py`
- Create: `tests/services/test_openai_compatible_chat_provider.py`
- Create: `tests/services/test_azure_openai_chat_provider.py`

**Step 1: Write failing provider tests**
Add tests for:
- router rejects unknown backend ID
- router rejects model not in backend allowlist
- OpenAI-compatible provider sends expected payload to `/chat/completions`
- Azure provider targets deployment URL and includes `api-version`
- stream parsing yields text chunks and handles non-200 responses

Example router assertion:
```python
with pytest.raises(ValueError, match="backend_id is not allowed"):
    await router.generate_chat_response_with_backend(
        backend_id="missing",
        model="gpt-4o-mini",
        system_prompt="s",
        user_prompt="u",
    )
```

**Step 2: Run targeted tests to verify failure**
Run: `pytest tests/services/test_chat_provider_router.py tests/services/test_openai_compatible_chat_provider.py tests/services/test_azure_openai_chat_provider.py -q`
Expected: FAIL because provider modules do not exist.

**Step 3: Implement provider interfaces and router**
Create a shared protocol and model option DTO:
```python
class BackendChatProvider(Protocol):
    async def generate_chat_response_with_model(self, model: str, system_prompt: str, user_prompt: str) -> str: ...
    async def stream_chat_response_with_model(self, model: str, system_prompt: str, user_prompt: str) -> AsyncIterator[str]: ...
```

Router behavior:
- accept `backend_id`, `model`, prompts
- resolve backend profile
- validate model membership
- delegate to selected provider instance
- log backend/provider/model per request

Provider behavior:
- OpenAI-compatible: `POST {base_url}/chat/completions`
- Azure OpenAI: `POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}`
- strict prompt validation and explicit runtime errors on non-200
- streaming SSE line handling aligned with current style

**Step 4: Re-run targeted provider tests**
Run: `pytest tests/services/test_chat_provider_router.py tests/services/test_openai_compatible_chat_provider.py tests/services/test_azure_openai_chat_provider.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add app/services/chat_provider_models.py app/services/chat_provider_router.py app/services/openai_compatible_chat_provider.py app/services/azure_openai_chat_provider.py app/services/__init__.py tests/services/test_chat_provider_router.py tests/services/test_openai_compatible_chat_provider.py tests/services/test_azure_openai_chat_provider.py
git commit -m "feat(chat): add provider router for openai-compatible and azure backends"
```

### Task 3: Wire app service construction to backend router while keeping embeddings fixed

**Files:**
- Modify: `app/main.py`
- Modify: `app/services/chat.py`
- Modify: `app/services/battleground.py`
- Modify: `tests/services/test_chat.py`
- Modify: `tests/services/test_battleground.py`

**Step 1: Write failing service tests**
Update service-level tests to require backend-aware chat calls:
- chat service methods now receive `backend_id` and `model`
- battleground compare receives backend/model per side
- fakes assert correct backend+model forwarding

**Step 2: Run targeted tests to verify failure**
Run: `pytest tests/services/test_chat.py tests/services/test_battleground.py -q`
Expected: FAIL due to signature mismatch and missing backend forwarding.

**Step 3: Implement minimal wiring changes**
- In `_build_services`, construct provider instances from parsed backend profiles and inject `ChatProviderRouter` into chat/battleground paths.
- Keep embedding clients on existing OpenRouter path for retrieval/ingest.
- Update `ChatService` and `BattlegroundService` method signatures and logging fields:
  - `backend_id`
  - `provider`
  - `model`

Example signature target:
```python
async def answer_question(self, question: str, history: list[ConversationTurn], backend_id: str, model: str) -> ChatResult:
    ...
```

**Step 4: Re-run targeted service tests**
Run: `pytest tests/services/test_chat.py tests/services/test_battleground.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add app/main.py app/services/chat.py app/services/battleground.py tests/services/test_chat.py tests/services/test_battleground.py
git commit -m "refactor(chat): thread backend_id through chat and battleground services"
```

### Task 4: Update API contracts to use provider-aware model options

**Files:**
- Modify: `app/main.py`
- Modify: `tests/api/test_chat_api.py`
- Modify: `tests/api/test_battleground_api.py`

**Step 1: Write failing API tests**
Update API tests for new contracts:
- `GET /models/chat` returns list of objects: `backend_id`, `provider`, `model`, `label`
- `GET /models/battleground` same shape
- `/chat` and `/chat/stream` require `backend_id` + `model`
- `/battleground/compare/stream` requires `model_a_backend_id` + `model_a` + `model_b_backend_id` + `model_b`
- reject same backend+model pair across sides

Example expected response item:
```python
{
    "backend_id": "azure_prod",
    "provider": "azure_openai",
    "model": "gpt-4o-mini",
    "label": "azure_prod (azure_openai) · gpt-4o-mini",
}
```

**Step 2: Run targeted API tests to verify failure**
Run: `pytest tests/api/test_chat_api.py tests/api/test_battleground_api.py -q`
Expected: FAIL because current payload/response schemas are string-only model IDs.

**Step 3: Implement minimal API schema and validation updates**
- Add Pydantic request fields for backend IDs.
- Add shared helper to validate `(backend_id, model)` selection.
- Replace model list endpoints with rich option objects.
- Preserve explicit 400 detail messages for invalid selections.

**Step 4: Re-run targeted API tests**
Run: `pytest tests/api/test_chat_api.py tests/api/test_battleground_api.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add app/main.py tests/api/test_chat_api.py tests/api/test_battleground_api.py
git commit -m "feat(api): require backend_id and expose provider-aware model options"
```

### Task 5: Update frontend selectors and request payloads with backend identity

**Files:**
- Modify: `app/static/js/chat.js`
- Modify: `app/static/js/battleground.js`
- Modify: `tests/ui/test_index_page.py`
- Modify: `tests/ui/test_battleground_page.py`
- Modify: `tests/ui/_battleground_harnesses.py`

**Step 1: Write failing UI tests**
Add/update tests for:
- dropdowns render provider-aware labels
- option values carry backend+model identity (encoded object key or structured data)
- chat stream request body includes `backend_id` and `model`
- battleground compare request body includes side-specific backend IDs and models

**Step 2: Run targeted UI tests to verify failure**
Run: `pytest tests/ui/test_index_page.py tests/ui/test_battleground_page.py -q`
Expected: FAIL because current JS sends only model IDs.

**Step 3: Implement minimal UI payload wiring**
- Parse model-option objects from `/models/chat` and `/models/battleground`.
- Render `label` text exactly as returned by API.
- Maintain strict non-empty validation for backend/model selection.
- Send new request fields:
  - chat: `backend_id`, `model`
  - battleground: `model_a_backend_id`, `model_a`, `model_b_backend_id`, `model_b`

**Step 4: Re-run targeted UI tests**
Run: `pytest tests/ui/test_index_page.py tests/ui/test_battleground_page.py -q`
Expected: PASS.

**Step 5: Commit**
```bash
git add app/static/js/chat.js app/static/js/battleground.js tests/ui/test_index_page.py tests/ui/test_battleground_page.py tests/ui/_battleground_harnesses.py
git commit -m "feat(ui): show provider labels and send backend-aware model selections"
```

### Task 6: Update integration tests and documentation, then verify full suite

**Files:**
- Modify: `tests/integration/test_upload_chat_flow.py`
- Modify: `README.md`
- Modify: `.env.example`
- Verify: `app/main.py`
- Verify: `app/config.py`

**Step 1: Write failing integration/doc expectation updates**
- Update integration payloads to include `backend_id` where chat stream is called.
- Update README environment/API sections to new backend catalog contract.

**Step 2: Run targeted integration test to verify failure**
Run: `pytest tests/integration/test_upload_chat_flow.py -q`
Expected: FAIL before payload updates are complete.

**Step 3: Implement minimal docs/integration updates**
- Ensure README examples and endpoint contracts match implemented schemas.
- Ensure `.env.example` exactly matches required env vars.

**Step 4: Run focused verification suite**
Run: `pytest tests/test_config.py tests/services/test_chat_provider_router.py tests/services/test_openai_compatible_chat_provider.py tests/services/test_azure_openai_chat_provider.py tests/services/test_chat.py tests/services/test_battleground.py tests/api/test_chat_api.py tests/api/test_battleground_api.py tests/ui/test_index_page.py tests/ui/test_battleground_page.py tests/integration/test_upload_chat_flow.py -q`
Expected: PASS.

**Step 5: Run full suite verification**
Run: `pytest -q`
Expected: PASS.

**Step 6: Commit**
```bash
git add tests/integration/test_upload_chat_flow.py README.md .env.example
git commit -m "docs: document multi-backend chat configuration and api contracts"
```

### Task 7: Final quality gate and branch readiness

**Files:**
- Verify: repository working tree and tests

**Step 1: Re-run final smoke commands**
Run: `pytest -q`
Expected: PASS.

Run: `git status --short`
Expected: clean working tree.

**Step 2: If any unexpected failure occurs, pause and apply `@superpowers:systematic-debugging` before code changes.**

**Step 3: Summarize migration impact**
Prepare a short release note:
- removed: `OPENROUTER_CHAT_MODEL`, `OPENROUTER_BATTLEGROUND_MODELS`
- added: `CHAT_BACKEND_IDS` and per-backend namespaced vars
- unchanged: embedding vars remain OpenRouter-backed
