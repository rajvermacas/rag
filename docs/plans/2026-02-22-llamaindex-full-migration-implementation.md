# LlamaIndex-Centered RAG Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace native ingestion/retrieval/provider plumbing with a LlamaIndex-centered architecture supporting OpenRouter, OpenAI, and Azure OpenAI while preserving all current UI workflows.

**Architecture:** Keep FastAPI routes and UI-facing workflows stable, but migrate internals to a provider registry + LlamaIndex ingestion/query pipeline backed by Chroma. Enforce strict config validation and server-side backend/model allowlists with explicit errors.

**Tech Stack:** Python 3.11, FastAPI, LlamaIndex, OpenAI SDK, Azure OpenAI SDK, ChromaDB, pytest.

---

### Task 1: Add Dependencies and Provider Types in Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing tests**

```python
def test_settings_accepts_openrouter_openai_and_azure_providers(monkeypatch):
    monkeypatch.setenv("CHAT_BACKEND_IDS", "openrouter_lab,openai_prod,azure_prod")
    monkeypatch.setenv("CHAT_BACKEND_OPENROUTER_LAB_PROVIDER", "openrouter")
    monkeypatch.setenv("CHAT_BACKEND_OPENAI_PROD_PROVIDER", "openai")
    monkeypatch.setenv("CHAT_BACKEND_AZURE_PROD_PROVIDER", "azure_openai")
    settings = Settings.from_env()
    assert settings.chat_backend_profiles["openrouter_lab"].provider == "openrouter"
    assert settings.chat_backend_profiles["openai_prod"].provider == "openai"
    assert settings.chat_backend_profiles["azure_prod"].provider == "azure_openai"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_settings_accepts_openrouter_openai_and_azure_providers -v`  
Expected: FAIL with provider validation error (unsupported provider values).

**Step 3: Write minimal implementation**

```python
_PROVIDER_OPENROUTER = "openrouter"
_PROVIDER_OPENAI = "openai"
_PROVIDER_AZURE_OPENAI = "azure_openai"
_ALLOWED_PROVIDERS = (_PROVIDER_OPENROUTER, _PROVIDER_OPENAI, _PROVIDER_AZURE_OPENAI)
```

- Update profile parsing so provider-required env vars are validated explicitly per provider.
- Keep no-default/fail-fast behavior.

**Step 4: Add migration dependencies**

- Add LlamaIndex and provider integration packages to `pyproject.toml`.
- Keep explicit pinned versions.

**Step 5: Run tests to verify pass**

Run: `pytest tests/test_config.py -v`  
Expected: PASS.

**Step 6: Commit**

```bash
git add pyproject.toml app/config.py tests/test_config.py
git commit -m "feat(config): add openrouter/openai/azure provider types for llamaindex"
```

### Task 2: Build LLM Registry Service

**Files:**
- Create: `app/services/llm_registry.py`
- Create: `tests/services/test_llm_registry.py`
- Modify: `app/services/__init__.py`

**Step 1: Write the failing tests**

```python
def test_registry_builds_openrouter_llm_profile():
    registry = LLMRegistry(settings_with_openrouter_profile())
    llm = registry.get_llm("openrouter_lab", "openai/gpt-4o-mini")
    assert llm is not None


def test_registry_rejects_unknown_backend_id():
    registry = LLMRegistry(settings_with_openrouter_profile())
    with pytest.raises(ValueError, match="backend_id is not allowed"):
        registry.get_llm("missing", "openai/gpt-4o-mini")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_llm_registry.py -v`  
Expected: FAIL with `ModuleNotFoundError` or missing `LLMRegistry`.

**Step 3: Write minimal implementation**

```python
class LLMRegistry:
    def __init__(self, backend_profiles: Mapping[str, ChatBackendProfile]) -> None:
        if len(backend_profiles) == 0:
            raise ValueError("backend_profiles must not be empty")
        self._profiles = MappingProxyType(dict(backend_profiles))

    def get_llm(self, backend_id: str, model: str):
        profile = self._resolve_profile(backend_id)
        self._validate_model(profile, model)
        return _build_llamaindex_llm(profile, model)
```

- Implement one builder per provider (`openrouter`, `openai`, `azure_openai`).
- For OpenRouter use explicit OpenAI-compatible API base.
- Raise explicit exceptions for missing endpoint/api version/deployment mapping.

**Step 4: Run tests to verify pass**

Run: `pytest tests/services/test_llm_registry.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/llm_registry.py app/services/__init__.py tests/services/test_llm_registry.py
git commit -m "feat(services): add llamaindex llm registry for multi-provider backends"
```

### Task 3: Build LlamaIndex Ingestion Service

**Files:**
- Create: `app/services/indexing.py`
- Create: `tests/services/test_indexing.py`
- Modify: `app/services/documents.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_ingest_upload_indexes_document_and_returns_counts(tmp_path):
    service = build_indexing_service(tmp_path)
    upload = fake_upload("notes.txt", "text/plain", b"hello world")
    result = await service.ingest_upload(upload)
    assert result.doc_id != ""
    assert result.chunks_indexed > 0


def test_delete_document_raises_when_doc_not_found(tmp_path):
    service = build_indexing_service(tmp_path)
    with pytest.raises(ValueError, match="document not found"):
        service.delete_document("missing")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_indexing.py -v`  
Expected: FAIL because `IndexingService` does not exist.

**Step 3: Write minimal implementation**

```python
class IndexingService:
    async def ingest_upload(self, upload: UploadLike) -> IngestResult:
        document = self._load_document(upload)
        nodes = await self._run_ingestion_pipeline(document)
        doc_id = self._extract_doc_id(nodes)
        return IngestResult(doc_id=doc_id, chunks_indexed=len(nodes))
```

- Use LlamaIndex readers/splitter + Chroma vector-store integration.
- Preserve strict upload validation (filename/content_type/non-empty/size).
- Keep document listing/deletion semantics used by UI.

**Step 4: Run tests to verify pass**

Run: `pytest tests/services/test_indexing.py tests/services/test_documents.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/indexing.py app/services/documents.py tests/services/test_indexing.py
git commit -m "feat(ingest): migrate upload indexing flow to llamaindex pipeline"
```

### Task 4: Build Query Service for Non-Streaming Chat

**Files:**
- Create: `app/services/query_engine.py`
- Create: `tests/services/test_query_engine.py`
- Modify: `app/services/chat.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_answer_question_uses_backend_model_and_returns_grounded_flag():
    service = build_query_engine_service()
    result = await service.answer_question(
        question="What is in my docs?",
        history=[],
        backend_id="openrouter_lab",
        model="openai/gpt-4o-mini",
    )
    assert isinstance(result.answer, str)
    assert isinstance(result.grounded, bool)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_query_engine.py::test_answer_question_uses_backend_model_and_returns_grounded_flag -v`  
Expected: FAIL with missing service or wiring.

**Step 3: Write minimal implementation**

```python
class QueryEngineService:
    async def answer_question(self, question: str, history: list[ConversationTurn], backend_id: str, model: str) -> ChatResult:
        self._validate_question_and_history(question, history)
        llm = self._llm_registry.get_llm(backend_id, model)
        engine = self._engine_factory.build_query_engine(llm)
        response = await engine.aquery(self._build_query(question, history))
        return self._to_chat_result(response)
```

- Keep explicit validation with clear errors.
- Preserve existing no-inline-citation output behavior.

**Step 4: Run tests to verify pass**

Run: `pytest tests/services/test_query_engine.py tests/services/test_chat.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/query_engine.py app/services/chat.py tests/services/test_query_engine.py
git commit -m "feat(chat): add llamaindex-backed non-streaming query engine"
```

### Task 5: Add Streaming Query Path for Chat

**Files:**
- Modify: `app/services/query_engine.py`
- Modify: `app/services/chat.py`
- Modify: `tests/services/test_query_engine.py`
- Modify: `tests/api/test_chat_api.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_stream_answer_question_yields_token_chunks_in_order():
    service = build_query_engine_service_with_stream(["Hello ", "world"])
    chunks = [chunk async for chunk in service.stream_answer_question("hi", [], "openrouter_lab", "openai/gpt-4o-mini")]
    assert chunks == ["Hello ", "world"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_query_engine.py::test_stream_answer_question_yields_token_chunks_in_order -v`  
Expected: FAIL because stream method not implemented.

**Step 3: Write minimal implementation**

```python
async def stream_answer_question(self, question: str, history: list[ConversationTurn], backend_id: str, model: str) -> AsyncIterator[str]:
    llm = self._llm_registry.get_llm(backend_id, model)
    engine = self._engine_factory.build_streaming_query_engine(llm)
    response = await engine.aquery(self._build_query(question, history))
    async for chunk in response.async_response_gen():
        if chunk != "":
            yield chunk
```

- Keep explicit request validation.
- Keep SSE event shape used by frontend.

**Step 4: Run tests to verify pass**

Run: `pytest tests/services/test_query_engine.py tests/api/test_chat_api.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/query_engine.py app/services/chat.py tests/services/test_query_engine.py tests/api/test_chat_api.py
git commit -m "feat(chat): add llamaindex streaming query path"
```

### Task 6: Migrate Battleground Compare Flow

**Files:**
- Modify: `app/services/battleground.py`
- Modify: `tests/services/test_battleground.py`
- Modify: `tests/api/test_battleground_api.py`

**Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_compare_stream_emits_left_and_right_chunks():
    service = build_battleground_service()
    events = [event async for event in service.compare_stream("question", [], "openrouter_lab", "openai/gpt-4o-mini", "azure_prod", "gpt-4o-mini")]
    assert any(event.side == "a" for event in events)
    assert any(event.side == "b" for event in events)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_battleground.py::test_compare_stream_emits_left_and_right_chunks -v`  
Expected: FAIL with old dependencies or mismatched service wiring.

**Step 3: Write minimal implementation**

```python
async def compare_stream(...):
    stream_a = self._query_service.stream_answer_question(question, history, model_a_backend_id, model_a)
    stream_b = self._query_service.stream_answer_question(question, history, model_b_backend_id, model_b)
    async for event in merge_streams(stream_a, stream_b):
        yield event
```

- Keep existing event typing and frontend-compatible tags.
- Validate both backend/model pairs independently.

**Step 4: Run tests to verify pass**

Run: `pytest tests/services/test_battleground.py tests/api/test_battleground_api.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/battleground.py tests/services/test_battleground.py tests/api/test_battleground_api.py
git commit -m "refactor(battleground): route compare streams through llamaindex query service"
```

### Task 7: Wire App Composition to New Services

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_health_smoke.py`
- Modify: `tests/integration/test_upload_chat_flow.py`

**Step 1: Write the failing integration test**

```python
@pytest.mark.asyncio
async def test_upload_then_chat_then_battleground_flow(client):
    # upload -> /chat -> /battleground/compare/stream should all succeed
    ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_upload_chat_flow.py -v`  
Expected: FAIL from startup wiring mismatch.

**Step 3: Write minimal implementation**

```python
def _build_services(settings: Settings) -> AppServices:
    vector_runtime = build_vector_runtime(settings)
    llm_registry = LLMRegistry(settings.chat_backend_profiles)
    indexing_service = IndexingService(vector_runtime=vector_runtime, ...)
    query_service = QueryEngineService(vector_runtime=vector_runtime, llm_registry=llm_registry, ...)
    ...
```

- Remove active wiring to legacy provider HTTP wrappers.
- Preserve route interfaces used by frontend scripts.

**Step 4: Run tests to verify pass**

Run: `pytest tests/integration/test_upload_chat_flow.py tests/test_health_smoke.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/main.py tests/integration/test_upload_chat_flow.py tests/test_health_smoke.py
git commit -m "refactor(app): wire fastapi services to llamaindex runtime"
```

### Task 8: Remove Deprecated Native Provider/Embedding Paths

**Files:**
- Delete: `app/services/openrouter_client.py`
- Delete: `app/services/openai_compatible_chat_provider.py`
- Delete: `app/services/azure_openai_chat_provider.py`
- Modify: `tests/services/test_openrouter_client.py`
- Modify: `tests/services/test_openai_compatible_chat_provider.py`
- Modify: `tests/services/test_azure_openai_chat_provider.py`

**Step 1: Write replacement tests first**

```python
def test_llm_registry_openrouter_configuration_uses_openai_compatible_base_url():
    registry = LLMRegistry(settings_with_openrouter_profile())
    llm = registry.get_llm("openrouter_lab", "openai/gpt-4o-mini")
    assert llm is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_llm_registry.py -v`  
Expected: FAIL for missing replacement assertions.

**Step 3: Remove deprecated modules and migrate tests**

- Delete old native transport test files.
- Move assertions into `tests/services/test_llm_registry.py` and `tests/services/test_query_engine.py`.

**Step 4: Run tests to verify pass**

Run: `pytest tests/services -v`  
Expected: PASS with no references to deprecated modules.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove native provider clients after llamaindex migration"
```

### Task 9: Update Documentation and Environment Contract

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/plans/2026-02-22-llamaindex-full-migration-design.md`

**Step 1: Write doc coverage tests (if present) or assertions in existing tests**

```python
def test_env_example_contains_required_provider_fields():
    content = Path(".env.example").read_text()
    assert "CHAT_BACKEND_" in content
    assert "openrouter" in content
    assert "openai" in content
    assert "azure_openai" in content
```

**Step 2: Run test to verify it fails (if new)**

Run: `pytest tests/test_config.py::test_env_example_contains_required_provider_fields -v`  
Expected: FAIL before docs/env update.

**Step 3: Update docs and env examples**

- Add exact backend examples for all three providers.
- Document migration impacts and stable UI workflow behavior.

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_config.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add README.md .env.example docs/plans/2026-02-22-llamaindex-full-migration-design.md tests/test_config.py
git commit -m "docs: update env and runtime docs for llamaindex provider architecture"
```

### Task 10: Final Verification Gate

**Files:**
- Modify if needed: failing test or lint fixes discovered during verification

**Step 1: Run full test suite**

Run: `pytest -v`  
Expected: PASS all tests.

**Step 2: Run application startup smoke check**

Run: `uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8099`  
Expected: app starts successfully with configured providers and no startup validation errors.

**Step 3: Perform manual UI workflow smoke checks**

- Upload a supported file.
- Execute chat and stream chat.
- Execute battleground comparison using two different backends.
- Confirm workflows complete end-to-end.

**Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: finalize llamaindex migration verification fixes"
```

