# Model Battleground + Chat Greeting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new Model Battleground tab that streams two RAG responses side by side for user-selected models, and update new-chat greeting text to `Hello! How can I assist you today?`.

**Architecture:** Keep existing FastAPI app boundaries and add a dedicated `BattlegroundService` that performs retrieval once, then fans out two model streams with identical prompts and multiplexed NDJSON output. Split frontend JavaScript from `index.html` into focused modules (`common`, `chat`, `battleground`) to stay under the 800-line template limit and preserve single responsibility.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, async iterators/StreamingResponse, OpenRouter API, vanilla JS, Tailwind CSS, pytest

---

**Execution rules for this plan:**
- Follow `@superpowers/test-driven-development` for every code task.
- Run `@superpowers/verification-before-completion` before any “done” claim.
- Keep functions under 80 LOC and split if needed.
- Keep files under 800 lines (explicitly check `app/templates/index.html` after refactor).
- No fallback/default values for required inputs/config; raise explicit exceptions.

### Task 1: Add strict battleground model allowlist config

**Files:**
- Modify: `app/config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/conftest.py`
- Modify: `.env.example`

**Step 1: Write failing tests**

```python
def test_missing_battleground_models_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_BATTLEGROUND_MODELS", raising=False)
    with pytest.raises(
        ValueError,
        match="Missing required environment variable: OPENROUTER_BATTLEGROUND_MODELS",
    ):
        Settings.from_env()


def test_invalid_battleground_models_empty_entry_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_BATTLEGROUND_MODELS", "openai/gpt-4o,,anthropic/claude")
    with pytest.raises(
        ValueError,
        match="OPENROUTER_BATTLEGROUND_MODELS contains empty model id",
    ):
        Settings.from_env()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -k battleground -v`
Expected: FAIL because `Settings` has no battleground field/parser.

**Step 3: Write minimal implementation**

```python
def _parse_required_csv(name: str) -> list[str]:
    raw_value = _require_env(name)
    values = [value.strip() for value in raw_value.split(",")]
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one model id")
    if any(value == "" for value in values):
        raise ValueError(f"{name} contains empty model id")
    return values


@dataclass(frozen=True)
class Settings:
    openrouter_battleground_models: list[str]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ...,
            openrouter_battleground_models=_parse_required_csv(
                "OPENROUTER_BATTLEGROUND_MODELS"
            ),
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/config.py tests/test_config.py tests/conftest.py .env.example
git commit -m "feat: add strict battleground model allowlist config"
```

### Task 2: Extend OpenRouter client for model override streaming

**Files:**
- Modify: `app/services/openrouter_client.py`
- Modify: `tests/services/test_openrouter_client.py`

**Step 1: Write failing tests**

```python
def test_stream_chat_with_model_override_uses_given_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_models: list[str] = []

    async def fake_stream(self, path: str, payload: dict):
        captured_models.append(payload["model"])
        yield '{"choices":[{"delta":{"content":"ok"}}]}'

    monkeypatch.setattr(OpenRouterClient, "_stream_post_data_lines", fake_stream)
    client = OpenRouterClient(api_key="k", embed_model="embed", chat_model="default-model")

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in client.stream_chat_response_with_model(
                "custom/model", "system", "user"
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks == ["ok"]
    assert captured_models == ["custom/model"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_openrouter_client.py -k model_override -v`
Expected: FAIL because method does not exist.

**Step 3: Write minimal implementation**

```python
async def stream_chat_response_with_model(
    self, model: str, system_prompt: str, user_prompt: str
) -> AsyncIterator[str]:
    if model.strip() == "":
        raise ValueError("model must not be empty")
    _validate_prompt_inputs(system_prompt, user_prompt)
    payload = _build_chat_payload(model, system_prompt, user_prompt, stream=True)
    async for data_line in self._stream_post_data_lines("/chat/completions", payload):
        chunk = _extract_stream_chunk(data_line)
        if chunk != "":
            yield chunk
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_openrouter_client.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/openrouter_client.py tests/services/test_openrouter_client.py
git commit -m "feat: add model override chat streaming in openrouter client"
```

### Task 3: Add battleground orchestration service (single retrieval, dual stream)

**Files:**
- Create: `app/services/battleground.py`
- Create: `tests/services/test_battleground.py`
- Modify: `app/services/__init__.py`

**Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_compare_stream_retrieves_once_and_tags_sides() -> None:
    retrieval_calls = 0

    class FakeRetrieval:
        async def retrieve(self, question: str):
            nonlocal retrieval_calls
            retrieval_calls += 1
            return [fake_chunk()]

    class FakeChatClient:
        async def stream_chat_response_with_model(self, model, system_prompt, user_prompt):
            if model == "model-a":
                for part in ["A1", "A2"]:
                    yield part
                return
            for part in ["B1", "B2"]:
                yield part

    service = BattlegroundService(...)
    events = [event async for event in service.compare_stream("q", [], "model-a", "model-b")]

    assert retrieval_calls == 1
    assert {event.side for event in events if event.kind == "chunk"} == {"A", "B"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_battleground.py -v`
Expected: FAIL because service module does not exist.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class CompareEvent:
    side: str
    kind: str
    chunk: str | None
    error: str | None


class BattlegroundService:
    async def compare_stream(
        self, question: str, history: list[ConversationTurn], model_a: str, model_b: str
    ) -> AsyncIterator[CompareEvent]:
        _validate_compare_inputs(question, model_a, model_b, self._allowed_models)
        chunks = await self._retrieve_chunks_or_empty(_build_retrieval_query(question, history))
        system_prompt = _build_system_prompt(len(chunks) > 0)
        user_prompt = _build_user_prompt(question, history, chunks, self._document_service.list_documents())
        async for event in _merge_model_streams(
            self._chat_client.stream_chat_response_with_model(model_a, system_prompt, user_prompt),
            self._chat_client.stream_chat_response_with_model(model_b, system_prompt, user_prompt),
        ):
            yield event
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/services/test_battleground.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/battleground.py app/services/__init__.py tests/services/test_battleground.py
git commit -m "feat: add battleground service with shared retrieval compare stream"
```

### Task 4: Add battleground API routes and streaming response format

**Files:**
- Modify: `app/main.py`
- Create: `tests/api/test_battleground_api.py`
- Modify: `tests/api/test_chat_api.py`

**Step 1: Write failing tests**

```python
def test_get_battleground_models_returns_allowlist(client: TestClient) -> None:
    response = client.get("/models/battleground")
    assert response.status_code == 200
    assert response.json() == {
        "models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
    }


def test_compare_stream_returns_tagged_ndjson(client: TestClient) -> None:
    response = client.post(
        "/battleground/compare/stream",
        json={"message": "hi", "model_a": "model-a", "model_b": "model-b", "history": []},
    )
    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip() != ""]
    first = json.loads(lines[0])
    assert first["side"] in {"A", "B"}
    assert "chunk" in first or "error" in first or "done" in first
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_battleground_api.py -v`
Expected: FAIL because routes/models do not exist.

**Step 3: Write minimal implementation**

```python
class BattlegroundModelsResponse(BaseModel):
    models: list[str]


class BattlegroundCompareRequest(BaseModel):
    message: str
    history: list[ChatHistoryTurn]
    model_a: str
    model_b: str


@app.get("/models/battleground")
async def list_battleground_models() -> BattlegroundModelsResponse:
    return BattlegroundModelsResponse(models=settings.openrouter_battleground_models)


@app.post("/battleground/compare/stream")
async def battleground_compare_stream(payload: BattlegroundCompareRequest) -> StreamingResponse:
    stream = services.battleground_service.compare_stream(
        payload.message,
        [ConversationTurn(role=turn.role, message=turn.message) for turn in payload.history],
        payload.model_a,
        payload.model_b,
    )
    return StreamingResponse(_stream_battleground_events(stream), media_type="application/x-ndjson")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_battleground_api.py tests/api/test_chat_api.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/main.py tests/api/test_battleground_api.py tests/api/test_chat_api.py
git commit -m "feat: add battleground models and compare stream endpoints"
```

### Task 5: Split `index.html` JavaScript into modular static files

**Files:**
- Modify: `app/templates/index.html`
- Create: `app/static/js/common.js`
- Create: `app/static/js/chat.js`
- Create: `app/static/js/battleground.js`
- Modify: `tests/ui/test_index_page.py`

**Step 1: Write failing UI tests**

```python
def test_index_page_includes_modular_scripts(...) -> None:
    response = client.get("/")
    html = response.text
    assert 'src="/static/js/common.js"' in html
    assert 'src="/static/js/chat.js"' in html
    assert 'src="/static/js/battleground.js"' in html


def test_index_page_has_battleground_tab(...) -> None:
    response = client.get("/")
    html = response.text
    assert 'id="nav-chat"' in html
    assert 'id="nav-battleground"' in html
    assert 'id="battleground-form"' in html
    assert 'id="model-a-select"' in html
    assert 'id="model-b-select"' in html
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_index_page.py -v`
Expected: FAIL because scripts/tab IDs do not exist.

**Step 3: Write minimal implementation**

```html
<!-- in index.html -->
<script src="/static/js/common.js"></script>
<script src="/static/js/chat.js"></script>
<script src="/static/js/battleground.js"></script>
```

```javascript
// common.js
window.RagUiCommon = { requireElement, requireString, requireNumber, renderMarkdown, escapeHtml };

// chat.js
window.RagChat = { initializeChatSessions, clearActiveChat, ... };

// battleground.js
window.RagBattleground = { initializeBattleground, submitComparePrompt, ... };
```

**Step 4: Run tests + line-count check**

Run: `pytest tests/ui/test_index_page.py -v`
Expected: PASS.

Run: `wc -l app/templates/index.html`
Expected: line count is `< 800`.

**Step 5: Commit**

```bash
git add app/templates/index.html app/static/js/common.js app/static/js/chat.js app/static/js/battleground.js tests/ui/test_index_page.py
git commit -m "refactor: split ui scripts and add battleground tab scaffold"
```

### Task 6: Update chat new-session default greeting text

**Files:**
- Modify: `app/static/js/chat.js`
- Modify: `tests/ui/test_index_page.py`

**Step 1: Write failing test**

```python
def test_index_page_uses_new_default_greeting(...) -> None:
    response = client.get("/")
    html = response.text
    assert "Hello! How can I assist you today?" in html
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_index_page.py -k greeting -v`
Expected: FAIL because old greeting string is still present.

**Step 3: Write minimal implementation**

```javascript
const NEW_CHAT_GREETING = "Hello! How can I assist you today?";
...
appendAssistantMessage(NEW_CHAT_GREETING);
...
function clearActiveChat() {
  ...
  appendAssistantMessage(NEW_CHAT_GREETING);
}
```

**Step 4: Run tests to verify it passes**

Run: `pytest tests/ui/test_index_page.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/static/js/chat.js tests/ui/test_index_page.py
git commit -m "feat: update new chat default greeting"
```

### Task 7: Implement battleground frontend behavior (model loading + parallel compare stream)

**Files:**
- Modify: `app/static/js/battleground.js`
- Modify: `app/templates/index.html`
- Modify: `tests/ui/test_index_page.py`

**Step 1: Write failing tests**

```python
def test_index_page_has_battleground_response_panes(...) -> None:
    response = client.get("/")
    html = response.text
    assert 'id="battleground-response-a"' in html
    assert 'id="battleground-response-b"' in html
    assert "Model A" in html
    assert "Model B" in html
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_index_page.py -k battleground -v`
Expected: FAIL until final markup IDs are in place.

**Step 3: Write minimal implementation**

```javascript
async function loadBattlegroundModels() {
  const response = await fetch("/models/battleground");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(requireString(payload, "detail", "battleground model list error"));
  }
  renderModelOptions(payload.models);
}

async function runBattlegroundCompare(message, modelA, modelB) {
  const response = await fetch("/battleground/compare/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history: [], model_a: modelA, model_b: modelB }),
  });
  await consumeNdjsonStream(response.body, handleBattlegroundEvent);
}
```

**Step 4: Run tests + quick manual smoke**

Run: `pytest tests/ui/test_index_page.py -v`
Expected: PASS.

Run: `pytest tests/api/test_battleground_api.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/static/js/battleground.js app/templates/index.html tests/ui/test_index_page.py
git commit -m "feat: add battleground ui streaming compare flow"
```

### Task 8: Final docs and end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Optional modify: `docs/plans/2026-02-21-model-battleground-design.md` (status links)

**Step 1: Write failing doc checks (manual checklist)**

- Confirm README is missing battleground endpoints and env var.
- Confirm `.env.example` includes new required env key.

**Step 2: Update docs**

Add to README:
- New env var `OPENROUTER_BATTLEGROUND_MODELS`
- New APIs:
  - `GET /models/battleground`
  - `POST /battleground/compare/stream`
- Brief user flow for battleground tab.

**Step 3: Run full verification suite**

Run: `pytest -v`
Expected: PASS.

Run: `npm run build:css`
Expected: PASS.

Run: `wc -l app/templates/index.html app/static/js/common.js app/static/js/chat.js app/static/js/battleground.js`
Expected: each file is `< 800` lines.

**Step 4: Commit final docs/verification updates**

```bash
git add README.md .env.example
# if design doc status updated, include it too
# git add docs/plans/2026-02-21-model-battleground-design.md
git commit -m "docs: document battleground configuration and api usage"
```

**Step 5: Prepare branch handoff**

Run:
- `git status --short`
- `git log --oneline -n 10`

Expected:
- clean status (or clearly listed intentional leftovers)
- commit history aligned to tasks above.
