# Chat Model Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a required chat model dropdown that controls the model used by `/chat` and `/chat/stream` end-to-end.

**Architecture:** Reuse the existing configured battleground allowlist as the chat model source of truth, expose it via a dedicated `GET /models/chat` endpoint, and require a `model` field in chat payloads. The frontend loads chat models on page load, forces explicit selection, and sends the selected model with stream requests.

**Tech Stack:** FastAPI, Pydantic, vanilla JavaScript, pytest (API/integration/UI harness).

---

### Task 1: Extend chat API contracts for explicit model selection

**Files:**
- Modify: `app/main.py`
- Test: `tests/api/test_chat_api.py`

**Step 1: Write failing tests**
- Add tests for required `model` in chat payload and `GET /models/chat`.
- Update existing chat API tests to include `model` in payloads.

**Step 2: Run targeted tests to confirm failures**
Run: `pytest tests/api/test_chat_api.py -q`
Expected: FAIL because chat payload and endpoint contract are not implemented.

**Step 3: Implement minimal backend changes**
- Add `model` to `ChatRequest` with non-empty validation.
- Add `GET /models/chat` returning allowlisted models.
- Validate requested chat model against allowlist in `/chat` and `/chat/stream`.

**Step 4: Re-run tests**
Run: `pytest tests/api/test_chat_api.py -q`
Expected: PASS.

### Task 2: Thread selected model through chat service and OpenRouter client

**Files:**
- Modify: `app/services/chat.py`
- Modify: `app/services/openrouter_client.py`
- Test: `tests/services/test_chat.py`

**Step 1: Write failing tests**
- Update service fakes/assertions to require model override methods.

**Step 2: Run targeted tests to confirm failures**
Run: `pytest tests/services/test_chat.py -q`
Expected: FAIL because service/client signatures have not been updated.

**Step 3: Implement minimal model-aware behavior**
- Add model-aware methods to chat client protocol and OpenRouter client.
- Pass selected model from chat service into client generate/stream calls.
- Add detailed model logging for request traceability.

**Step 4: Re-run tests**
Run: `pytest tests/services/test_chat.py -q`
Expected: PASS.

### Task 3: Add chat model selector UI and wire request payload

**Files:**
- Modify: `app/templates/index.html`
- Modify: `app/static/js/chat.js`
- Test: `tests/ui/test_index_page.py`
- Test: `tests/integration/test_upload_chat_flow.py`

**Step 1: Write failing tests**
- Assert chat model selector exists on index page.
- Update JS harness fetch mocks for `/models/chat` and verify payload includes model.
- Update integration payloads to include model.

**Step 2: Run targeted tests to confirm failures**
Run: `pytest tests/ui/test_index_page.py tests/integration/test_upload_chat_flow.py -q`
Expected: FAIL because selector/fetch/payload are not yet implemented.

**Step 3: Implement minimal UI wiring**
- Add chat model dropdown near chat controls.
- Load options from `/models/chat` and require explicit selection.
- Send selected `model` in `/chat/stream` request body and fail with clear errors if unavailable.

**Step 4: Re-run tests**
Run: `pytest tests/ui/test_index_page.py tests/integration/test_upload_chat_flow.py -q`
Expected: PASS.

### Task 4: End-to-end verification

**Files:**
- Verify: `app/main.py`
- Verify: `app/services/chat.py`
- Verify: `app/services/openrouter_client.py`
- Verify: `app/static/js/chat.js`

**Step 1: Run focused suite**
Run: `pytest tests/api/test_chat_api.py tests/services/test_chat.py tests/ui/test_index_page.py tests/integration/test_upload_chat_flow.py -q`
Expected: PASS.

**Step 2: Run full test suite**
Run: `pytest -q`
Expected: PASS.
