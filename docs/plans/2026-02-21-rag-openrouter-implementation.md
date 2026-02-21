# OpenRouter RAG MVP Implementation Plan

**Goal:** Build a single-user FastAPI + Tailwind app that ingests `pdf/docx/txt` documents and provides strictly grounded chat answers with citations using OpenRouter and local Chroma.

**Architecture:** A monolithic FastAPI service with clear modules for config, logging, parsing, chunking, indexing, retrieval, and chat. The UI is server-rendered with Jinja2 and HTMX for upload/chat actions. Strict fail-fast behavior is enforced for missing config, unsupported files, empty extraction, and low-evidence retrieval.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Jinja2, HTMX, Tailwind CSS, ChromaDB, httpx, pypdf, python-docx, pytest

---

### Task 1: Bootstrap app and test scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `tests/conftest.py`
- Create: `tests/test_health_smoke.py`

**Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient
from app.main import create_app

def test_health_endpoint_exists():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_health_smoke.py::test_health_endpoint_exists -v`  
Expected: FAIL due to missing app factory or route.

**Step 3: Write minimal implementation**

```python
from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_health_smoke.py::test_health_endpoint_exists -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml app/__init__.py app/main.py tests/conftest.py tests/test_health_smoke.py
git commit -m "chore: bootstrap fastapi app and smoke test"
```

### Task 2: Add strict settings validation and structured logging

**Files:**
- Create: `app/config.py`
- Create: `app/logging_config.py`
- Modify: `app/main.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

```python
import os
import pytest
from app.config import Settings

def test_missing_required_env_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Settings.from_env()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_missing_required_env_raises -v`  
Expected: FAIL because `Settings` is not implemented.

**Step 3: Write minimal implementation**

```python
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_chat_model: str
    openrouter_embed_model: str
    chroma_persist_dir: str
    max_upload_mb: int
    retrieval_top_k: int
    min_relevance_score: float

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if value is None or value == "":
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=cls._require_env("OPENROUTER_API_KEY"),
            openrouter_chat_model=cls._require_env("OPENROUTER_CHAT_MODEL"),
            openrouter_embed_model=cls._require_env("OPENROUTER_EMBED_MODEL"),
            chroma_persist_dir=cls._require_env("CHROMA_PERSIST_DIR"),
            max_upload_mb=int(cls._require_env("MAX_UPLOAD_MB")),
            retrieval_top_k=int(cls._require_env("RETRIEVAL_TOP_K")),
            min_relevance_score=float(cls._require_env("MIN_RELEVANCE_SCORE")),
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_config.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/config.py app/logging_config.py app/main.py tests/test_config.py
git commit -m "feat: add fail-fast settings and structured logging"
```

### Task 3: Implement parsers for txt, docx, and pdf text layer

**Files:**
- Create: `app/services/parsers.py`
- Test: `tests/services/test_parsers.py`

**Step 1: Write failing tests**

```python
import pytest
from app.services.parsers import parse_text_file, UnsupportedFileTypeError, EmptyExtractionError

def test_txt_parser_returns_text(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello", encoding="utf-8")
    assert parse_text_file(file_path, "text/plain") == "hello"

def test_unsupported_file_type_raises(tmp_path):
    file_path = tmp_path / "a.csv"
    file_path.write_text("x,y", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        parse_text_file(file_path, "text/csv")

def test_empty_text_raises(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    with pytest.raises(EmptyExtractionError):
        parse_text_file(file_path, "text/plain")
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/services/test_parsers.py -v`  
Expected: FAIL due to missing parser module.

**Step 3: Write minimal implementation**

```python
from pathlib import Path

class UnsupportedFileTypeError(ValueError):
    pass

class EmptyExtractionError(ValueError):
    pass

def parse_text_file(path: Path, content_type: str) -> str:
    if content_type not in {"text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}:
        raise UnsupportedFileTypeError(f"Unsupported file type: {content_type}")
    text = path.read_text(encoding="utf-8") if content_type == "text/plain" else _parse_binary(path, content_type)
    if text.strip() == "":
        raise EmptyExtractionError(f"No extractable text found in {path.name}")
    return text
```

**Step 4: Expand implementation for docx/pdf and rerun tests**

Run: `pytest tests/services/test_parsers.py -v`  
Expected: PASS for txt and error behavior; add docx/pdf tests next and make them pass.

**Step 5: Commit**

```bash
git add app/services/parsers.py tests/services/test_parsers.py
git commit -m "feat: add strict txt/docx/pdf parsing with explicit errors"
```

### Task 4: Implement chunking with deterministic overlap

**Files:**
- Create: `app/services/chunking.py`
- Test: `tests/services/test_chunking.py`

**Step 1: Write failing tests**

```python
from app.services.chunking import chunk_text

def test_chunk_text_overlap():
    text = "a" * 30
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == ["a" * 10, "a" * 10, "a" * 10, "a" * 6]
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/services/test_chunking.py -v`  
Expected: FAIL due to missing function.

**Step 3: Write minimal implementation**

```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if text == "":
        raise ValueError("text must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >=0 and < chunk_size")
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks
```

**Step 4: Run tests**

Run: `pytest tests/services/test_chunking.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/chunking.py tests/services/test_chunking.py
git commit -m "feat: add deterministic chunking with validation"
```

### Task 5: Implement OpenRouter client for embeddings and chat

**Files:**
- Create: `app/services/openrouter_client.py`
- Test: `tests/services/test_openrouter_client.py`

**Step 1: Write failing tests**

```python
import pytest
from app.services.openrouter_client import OpenRouterClient

@pytest.mark.asyncio
async def test_embed_raises_on_non_200(httpx_mock):
    httpx_mock.add_response(status_code=401, json={"error": "bad key"})
    client = OpenRouterClient(api_key="k", embed_model="e", chat_model="c")
    with pytest.raises(RuntimeError, match="OpenRouter embeddings request failed"):
        await client.embed_texts(["hello"])
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/services/test_openrouter_client.py -v`  
Expected: FAIL due to missing class/module.

**Step 3: Write minimal implementation**

```python
import httpx

class OpenRouterClient:
    def __init__(self, api_key: str, embed_model: str, chat_model: str) -> None:
        self._api_key = api_key
        self._embed_model = embed_model
        self._chat_model = chat_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._embed_model, "input": texts},
            )
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter embeddings request failed: {response.status_code} {response.text}")
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]
```

**Step 4: Add chat method and run tests**

Run: `pytest tests/services/test_openrouter_client.py -v`  
Expected: PASS for embeddings and chat cases, plus explicit upstream-error tests.

**Step 5: Commit**

```bash
git add app/services/openrouter_client.py tests/services/test_openrouter_client.py
git commit -m "feat: add openrouter embeddings and chat client"
```

### Task 6: Implement Chroma wrapper and retrieval scorer

**Files:**
- Create: `app/services/vector_store.py`
- Create: `app/services/retrieval.py`
- Test: `tests/services/test_retrieval.py`

**Step 1: Write failing tests**

```python
from app.services.retrieval import filter_by_relevance

def test_filter_by_relevance_drops_weak_results():
    results = [{"score": 0.91}, {"score": 0.42}]
    assert filter_by_relevance(results, 0.6) == [{"score": 0.91}]
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/services/test_retrieval.py -v`  
Expected: FAIL due to missing retrieval module.

**Step 3: Write minimal implementation**

```python
def filter_by_relevance(results: list[dict], min_relevance_score: float) -> list[dict]:
    if len(results) == 0:
        raise ValueError("retrieval returned no results")
    kept = [item for item in results if item["score"] >= min_relevance_score]
    if len(kept) == 0:
        raise ValueError("no results passed relevance threshold")
    return kept
```

**Step 4: Implement Chroma storage/query and run tests**

Run: `pytest tests/services/test_retrieval.py -v`  
Expected: PASS including no-results and low-score error paths.

**Step 5: Commit**

```bash
git add app/services/vector_store.py app/services/retrieval.py tests/services/test_retrieval.py
git commit -m "feat: add chroma wrapper and strict retrieval filtering"
```

### Task 7: Implement upload ingestion endpoint

**Files:**
- Create: `app/services/ingest.py`
- Modify: `app/main.py`
- Test: `tests/api/test_upload_api.py`

**Step 1: Write failing API test**

```python
from fastapi.testclient import TestClient
from app.main import create_app

def test_upload_txt_indexes_document(tmp_path):
    client = TestClient(create_app())
    file_path = tmp_path / "a.txt"
    file_path.write_text("rag text", encoding="utf-8")
    with file_path.open("rb") as fh:
        response = client.post("/upload", files={"file": ("a.txt", fh, "text/plain")})
    assert response.status_code == 200
    assert "doc_id" in response.json()
    assert response.json()["chunks_indexed"] > 0
```

**Step 2: Run test to verify fail**

Run: `pytest tests/api/test_upload_api.py::test_upload_txt_indexes_document -v`  
Expected: FAIL due to missing endpoint.

**Step 3: Write minimal implementation**

```python
@app.post("/upload")
async def upload(file: UploadFile) -> dict[str, object]:
    if file.content_type is None:
        raise HTTPException(status_code=400, detail="Missing content type")
    result = await ingest_service.ingest_upload(file)
    return {"doc_id": result.doc_id, "chunks_indexed": result.chunks_indexed}
```

**Step 4: Run test suite for upload**

Run: `pytest tests/api/test_upload_api.py -v`  
Expected: PASS, with negative tests for unsupported mime and empty extraction.

**Step 5: Commit**

```bash
git add app/services/ingest.py app/main.py tests/api/test_upload_api.py
git commit -m "feat: add upload endpoint and ingestion workflow"
```

### Task 8: Implement strict-grounded chat endpoint with citations

**Files:**
- Create: `app/services/chat.py`
- Modify: `app/main.py`
- Test: `tests/api/test_chat_api.py`

**Step 1: Write failing chat tests**

```python
from fastapi.testclient import TestClient
from app.main import create_app

def test_chat_returns_unknown_when_no_evidence():
    client = TestClient(create_app())
    response = client.post("/chat", json={"message": "What is revenue?"})
    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert "do not know" in response.json()["answer"].lower()
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/api/test_chat_api.py -v`  
Expected: FAIL due to missing endpoint/service.

**Step 3: Write minimal implementation**

```python
@app.post("/chat")
async def chat(payload: ChatRequest) -> ChatResponse:
    return await chat_service.answer_question(payload.message)
```

```python
async def answer_question(self, message: str) -> ChatResponse:
    retrieved = await self._retrieval.retrieve(message)
    if len(retrieved) == 0:
        return ChatResponse(
            answer="I do not know from the provided documents.",
            citations=[],
            grounded=False,
            retrieved_count=0,
        )
    # build grounded prompt and call OpenRouter
```

**Step 4: Run tests**

Run: `pytest tests/api/test_chat_api.py -v`  
Expected: PASS for unknown path and grounded path with citations.

**Step 5: Commit**

```bash
git add app/services/chat.py app/main.py tests/api/test_chat_api.py
git commit -m "feat: add strict-grounded chat endpoint with citations"
```

### Task 9: Build Tailwind + HTMX UI

**Files:**
- Create: `app/templates/index.html`
- Create: `app/static/css/input.css`
- Create: `app/static/css/output.css`
- Create: `tailwind.config.js`
- Modify: `app/main.py`
- Test: `tests/ui/test_index_page.py`

**Step 1: Write failing UI test**

```python
from fastapi.testclient import TestClient
from app.main import create_app

def test_index_page_has_upload_and_chat():
    client = TestClient(create_app())
    response = client.get("/")
    html = response.text
    assert 'id="upload-form"' in html
    assert 'id="chat-form"' in html
```

**Step 2: Run tests to verify fail**

Run: `pytest tests/ui/test_index_page.py -v`  
Expected: FAIL due to missing template content.

**Step 3: Write minimal implementation**

```html
<form id="upload-form" hx-post="/upload" hx-encoding="multipart/form-data">
  <input type="file" name="file" accept=".pdf,.docx,.txt" required />
  <button type="submit">Upload</button>
</form>
<form id="chat-form" hx-post="/chat">
  <input type="text" name="message" required />
  <button type="submit">Ask</button>
</form>
```

**Step 4: Compile CSS and run tests**

Run: `npm run build:css`  
Run: `pytest tests/ui/test_index_page.py -v`  
Expected: PASS and styled page loads.

**Step 5: Commit**

```bash
git add app/templates/index.html app/static/css/input.css app/static/css/output.css tailwind.config.js tests/ui/test_index_page.py app/main.py
git commit -m "feat: add tailwind and htmx upload/chat interface"
```

### Task 10: End-to-end test pass and runbook

**Files:**
- Create: `tests/integration/test_upload_chat_flow.py`
- Create: `README.md`
- Modify: `docs/plans/2026-02-21-rag-openrouter-design.md`

**Step 1: Write failing integration test**

```python
def test_upload_then_chat_returns_grounded_answer(client, sample_txt_file):
    upload = client.post("/upload", files={"file": ("sample.txt", sample_txt_file, "text/plain")})
    assert upload.status_code == 200
    chat = client.post("/chat", json={"message": "What is in the document?"})
    assert chat.status_code == 200
    body = chat.json()
    assert "citations" in body
```

**Step 2: Run test to verify fail**

Run: `pytest tests/integration/test_upload_chat_flow.py -v`  
Expected: FAIL until all pieces are wired with test doubles.

**Step 3: Implement minimal glue fixes**

```python
# Ensure app startup constructs services with validated settings
# and injects them through FastAPI dependencies for test overriding.
```

**Step 4: Run full test suite**

Run: `pytest -v`  
Expected: PASS across unit, API, UI, and integration tests.

**Step 5: Commit**

```bash
git add tests/integration/test_upload_chat_flow.py README.md docs/plans/2026-02-21-rag-openrouter-design.md
git commit -m "test: add end-to-end upload to chat coverage and runbook"
```

## Execution Notes

- Keep all modules below 800 lines per file.
- Keep each function below 80 lines with single responsibility.
- Do not introduce defaults/fallbacks for required inputs or required configuration.
- Use detailed logging in every service boundary and endpoint.
- Use dependency injection for testability.
- Keep commit sizes small and aligned with each task.

Plan complete and saved to `docs/plans/2026-02-21-rag-openrouter-implementation.md`
