# RAG Pipeline Design (OpenRouter, FastAPI, Tailwind)

Date: 2026-02-21
Status: Validated brainstorm design
Scope: MVP, single-user, locally persisted vector store

## 1. Decisions Locked

- Vector store: Local Chroma (persisted on disk).
- User model: Single-user app (no auth in v1).
- Answer policy: Strict grounding only.
- Parsing policy: Text-only extraction.
- Supported uploads: `pdf`, `docx`, `txt`.
- Model provider: OpenRouter for both embeddings and chat.
- UI delivery: FastAPI server-rendered templates with Tailwind CSS and HTMX interactions.

## 2. Goals and Non-Goals

Goals:
- Upload documents and index them for retrieval.
- Ask questions in a chat UI and receive document-grounded answers.
- Return explicit citations for each answer.
- Return a clear "I do not know from the provided documents" response when evidence is insufficient.

Non-goals:
- Multi-user isolation and auth.
- OCR for scanned PDFs.
- Agentic tool orchestration.
- Managed cloud vector databases.

## 3. Architecture

The MVP is a single FastAPI application containing four backend flows:
1. Ingestion: file validation and text extraction.
2. Indexing: chunking and embedding generation.
3. Retrieval: top-k semantic search with relevance gating.
4. Chat: strict grounded generation with citations.

The frontend is a single page rendered by FastAPI using Jinja2 templates and styled with Tailwind. HTMX handles upload and chat requests without a separate SPA runtime. This minimizes moving parts and reduces implementation risk for v1.

## 4. Data Flow

### Upload path (`POST /upload`)
1. Validate file extension and size.
2. Parse text using format-specific extractor:
- PDF: text layer only.
- DOCX: paragraph text extraction.
- TXT: direct decode with explicit encoding handling.
3. Fail fast if no text is extracted.
4. Chunk extracted text with overlap.
5. Call OpenRouter embeddings API for all chunks.
6. Upsert vectors and metadata into Chroma.
7. Return upload/index summary (`doc_id`, `chunks_indexed`).

### Chat path (`POST /chat`)
1. Accept user question.
2. Embed question via OpenRouter embeddings model.
3. Retrieve top-k chunks from Chroma.
4. Enforce `MIN_RELEVANCE_SCORE` threshold.
5. If insufficient evidence, return strict unknown response.
6. Otherwise call OpenRouter chat model with grounded prompt and retrieved context.
7. Return answer and citations.

## 5. Strict Grounding Rules

Grounding is enforced in both prompt and code:
- Prompt constraint: answer only from provided context.
- Citation requirement: include source references for claims.
- Backend gate: if retrieval evidence is weak, do not generate speculative answers.
- Unknown handling: return a deterministic fallback message explaining evidence is unavailable in uploaded docs.

No fallback/default behavior is allowed for missing critical config or missing retrieval evidence.

## 6. API Surface

- `GET /`: render upload + chat interface.
- `POST /upload`: upload and index one document.
- `POST /chat`: retrieve and answer with citations.
- `GET /health`: readiness check for app and vector store.

## 7. Project Structure

```text
app/
  main.py
  config.py
  logging.py
  services/
    ingest.py
    parsers.py
    chunking.py
    vector_store.py
    retrieval.py
    openrouter_client.py
    chat.py
  templates/
    index.html
  static/
    css/
      output.css
tests/
docs/plans/
```

## 8. Configuration Contract

All config values are required and must raise startup exceptions if missing:
- `OPENROUTER_API_KEY`
- `OPENROUTER_CHAT_MODEL`
- `OPENROUTER_EMBED_MODEL`
- `CHROMA_PERSIST_DIR`
- `MAX_UPLOAD_MB`
- `RETRIEVAL_TOP_K`
- `MIN_RELEVANCE_SCORE`

## 9. Logging and Exceptions

- Use structured logging with request IDs.
- Log timing for parse, chunk, embed, retrieve, and generate stages.
- Raise explicit exceptions for:
- Unsupported file type.
- Empty or non-extractable text payload.
- Missing required environment config.
- Upstream OpenRouter API errors.

## 10. Milestones

1. Foundation
- FastAPI app skeleton, config validation, logging setup, `.gitignore`.

2. Ingestion and Indexing
- Upload endpoint, parsers (`pdf`, `docx`, `txt`), chunker, Chroma upsert.

3. Retrieval and Chat
- Query embedding, top-k retrieval, relevance gating, strict grounded answer generation.

4. UI and Tests
- Tailwind + HTMX UI for upload/chat.
- Unit tests for parsers/chunking/retrieval gating.
- Integration test for upload to chat flow.
- Negative tests for unsupported file and empty extraction.

## 11. Success Criteria

- Supported files upload and index successfully.
- Chat answers are grounded and cited.
- No-evidence queries return explicit unknown response.
- Configuration failures surface immediately at startup.

