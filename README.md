# RAG OpenRouter App

Single-user RAG pipeline built with FastAPI and Tailwind CSS.

## Features

- Upload and index `pdf`, `docx`, and `txt` files
- Local vector storage with ChromaDB
- OpenRouter embeddings + chat generation
- Hybrid answers with document citations plus general knowledge
- Chat history sessions with new-chat greeting bootstrap
- Model Battleground tab for side-by-side model comparison with shared RAG context
- Fail-fast configuration validation (no silent defaults)

## Required Environment Variables

All variables are required. App startup fails if any are missing.
The app loads values from `.env` automatically at startup via `python-dotenv`.

```bash
OPENROUTER_API_KEY=...
OPENROUTER_CHAT_MODEL=...
OPENROUTER_EMBED_MODEL=...
OPENROUTER_BATTLEGROUND_MODELS=openai/gpt-4o-mini,anthropic/claude-3.5-sonnet
CHROMA_PERSIST_DIR=./chroma
CHROMA_COLLECTION_NAME=rag_docs
MAX_UPLOAD_MB=25
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=5
MIN_RELEVANCE_SCORE=0.4
APP_LOG_LEVEL=INFO
```

## Install

```bash
python -m pip install -e ".[dev]"
npm install
```

## Run

```bash
npm run build:css
uvicorn app.main:create_app --factory --reload
```

## Test

```bash
pytest -v
```

## API

- `GET /` UI page
- `POST /upload` upload and index file
- `GET /documents` list indexed documents
- `DELETE /documents/{doc_id}` remove one indexed document
- `POST /chat` non-streaming hybrid document-grounded + general-knowledge answering
- `POST /chat/stream` streaming chat response
- `GET /models/battleground` list configured battleground model IDs
- `POST /battleground/compare/stream` streaming side-by-side battleground comparison
- `GET /health` health check

### Chat Request Body

`POST /chat` requires both fields:

- `message`: current user message
- `history`: ordered prior turns, each with `role` (`user` or `assistant`) and `message`

### Battleground Compare Request Body

`POST /battleground/compare/stream` requires:

- `message`: user prompt for comparison
- `history`: currently sent as an empty array (`[]`)
- `model_a`: selected model ID for left panel
- `model_b`: selected model ID for right panel
