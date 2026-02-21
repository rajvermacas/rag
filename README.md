# RAG OpenRouter App

Single-user RAG pipeline built with FastAPI and Tailwind CSS.

## Features

- Upload and index `pdf`, `docx`, and `txt` files
- Local vector storage with ChromaDB
- OpenRouter embeddings + chat generation
- Strict grounded answers with citations
- Fail-fast configuration validation (no silent defaults)

## Required Environment Variables

All variables are required. App startup fails if any are missing.
The app loads values from `.env` automatically at startup via `python-dotenv`.

```bash
OPENROUTER_API_KEY=...
OPENROUTER_CHAT_MODEL=...
OPENROUTER_EMBED_MODEL=...
CHROMA_PERSIST_DIR=./chroma
CHROMA_COLLECTION_NAME=rag_docs
MAX_UPLOAD_MB=25
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=5
MIN_RELEVANCE_SCORE=0.75
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
- `POST /chat` grounded question answering
- `GET /health` health check
