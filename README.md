# RAG OpenRouter App

Single-user RAG pipeline built with FastAPI, Tailwind CSS, and a LlamaIndex-centered runtime.

## Features

- Upload and index `pdf`, `docx`, and `txt` files
- Local vector storage with ChromaDB
- OpenRouter embeddings + configurable multi-backend chat generation
- Backend/model allowlists across `openrouter`, `openai`, and `azure_openai`
- LlamaIndex-backed ingestion and query execution on ChromaDB
- Chat history sessions with new-chat greeting bootstrap
- Model Battleground tab for side-by-side model comparison with shared RAG context
- Fail-fast configuration validation (no silent defaults)

## Required Environment Variables

All variables are required. App startup fails if any are missing.
The app loads values from `.env` automatically at startup via `python-dotenv`.

```bash
OPENROUTER_API_KEY=...
OPENROUTER_EMBED_MODEL=text-embedding-3-small
CHAT_BACKEND_IDS=openrouter_lab,openai_prod,azure_prod

CHAT_BACKEND_OPENROUTER_LAB_PROVIDER=openrouter
CHAT_BACKEND_OPENROUTER_LAB_MODELS=openai/gpt-4o-mini,anthropic/claude-3.5-sonnet
CHAT_BACKEND_OPENROUTER_LAB_API_KEY=...

CHAT_BACKEND_OPENAI_PROD_PROVIDER=openai
CHAT_BACKEND_OPENAI_PROD_MODELS=gpt-4o-mini,gpt-4.1-mini
CHAT_BACKEND_OPENAI_PROD_API_KEY=...

CHAT_BACKEND_AZURE_PROD_PROVIDER=azure_openai
CHAT_BACKEND_AZURE_PROD_MODELS=gpt-4o-mini
CHAT_BACKEND_AZURE_PROD_API_KEY=...
CHAT_BACKEND_AZURE_PROD_AZURE_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com
CHAT_BACKEND_AZURE_PROD_AZURE_API_VERSION=2024-10-21
CHAT_BACKEND_AZURE_PROD_AZURE_DEPLOYMENTS=gpt-4o-mini=chat-gpt4o-mini

CHROMA_PERSIST_DIR=/workspaces/rag/chroma
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
uv run uvicorn app.main:create_app --factory --reload
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
- `GET /models/chat` list provider-aware chat model options
- `POST /chat` non-streaming LlamaIndex query-engine answer
- `POST /chat/stream` streaming chat response
- `GET /models/battleground` list provider-aware battleground model options
- `POST /battleground/compare/stream` streaming side-by-side battleground comparison
- `GET /health` health check

### Chat Request Body

`POST /chat` requires both fields:

- `message`: current user message
- `history`: ordered prior turns, each with `role` (`user` or `assistant`) and `message`
- `backend_id`: selected chat backend profile ID
- `model`: selected model ID allowed by the chosen backend

### Battleground Compare Request Body

`POST /battleground/compare/stream` requires:

- `message`: user prompt for comparison
- `history`: ordered prior turns (`user`/`assistant`)
- `model_a_backend_id`: selected backend ID for left panel
- `model_a`: selected model ID for left panel
- `model_b_backend_id`: selected backend ID for right panel
- `model_b`: selected model ID for right panel
