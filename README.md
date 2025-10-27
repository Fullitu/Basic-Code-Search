# Basic Code Search

This project provides a minimal semantic search API backed by the Qdrant vector database and a Hugging Face sentence-transformer embedding model. Supply a set of documents at startup, and the service exposes an HTTP endpoint to query the corpus.

## Features
- Sentence-transformer embeddings via Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`).
- Qdrant vector storage (runs in-memory by default, optionally persistent).
- FastAPI service with `/search` endpoint returning the best matching documents.
- Simple JSON loader for bootstrapping a corpus at startup.

## Getting Started

### 1. Install dependencies
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare documents
Create a JSON file containing an array of objects with `id`, `title`, and `content` fields (see `data/sample_documents.json`). You can change the source file path with the `DOCUMENTS_PATH` environment variable.

### 3. Run the API
```cmd
set DOCUMENTS_PATH=data\sample_documents.json
uvicorn main:app --reload
```

### 4. Query the API
Send a POST request to `/search` with a JSON payload containing a `query` string and optional `limit`.

```cmd
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d "{\"query\": \"python search API\", \"limit\": 3}"
```

## Configuration
- `DOCUMENTS_PATH`: Optional path to the JSON file with documents.
- `QDRANT_LOCATION`: If set, passed to `QdrantClient` as `location` for persistence (e.g., `"qdrant_data"`).
- `EMBED_MODEL_NAME`: Optional Hugging Face model name.

## Testing the Search Logic
```cmd
pytest
```

## Project Structure
```
README.md
requirements.txt
main.py
src/basic_code_search/
    __init__.py
    api.py
    embeddings.py
    search_engine.py
    vector_store.py
tests/
    test_search_engine.py
data/
    sample_documents.json
```
