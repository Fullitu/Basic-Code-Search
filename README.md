## Basic Code Search

Lightweight experimentation framework for code-aware semantic search built on top of Sentence Transformers, Qdrant, and the CosQA benchmark. The repository demonstrates how to ingest text from PDFs, embed it with transformer models, persist vectors in Qdrant, and evaluate retrieval quality before and after fine-tuning.

### Key Capabilities
- Document ingestion pipeline that extracts text from PDFs and chunks it for dense retrieval.
- Embedding wrapper that supports both off-the-shelf and fine-tuned Sentence Transformer models.
- Qdrant-backed vector store with convenience methods for collection setup, batch upserts, and search.
- Retrieval evaluation against CosQA using Recall@10, MRR@10, and NDCG@10 metrics.
- Fine-tuning workflows for Multiple Negatives Ranking Loss and Contrastive Loss with tracked loss curves.

### Project Layout
- `src/basic_code_search/document_loader.py` — PDF ingestion and chunking helpers.
- `src/basic_code_search/embedding_model.py` — Sentence Transformer wrapper with training hooks.
- `src/basic_code_search/database_client.py` — Qdrant client orchestration (Docker-based startup).
- `src/basic_code_search/search_engine.py` — End-to-end search engine including evaluation helpers.
- `src/basic_code_search/metrics.py` — Retrieval metric implementations (Recall@10, MRR@10, NDCG@10).
- `report.ipynb` — Narrative walkthrough covering preprocessing, evaluation, and fine-tuning experiments.
- `results/` — Saved fine-tuned checkpoints for CosQA experiments.

### Prerequisites
- Python 3.10+ recommended.
- Docker Desktop running locally (required for the embedded Qdrant instance).
- Adequate disk space and bandwidth to download Sentence Transformer checkpoints and CosQA datasets.

### Environment Setup
1. Create and activate a virtual environment.
	```cmd
	python -m venv .venv
	.venv\Scripts\activate
	```
2. Install dependencies.
	```cmd
	pip install -r requirements.txt
	```
3. (Optional) Install additional Jupyter-related packages if you plan to rerun `report.ipynb`.

### Running the Search Engine Demo
1. Ensure Docker Desktop is running.
2. Populate the `data/` folder with PDF files you want to index.
3. Execute the workflow described in `report.ipynb` or adapt the code snippets from Task 1:
	- Load and chunk documents via `load_pdf_documents` and `RecursiveCharacterTextSplitter`.
	- Instantiate `EmbeddingModel` with `sentence-transformers/all-MiniLM-L6-v2`.
	- Create a `SearchEngine`, call `open()`, and invoke `load_text_data(...)` to upsert embeddings.
	- Query with `search_engine.search("What is a transformer model?")` (or any custom question).
4. Call `search_engine.close()` when finished to shut down the Dockerized Qdrant container.

### Evaluating on CosQA
The notebook demonstrates metric computation against the CosQA benchmark:
- Load CosQA splits with `datasets.load_dataset`.
- Populate Qdrant using the corpus split and run retrieval for each query.
- Evaluate predictions against ground-truth corpus IDs using `search_engine.evaluate(...)`.
Typical baseline results (all-MiniLM-L6-v2) observed during development:
- `Recall@10 ≈ 0.476`
- `MRR@10 ≈ 0.244`
- `NDCG@10 ≈ 0.299`

### Fine-Tuning Recipes
- **Multiple Negatives Ranking Loss**: Yields improved retrieval quality (Recall@10 ≈ 0.530, MRR@10 ≈ 0.248, NDCG@10 ≈ 0.315). Training artifacts land in `results/fine-tuned/cosqa-mnr-loss/`.
- **Contrastive Loss**: Included as an alternative experiment; current run underperforms the baseline, likely due to overfitting or data handling nuances. Artifacts live in `results/fine-tuned/cosqa-contrastive-loss/`.

You can reproduce these workflows by executing the Task 3 cells inside `report.ipynb`, which perform dataset reshaping, launch `SentenceTransformerTrainer`, plot learning curves, and re-evaluate the fine-tuned checkpoints.

### Troubleshooting Tips
- Qdrant startup failures usually mean Docker Desktop is not running—start it and retry `search_engine.open()`.
- Large upserts may require increasing the `batch_size` parameter in `load_text_data` to balance memory usage and throughput.
- Hugging Face dataset downloads respect the local cache; clear `~/.cache/huggingface` if you need a clean slate.
- When fine-tuning, monitor GPU/CPU memory usage; adjust `per_device_*_batch_size` in `SentenceTransformerTrainingArguments` as needed.

### Next Steps
- Add a CLI that wraps document ingestion and search queries.
- Deploy Qdrant as a managed service and update `DatabaseClient` to target remote instances.
- Extend evaluations with additional datasets or hard negative mining strategies.
