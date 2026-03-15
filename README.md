# GenAI Knowledge Assistant

A source-grounded Retrieval Augmented Generation (RAG) system that indexes local documents, retrieves relevant chunks with FAISS, and answers questions with OpenAI models or an offline extractive fallback.

## Features

- Document ingestion for `md`, `txt`, and `pdf` files.
- Chunking and embedding pipelines for vector search indexing.
- FAISS cosine-similarity retrieval with persisted index artifacts.
- Source-grounded prompting with citation validation to reduce hallucinations.
- Evaluation utilities for answer quality and retrieval performance.
- CLI workflow for indexing, querying, and benchmarking.

## Project Structure

```text
genai-knowledge-assistant/
|-- app/
|   |-- cli.py
|   |-- config.py
|   |-- documents.py
|   |-- embeddings.py
|   |-- evaluation.py
|   |-- prompting.py
|   |-- rag.py
|   |-- validation.py
|   `-- vector_store.py
|-- data/
|   |-- evaluation_examples.json
|   `-- sample_docs/
|-- tests/
|-- .env.example
`-- pyproject.toml
```

## Quick Start

1. Create and activate a virtual environment.
2. Install the project with dev dependencies.
3. Export `OPENAI_API_KEY` if you want OpenAI-backed embeddings and generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

## Index Sample Documents

OpenAI mode:

```bash
rag-assistant index
```

Offline demo mode:

```bash
rag-assistant index --embedding-backend hashing
```

## Ask Questions

OpenAI generation:

```bash
rag-assistant ask --question "How quickly must security incidents be reported?"
```

Offline extractive mode:

```bash
rag-assistant ask \
  --question "How many remote days are allowed per week?" \
  --embedding-backend hashing \
  --generation-backend extractive
```

## Streamlit Demo UI

Run the local demo UI:

```bash
streamlit run streamlit_app.py
```

The app lets you:

- Index the bundled sample documents or upload your own `md`, `txt`, and `pdf` files.
- Switch between offline demo mode (`hashing` + `extractive`) and OpenAI-backed mode.
- Ask questions in a chat interface.
- Inspect retrieved chunks and relevance scores.
- Run the bundled evaluation set when using the sample corpus.

## Evaluate Retrieval and Answers

```bash
rag-assistant evaluate \
  --embedding-backend hashing \
  --generation-backend extractive
```

This writes an `evaluation_report.json` file inside `data/index/` with:

- Token-overlap answer F1
- Retrieval hit rate
- Source precision@k
- Source recall@k
- Mean reciprocal rank

## Resume Mapping

- Retrieval Augmented Generation system: `app/rag.py` orchestrates embedding, retrieval, prompting, and answer generation.
- Embedding pipelines and vector search: `app/embeddings.py` and `app/vector_store.py`.
- Hallucination reduction: `app/prompting.py` injects source context and `app/validation.py` enforces citation checks.
- Automated evaluation: `app/evaluation.py` scores answer quality and retrieval behavior.

## Notes

- `text-embedding-3-small` is the default embedding model.
- `gpt-4.1-mini` is the default answer-generation model.
- If you index with one embedding backend, query with the same backend for consistent retrieval.
- Use `hashing` + `extractive` when you want a stable live demo without depending on an API key.
