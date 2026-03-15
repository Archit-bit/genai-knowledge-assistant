from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import DocumentChunk, RetrievedChunk

try:
    import faiss
except ImportError:  # pragma: no cover - handled at runtime.
    faiss = None


INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


class FaissVectorStore:
    def __init__(self, index, chunks: list[DocumentChunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def from_embeddings(
        cls,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> "FaissVectorStore":
        _require_faiss()
        if not chunks:
            raise ValueError("At least one chunk is required to build the index.")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        matrix = _normalize_rows(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, chunks=list(chunks))

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        _require_faiss()
        index_path = directory / INDEX_FILENAME
        chunks_path = directory / CHUNKS_FILENAME
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Index artifacts not found in {directory}")

        index = faiss.read_index(str(index_path))
        chunks = [
            DocumentChunk.from_dict(payload)
            for payload in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        return cls(index=index, chunks=chunks)

    def save(self, directory: Path) -> None:
        _require_faiss()
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        payload = [chunk.to_dict() for chunk in self.chunks]
        (directory / CHUNKS_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        _require_faiss()
        if not self.chunks:
            return []

        vector = np.asarray([query_embedding], dtype=np.float32)
        if not np.any(vector):
            return []
        vector = _normalize_rows(vector)

        limit = max(1, min(top_k, len(self.chunks)))
        scores, indices = self.index.search(vector, limit)

        results: list[RetrievedChunk] = []
        for rank, (score, raw_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if raw_index < 0:
                continue
            results.append(
                RetrievedChunk(chunk=self.chunks[int(raw_index)], score=float(score), rank=rank)
            )
        return results


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError("faiss-cpu is required to use the FAISS vector store.")
