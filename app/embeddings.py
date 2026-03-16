from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import re

import numpy as np

from .config import Settings, default_embedding_model

try:
    from google import genai
except ImportError:  # pragma: no cover - handled at runtime.
    genai = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime.
    OpenAI = None


class OpenAIEmbedder:
    backend = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 64):
        if OpenAI is None:
            raise RuntimeError("openai is required to use the OpenAI embedder.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedder.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    @property
    def descriptor(self) -> dict[str, object]:
        return {"backend": self.backend, "model": self.model}

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self.batch_size):
            response = self.client.embeddings.create(model=self.model, input=list(batch))
            vectors.extend([list(map(float, item.embedding)) for item in response.data])
        return vectors


class GeminiEmbedder:
    backend = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        batch_size: int = 32,
    ):
        if genai is None:
            raise RuntimeError("google-genai is required to use the Gemini embedder.")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini embedder.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    @property
    def descriptor(self) -> dict[str, object]:
        return {"backend": self.backend, "model": self.model}

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return _extract_gemini_vectors(response)[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self.batch_size):
            response = self.client.models.embed_content(
                model=self.model,
                contents=list(batch),
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            vectors.extend(_extract_gemini_vectors(response))
        return vectors


class HashingEmbedder:
    backend = "hashing"

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    @property
    def descriptor(self) -> dict[str, object]:
        return {"backend": self.backend, "dimension": self.dimension}

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text).tolist() for text in texts]

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


def build_embedder(settings: Settings, backend: str | None = None, model: str | None = None):
    selected_backend = (backend or settings.embedding_backend).lower()
    selected_model = model or settings.embedding_model or default_embedding_model(selected_backend)
    if selected_backend == "gemini":
        return GeminiEmbedder(
            api_key=settings.gemini_api_key or "",
            model=selected_model,
        )
    if selected_backend == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key or "",
            model=selected_model,
        )
    if selected_backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unsupported embedding backend: {selected_backend}")


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _extract_gemini_vectors(response) -> list[list[float]]:
    items = getattr(response, "embeddings", None)
    if items is None:
        single_item = getattr(response, "embedding", None)
        items = [single_item] if single_item is not None else []

    vectors: list[list[float]] = []
    for item in items:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values")
        if values is None:
            raise RuntimeError("Gemini embedding response did not contain vector values.")
        vectors.append([float(value) for value in values])
    return vectors
