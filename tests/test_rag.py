import pytest

faiss = pytest.importorskip("faiss")

from app.embeddings import HashingEmbedder
from app.models import DocumentChunk
from app.rag import ExtractiveGenerator, RAGPipeline
from app.vector_store import FaissVectorStore


def test_pipeline_returns_grounded_answer_with_citations() -> None:
    chunks = [
        DocumentChunk(
            id="remote_work_policy.md#chunk-0",
            source="remote_work_policy.md",
            text="Employees may work remotely up to three days per week after onboarding.",
        ),
        DocumentChunk(
            id="security_policy.md#chunk-0",
            source="security_policy.md",
            text="Suspected security incidents must be reported within one hour of discovery.",
        ),
    ]

    embedder = HashingEmbedder(dimension=256)
    vector_store = FaissVectorStore.from_embeddings(
        chunks,
        embedder.embed_texts([chunk.text for chunk in chunks]),
    )
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=ExtractiveGenerator(),
        top_k=2,
        strict_grounding=True,
    )

    result = pipeline.ask("How many days can employees work remotely each week?")

    assert "three days" in result.answer.lower()
    assert "remote_work_policy.md#chunk-0" in result.answer
    assert "remote_work_policy.md#chunk-0" in result.citations
