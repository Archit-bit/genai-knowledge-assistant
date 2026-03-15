from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    id: str
    source: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DocumentChunk":
        return cls(
            id=str(data["id"]),
            source=str(data["source"]),
            text=str(data["text"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rank: int


@dataclass
class AnswerResult:
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    warnings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class EvaluationExample:
    question: str
    expected_answer: str
    expected_sources: list[str]
    metadata: dict[str, object] = field(default_factory=dict)
