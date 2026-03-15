from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re

from .models import RetrievedChunk


PROMPT_INJECTION_PATTERNS = (
    re.compile(r"\bignore\b.{0,20}\b(previous|prior|above)\b", re.IGNORECASE),
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper prompt\b", re.IGNORECASE),
    re.compile(r"\bact as\b", re.IGNORECASE),
    re.compile(r"\boverride\b.{0,20}\binstructions\b", re.IGNORECASE),
)


@dataclass
class QueryValidation:
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class CitationValidation:
    valid_citations: list[str]
    invalid_citations: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return bool(self.valid_citations) and not self.invalid_citations


def validate_question(question: str) -> QueryValidation:
    stripped = question.strip()
    errors: list[str] = []
    warnings: list[str] = []

    if not stripped:
        errors.append("Question cannot be empty.")
    if len(stripped) > 2_000:
        errors.append("Question must be 2000 characters or fewer.")
    if any(pattern.search(stripped) for pattern in PROMPT_INJECTION_PATTERNS):
        warnings.append("Potential prompt-injection content detected; treating the question as plain text.")

    return QueryValidation(is_valid=not errors, warnings=warnings, errors=errors)


def extract_citations(answer: str) -> list[str]:
    return re.findall(r"\[([^\[\]]+?)\]", answer)


def validate_citations(answer: str, retrieved_chunks: Sequence[RetrievedChunk]) -> CitationValidation:
    allowed = {item.chunk.id for item in retrieved_chunks}
    citations = extract_citations(answer)
    valid = [citation for citation in citations if citation in allowed]
    invalid = [citation for citation in citations if citation not in allowed]

    warnings: list[str] = []
    if not citations:
        warnings.append("Model answer did not include any source citations.")
    if invalid:
        warnings.append(f"Model answer referenced unknown citations: {', '.join(invalid)}")

    return CitationValidation(valid_citations=valid, invalid_citations=invalid, warnings=warnings)


def build_grounding_fallback(retrieved_chunks: Sequence[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "I could not find relevant context in the indexed documents."

    top_sources = ", ".join(f"[{item.chunk.id}]" for item in retrieved_chunks[:2])
    return (
        "I could not validate a source-grounded answer from the retrieved context. "
        f"Review the top retrieved chunks: {top_sources}"
    )
