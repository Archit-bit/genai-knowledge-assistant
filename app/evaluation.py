from __future__ import annotations

import json
from pathlib import Path
import re
from statistics import mean

from .models import EvaluationExample, RetrievedChunk


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def load_examples(path: Path) -> list[EvaluationExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationExample(
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_sources=list(item.get("expected_sources", [])),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload
    ]


def evaluate_examples(pipeline, examples: list[EvaluationExample]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    answer_f1_scores: list[float] = []
    retrieval_hit_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []

    for example in examples:
        result = pipeline.ask(example.question)
        answer_f1 = token_f1(result.answer, example.expected_answer)
        retrieval_hit = source_hit_rate(example.expected_sources, result.retrieved_chunks)
        precision_at_k = source_precision_at_k(example.expected_sources, result.retrieved_chunks)
        recall_at_k = source_recall_at_k(example.expected_sources, result.retrieved_chunks)
        mrr = reciprocal_rank(example.expected_sources, result.retrieved_chunks)

        answer_f1_scores.append(answer_f1)
        retrieval_hit_scores.append(retrieval_hit)
        precision_scores.append(precision_at_k)
        recall_scores.append(recall_at_k)
        mrr_scores.append(mrr)

        records.append(
            {
                "question": example.question,
                "expected_answer": example.expected_answer,
                "generated_answer": result.answer,
                "expected_sources": example.expected_sources,
                "retrieved_sources": [item.chunk.source for item in result.retrieved_chunks],
                "citations": result.citations,
                "warnings": result.warnings,
                "metrics": {
                    "answer_token_f1": round(answer_f1, 4),
                    "retrieval_hit_rate": round(retrieval_hit, 4),
                    "source_precision_at_k": round(precision_at_k, 4),
                    "source_recall_at_k": round(recall_at_k, 4),
                    "mean_reciprocal_rank": round(mrr, 4),
                },
            }
        )

    summary = {
        "examples": len(examples),
        "average_answer_token_f1": round(_average(answer_f1_scores), 4),
        "average_retrieval_hit_rate": round(_average(retrieval_hit_scores), 4),
        "average_source_precision_at_k": round(_average(precision_scores), 4),
        "average_source_recall_at_k": round(_average(recall_scores), 4),
        "average_mean_reciprocal_rank": round(_average(mrr_scores), 4),
    }
    return {"summary": summary, "records": records}


def save_report(report: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _content_tokens(prediction)
    reference_tokens = _content_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    overlap = len(prediction_tokens & reference_tokens)
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def source_hit_rate(expected_sources: list[str], retrieved_chunks: list[RetrievedChunk]) -> float:
    if not expected_sources:
        return 0.0
    expected = {_normalize_source_name(item) for item in expected_sources}
    retrieved = {_normalize_source_name(item.chunk.source) for item in retrieved_chunks}
    return 1.0 if expected & retrieved else 0.0


def source_precision_at_k(expected_sources: list[str], retrieved_chunks: list[RetrievedChunk]) -> float:
    if not retrieved_chunks:
        return 0.0
    expected = {_normalize_source_name(item) for item in expected_sources}
    matches = sum(1 for item in retrieved_chunks if _normalize_source_name(item.chunk.source) in expected)
    return matches / len(retrieved_chunks)


def source_recall_at_k(expected_sources: list[str], retrieved_chunks: list[RetrievedChunk]) -> float:
    if not expected_sources:
        return 0.0
    expected = {_normalize_source_name(item) for item in expected_sources}
    retrieved = {_normalize_source_name(item.chunk.source) for item in retrieved_chunks}
    return len(expected & retrieved) / len(expected)


def reciprocal_rank(expected_sources: list[str], retrieved_chunks: list[RetrievedChunk]) -> float:
    expected = {_normalize_source_name(item) for item in expected_sources}
    for item in retrieved_chunks:
        if _normalize_source_name(item.chunk.source) in expected:
            return 1.0 / item.rank
    return 0.0


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if token not in STOPWORDS
    }


def _normalize_source_name(value: str) -> str:
    return Path(value).name.lower()


def _average(values: list[float]) -> float:
    return mean(values) if values else 0.0
