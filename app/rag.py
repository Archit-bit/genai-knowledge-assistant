from __future__ import annotations

from collections.abc import Sequence
import re

from .config import Settings
from .models import AnswerResult, RetrievedChunk
from .prompting import SYSTEM_PROMPT, build_user_prompt
from .validation import build_grounding_fallback, validate_citations, validate_question

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime.
    OpenAI = None


class OpenAIGenerator:
    backend = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", temperature: float = 0.1):
        if OpenAI is None:
            raise RuntimeError("openai is required to use the OpenAI generator.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI generation.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        return response.output_text.strip()


class ExtractiveGenerator:
    backend = "extractive"

    def generate(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        del system_prompt
        del user_prompt
        if not retrieved_chunks:
            return "I could not find relevant context in the indexed documents."

        keywords = set(_important_tokens(question))
        candidates: list[tuple[tuple[int, int], float, str, str]] = []
        for item in retrieved_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", item.chunk.text)
            best_sentence = max(
                (sentence.strip() for sentence in sentences if sentence.strip()),
                key=lambda sentence: _sentence_score(sentence, keywords),
                default=item.chunk.text.strip(),
            )
            candidates.append(
                (
                    _sentence_score(best_sentence, keywords),
                    item.score,
                    best_sentence,
                    item.chunk.id,
                )
            )

        candidates.sort(key=lambda item: (item[0][0], item[0][1], item[1]), reverse=True)
        if not candidates:
            return "I could not find relevant context in the indexed documents."

        max_overlap = candidates[0][0][0]
        if max_overlap <= 0:
            selected = candidates[:1]
        else:
            selected = [item for item in candidates if item[0][0] == max_overlap][:2]

        snippets = [f"{sentence} [{citation}]" for _, _, sentence, citation in selected[:2]]
        return " ".join(snippets)


class RAGPipeline:
    def __init__(
        self,
        embedder,
        vector_store,
        generator,
        top_k: int = 4,
        strict_grounding: bool = True,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k
        self.strict_grounding = strict_grounding

    def ask(self, question: str) -> AnswerResult:
        question_check = validate_question(question)
        if not question_check.is_valid:
            raise ValueError("; ".join(question_check.errors))

        query_embedding = self.embedder.embed_query(question)
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=self.top_k)
        if not retrieved_chunks:
            return AnswerResult(
                question=question,
                answer="I could not find relevant context in the indexed documents.",
                retrieved_chunks=[],
                warnings=question_check.warnings,
                citations=[],
            )

        user_prompt = build_user_prompt(question, retrieved_chunks)
        raw_answer = self.generator.generate(question, retrieved_chunks, SYSTEM_PROMPT, user_prompt).strip()
        citation_check = validate_citations(raw_answer, retrieved_chunks)

        warnings = [*question_check.warnings, *citation_check.warnings]
        answer = raw_answer
        if self.strict_grounding and not citation_check.is_grounded:
            answer = build_grounding_fallback(retrieved_chunks)

        return AnswerResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            warnings=warnings,
            citations=citation_check.valid_citations,
        )


def build_generator(settings: Settings, backend: str | None = None, model: str | None = None):
    selected_backend = (backend or settings.generation_backend).lower()
    if selected_backend == "openai":
        return OpenAIGenerator(
            api_key=settings.openai_api_key or "",
            model=model or settings.generation_model,
            temperature=settings.temperature,
        )
    if selected_backend == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"Unsupported generation backend: {selected_backend}")


def _important_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2]


def _sentence_score(sentence: str, keywords: set[str]) -> tuple[int, int]:
    sentence_tokens = set(_important_tokens(sentence))
    overlap = len(sentence_tokens & keywords)
    return overlap, len(sentence)
