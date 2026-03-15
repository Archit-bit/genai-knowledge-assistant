from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os

from dotenv import load_dotenv


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    openai_api_key: str | None
    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    generation_backend: str = "openai"
    generation_model: str = "gpt-4.1-mini"
    chunk_size: int = 220
    chunk_overlap: int = 40
    top_k: int = 4
    strict_grounding: bool = True
    temperature: float = 0.1

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = project_root or Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
        return cls(
            project_root=root,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "openai"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            generation_backend=os.getenv("GENERATION_BACKEND", "openai"),
            generation_model=os.getenv("GENERATION_MODEL", "gpt-4.1-mini"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "220")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
            top_k=int(os.getenv("TOP_K", "4")),
            strict_grounding=_read_bool("STRICT_GROUNDING", True),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        updates = {key: value for key, value in kwargs.items() if value is not None}
        return replace(self, **updates)

    @property
    def documents_dir(self) -> Path:
        return self.project_root / "data" / "sample_docs"

    @property
    def index_dir(self) -> Path:
        return self.project_root / "data" / "index"

    @property
    def evaluation_examples_path(self) -> Path:
        return self.project_root / "data" / "evaluation_examples.json"
