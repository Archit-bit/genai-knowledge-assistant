from __future__ import annotations

from pathlib import Path
import re

from .models import DocumentChunk

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime.
    PdfReader = None


SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def load_and_chunk_documents(
    documents_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")

    files = sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported documents found in {documents_dir}")

    chunks: list[DocumentChunk] = []
    for path in files:
        relative_source = path.relative_to(documents_dir).as_posix()
        text = _read_document(path)
        for chunk_index, payload in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunk_id = f"{relative_source}#chunk-{chunk_index}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source=relative_source,
                    text=payload,
                    metadata={"chunk_index": chunk_index},
                )
            )
    return chunks


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    words = cleaned.split(" ")
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
    return chunks


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is required to ingest PDF files.")
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported document type: {path}")
