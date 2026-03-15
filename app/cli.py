from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .documents import load_and_chunk_documents
from .embeddings import build_embedder
from .evaluation import evaluate_examples, load_examples, save_report
from .rag import RAGPipeline, build_generator
from .vector_store import FaissVectorStore


MANIFEST_FILENAME = "manifest.json"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(project_root=project_root)

    if args.command == "index":
        return run_index(args, settings)
    if args.command == "ask":
        return run_ask(args, settings)
    if args.command == "evaluate":
        return run_evaluate(args, settings)
    raise ValueError(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenAI Knowledge Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build a FAISS index from local documents.")
    index_parser.add_argument("--documents-dir", type=Path, default=None)
    index_parser.add_argument("--index-dir", type=Path, default=None)
    index_parser.add_argument("--chunk-size", type=int, default=None)
    index_parser.add_argument("--chunk-overlap", type=int, default=None)
    index_parser.add_argument("--embedding-backend", choices=["openai", "hashing"], default=None)
    index_parser.add_argument("--embedding-model", default=None)

    ask_parser = subparsers.add_parser("ask", help="Query the indexed knowledge base.")
    ask_parser.add_argument("--index-dir", type=Path, default=None)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.add_argument("--embedding-backend", choices=["openai", "hashing"], default=None)
    ask_parser.add_argument("--embedding-model", default=None)
    ask_parser.add_argument("--generation-backend", choices=["openai", "extractive"], default=None)
    ask_parser.add_argument("--generation-model", default=None)
    ask_parser.add_argument("--strict-grounding", choices=["true", "false"], default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run retrieval and answer evaluation.")
    evaluate_parser.add_argument("--index-dir", type=Path, default=None)
    evaluate_parser.add_argument("--examples", type=Path, default=None)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.add_argument("--top-k", type=int, default=None)
    evaluate_parser.add_argument("--embedding-backend", choices=["openai", "hashing"], default=None)
    evaluate_parser.add_argument("--embedding-model", default=None)
    evaluate_parser.add_argument("--generation-backend", choices=["openai", "extractive"], default=None)
    evaluate_parser.add_argument("--generation-model", default=None)
    evaluate_parser.add_argument("--strict-grounding", choices=["true", "false"], default=None)

    return parser


def run_index(args: argparse.Namespace, settings: Settings) -> int:
    documents_dir = _resolve_path(args.documents_dir, settings.documents_dir)
    index_dir = _resolve_path(args.index_dir, settings.index_dir)
    configured = settings.with_overrides(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )

    chunks = load_and_chunk_documents(
        documents_dir=documents_dir,
        chunk_size=configured.chunk_size,
        chunk_overlap=configured.chunk_overlap,
    )
    embedder = build_embedder(configured, backend=configured.embedding_backend, model=configured.embedding_model)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    vector_store = FaissVectorStore.from_embeddings(chunks, embeddings)
    vector_store.save(index_dir)

    manifest = {
        "documents_dir": str(documents_dir),
        "embedding_backend": embedder.descriptor["backend"],
        "embedding_model": embedder.descriptor.get("model"),
        "chunk_size": configured.chunk_size,
        "chunk_overlap": configured.chunk_overlap,
        "chunk_count": len(chunks),
    }
    _write_manifest(index_dir, manifest)

    print(f"Indexed {len(chunks)} chunks from {documents_dir} into {index_dir}")
    return 0


def run_ask(args: argparse.Namespace, settings: Settings) -> int:
    index_dir = _resolve_path(args.index_dir, settings.index_dir)
    manifest = _read_manifest(index_dir)

    embedding_backend = args.embedding_backend or manifest.get("embedding_backend") or settings.embedding_backend
    embedding_model = args.embedding_model or manifest.get("embedding_model") or settings.embedding_model
    strict_grounding = _coerce_bool(args.strict_grounding, settings.strict_grounding)
    configured = settings.with_overrides(
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        generation_backend=args.generation_backend,
        generation_model=args.generation_model,
        strict_grounding=strict_grounding,
        top_k=args.top_k,
    )

    vector_store = FaissVectorStore.load(index_dir)
    embedder = build_embedder(configured, backend=embedding_backend, model=embedding_model)
    generator = build_generator(
        configured,
        backend=configured.generation_backend,
        model=configured.generation_model,
    )
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=configured.top_k,
        strict_grounding=configured.strict_grounding,
    )

    result = pipeline.ask(args.question)
    print(result.answer)
    if result.citations:
        print(f"\nCitations: {', '.join(result.citations)}")
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    print("\nRetrieved Chunks:")
    for item in result.retrieved_chunks:
        print(f"- rank={item.rank} score={item.score:.4f} source={item.chunk.id}")
    return 0


def run_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    index_dir = _resolve_path(args.index_dir, settings.index_dir)
    examples_path = _resolve_path(args.examples, settings.evaluation_examples_path)
    output_path = _resolve_path(args.output, index_dir / "evaluation_report.json")
    manifest = _read_manifest(index_dir)

    embedding_backend = args.embedding_backend or manifest.get("embedding_backend") or settings.embedding_backend
    embedding_model = args.embedding_model or manifest.get("embedding_model") or settings.embedding_model
    strict_grounding = _coerce_bool(args.strict_grounding, settings.strict_grounding)
    configured = settings.with_overrides(
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        generation_backend=args.generation_backend,
        generation_model=args.generation_model,
        strict_grounding=strict_grounding,
        top_k=args.top_k,
    )

    vector_store = FaissVectorStore.load(index_dir)
    embedder = build_embedder(configured, backend=embedding_backend, model=embedding_model)
    generator = build_generator(
        configured,
        backend=configured.generation_backend,
        model=configured.generation_model,
    )
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=configured.top_k,
        strict_grounding=configured.strict_grounding,
    )

    examples = load_examples(examples_path)
    report = evaluate_examples(pipeline, examples)
    save_report(report, output_path)

    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved report to {output_path}")
    return 0


def _resolve_path(value: Path | None, default: Path) -> Path:
    target = value or default
    if target.is_absolute():
        return target
    return Path(__file__).resolve().parents[1] / target


def _read_manifest(index_dir: Path) -> dict[str, object]:
    manifest_path = index_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict[str, object]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() == "true"
