from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app.config import Settings
from app.documents import load_and_chunk_documents
from app.embeddings import build_embedder
from app.evaluation import evaluate_examples, load_examples
from app.rag import RAGPipeline, build_generator
from app.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_UPLOAD_TYPES = ["md", "txt", "pdf"]


def main() -> None:
    st.set_page_config(
        page_title="GenAI Knowledge Assistant",
        page_icon=":books:",
        layout="wide",
    )
    settings = Settings.from_env(project_root=PROJECT_ROOT)
    _ensure_session_state()
    _apply_demo_bootstrap(settings)

    st.title("GenAI Knowledge Assistant")
    st.caption(
        "A simple RAG demo that indexes documents, retrieves source-grounded context, "
        "and answers questions with citations."
    )

    controls = _render_sidebar(settings)
    _render_index_overview()
    _render_evaluation_summary()
    _render_chat(settings, controls)
    _render_retrieved_chunks()


def _render_sidebar(settings: Settings) -> dict[str, object]:
    with st.sidebar:
        st.header("Demo Controls")
        corpus_source = st.radio("Corpus Source", ["Sample Documents", "Upload Files"], index=0)
        uploaded_files = []
        if corpus_source == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
            )

        st.divider()
        st.subheader("Retrieval")
        embedding_backend = st.selectbox("Embedding Backend", ["hashing", "openai"], index=0)
        embedding_model = (
            st.text_input("Embedding Model", value=settings.embedding_model)
            if embedding_backend == "openai"
            else None
        )

        st.subheader("Generation")
        generation_backend = st.selectbox("Generation Backend", ["extractive", "openai"], index=0)
        generation_model = (
            st.text_input("Generation Model", value=settings.generation_model)
            if generation_backend == "openai"
            else None
        )

        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=6, value=settings.top_k)
        strict_grounding = st.checkbox("Strict Grounding", value=settings.strict_grounding)

        if (embedding_backend == "openai" or generation_backend == "openai") and not settings.openai_api_key:
            st.warning("OPENAI_API_KEY is not configured. Use offline demo mode or add it to `.env`.")

        if st.button("Build Index", use_container_width=True):
            _handle_build_index(
                settings=settings,
                corpus_source=corpus_source,
                uploaded_files=uploaded_files,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
            )

        artifact = st.session_state.get("index_artifact") or {}
        index_ready = bool(artifact)
        using_sample_docs = artifact.get("corpus_source") == "sample"
        if st.button(
            "Run Sample Evaluation",
            use_container_width=True,
            disabled=not index_ready or not using_sample_docs,
        ):
            _handle_run_evaluation(
                settings=settings,
                generation_backend=generation_backend,
                generation_model=generation_model,
                top_k=top_k,
                strict_grounding=strict_grounding,
            )

        st.caption("Recommended live demo mode: `hashing` + `extractive`.")

    return {
        "generation_backend": generation_backend,
        "generation_model": generation_model,
        "top_k": top_k,
        "strict_grounding": strict_grounding,
    }


def _render_index_overview() -> None:
    artifact = st.session_state.get("index_artifact")
    if not artifact:
        st.info("Build an index from the sidebar to start the demo.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", artifact["document_count"])
    col2.metric("Chunks", artifact["chunk_count"])
    col3.metric("Embedding Backend", artifact["embedding_backend"])
    st.caption(f"Indexed corpus: {artifact['source_label']}")


def _render_evaluation_summary() -> None:
    report = st.session_state.get("evaluation_report")
    if not report:
        return

    st.subheader("Bundled Evaluation")
    summary = report["summary"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Answer Token F1", summary["average_answer_token_f1"])
    col2.metric("Hit Rate", summary["average_retrieval_hit_rate"])
    col3.metric("MRR", summary["average_mean_reciprocal_rank"])
    with st.expander("Evaluation Summary JSON"):
        st.json(summary)


def _render_chat(settings: Settings, controls: dict[str, object]) -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask a question about the indexed documents",
        disabled=st.session_state.get("index_artifact") is None,
    )
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    try:
        with st.spinner("Retrieving context and generating answer..."):
            pipeline = _build_pipeline(settings, controls)
            result = pipeline.ask(prompt)
    except Exception as exc:  # pragma: no cover - UI error surface.
        st.session_state["messages"].append(
            {"role": "assistant", "content": f"Error: `{type(exc).__name__}` - {exc}"}
        )
        st.session_state["last_result"] = None
        st.rerun()
        return

    answer = result.answer
    if result.warnings:
        warning_block = "\n".join(f"- {item}" for item in result.warnings)
        answer = f"{answer}\n\nWarnings:\n{warning_block}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["last_result"] = result
    st.rerun()


def _render_retrieved_chunks() -> None:
    result = st.session_state.get("last_result")
    if not result:
        return

    with st.expander("Retrieved Chunks", expanded=True):
        for item in result.retrieved_chunks:
            st.markdown(
                f"**Rank {item.rank}** | score `{item.score:.4f}` | source `{item.chunk.id}`"
            )
            st.code(item.chunk.text)


def _handle_build_index(
    settings: Settings,
    corpus_source: str,
    uploaded_files,
    embedding_backend: str,
    embedding_model: str | None,
) -> None:
    try:
        documents_dir, source_label, source_type, document_count = _prepare_documents(
            settings=settings,
            corpus_source=corpus_source,
            uploaded_files=uploaded_files,
        )
        configured = settings.with_overrides(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        chunks = load_and_chunk_documents(
            documents_dir=documents_dir,
            chunk_size=configured.chunk_size,
            chunk_overlap=configured.chunk_overlap,
        )
        embedder = build_embedder(
            configured,
            backend=configured.embedding_backend,
            model=configured.embedding_model,
        )
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_store = FaissVectorStore.from_embeddings(chunks, embeddings)

        index_dir = Path(tempfile.mkdtemp(prefix="rag_streamlit_index_"))
        vector_store.save(index_dir)

        st.session_state["index_artifact"] = {
            "index_dir": str(index_dir),
            "embedding_backend": embedder.descriptor["backend"],
            "embedding_model": embedder.descriptor.get("model"),
            "source_label": source_label,
            "corpus_source": source_type,
            "document_count": document_count,
            "chunk_count": len(chunks),
        }
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"Index ready for **{source_label}**. Ask a question to inspect retrieval, "
                    "citations, and grounding behavior."
                ),
            }
        ]
        st.session_state["last_result"] = None
        st.session_state["evaluation_report"] = None
        st.success("Index built successfully.")
    except Exception as exc:  # pragma: no cover - UI error surface.
        st.error(f"{type(exc).__name__}: {exc}")


def _handle_run_evaluation(
    settings: Settings,
    generation_backend: str,
    generation_model: str | None,
    top_k: int,
    strict_grounding: bool,
) -> None:
    try:
        pipeline = _build_pipeline(
            settings,
            {
                "generation_backend": generation_backend,
                "generation_model": generation_model,
                "top_k": top_k,
                "strict_grounding": strict_grounding,
            },
        )
        report = evaluate_examples(
            pipeline,
            load_examples(settings.evaluation_examples_path),
        )
        st.session_state["evaluation_report"] = report
        st.success("Evaluation completed.")
    except Exception as exc:  # pragma: no cover - UI error surface.
        st.error(f"{type(exc).__name__}: {exc}")


def _build_pipeline(settings: Settings, controls: dict[str, object]) -> RAGPipeline:
    artifact = st.session_state.get("index_artifact")
    if not artifact:
        raise RuntimeError("Build an index before asking questions.")

    configured = settings.with_overrides(
        embedding_backend=artifact["embedding_backend"],
        embedding_model=artifact["embedding_model"],
        generation_backend=controls["generation_backend"],
        generation_model=controls["generation_model"],
        top_k=controls["top_k"],
        strict_grounding=controls["strict_grounding"],
    )
    vector_store = FaissVectorStore.load(Path(artifact["index_dir"]))
    embedder = build_embedder(
        configured,
        backend=artifact["embedding_backend"],
        model=artifact["embedding_model"],
    )
    generator = build_generator(
        configured,
        backend=controls["generation_backend"],
        model=controls["generation_model"],
    )
    return RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=configured.top_k,
        strict_grounding=configured.strict_grounding,
    )


def _prepare_documents(
    settings: Settings,
    corpus_source: str,
    uploaded_files,
) -> tuple[Path, str, str, int]:
    if corpus_source == "Sample Documents":
        files = sorted(path for path in settings.documents_dir.rglob("*") if path.is_file())
        return settings.documents_dir, "Bundled sample documents", "sample", len(files)

    if not uploaded_files:
        raise ValueError("Upload at least one supported document before building the index.")

    upload_dir = Path(tempfile.mkdtemp(prefix="rag_streamlit_docs_"))
    for uploaded_file in uploaded_files:
        destination = upload_dir / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getbuffer())

    names = ", ".join(Path(file.name).name for file in uploaded_files)
    return upload_dir, names, "uploaded", len(uploaded_files)


def _ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Build an index from the sidebar. For a stable showcase, use "
                    "**hashing** embeddings with **extractive** generation."
                ),
            }
        ]
    st.session_state.setdefault("index_artifact", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("evaluation_report", None)
    st.session_state.setdefault("demo_bootstrap_mode", None)


def _apply_demo_bootstrap(settings: Settings) -> None:
    demo_mode = st.query_params.get("demo")
    if not demo_mode:
        return

    if st.session_state.get("demo_bootstrap_mode") == demo_mode:
        return

    if demo_mode == "showcase":
        _bootstrap_demo_session(settings, include_evaluation=False)
    elif demo_mode == "evaluation":
        _bootstrap_demo_session(settings, include_evaluation=True)
    else:
        return

    st.session_state["demo_bootstrap_mode"] = demo_mode


def _bootstrap_demo_session(settings: Settings, include_evaluation: bool) -> None:
    configured = settings.with_overrides(
        embedding_backend="hashing",
        generation_backend="extractive",
        strict_grounding=True,
        top_k=4,
    )
    chunks = load_and_chunk_documents(
        documents_dir=settings.documents_dir,
        chunk_size=configured.chunk_size,
        chunk_overlap=configured.chunk_overlap,
    )
    embedder = build_embedder(configured, backend="hashing", model=None)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    vector_store = FaissVectorStore.from_embeddings(chunks, embeddings)

    index_dir = Path(tempfile.mkdtemp(prefix="rag_streamlit_demo_index_"))
    vector_store.save(index_dir)
    st.session_state["index_artifact"] = {
        "index_dir": str(index_dir),
        "embedding_backend": "hashing",
        "embedding_model": None,
        "source_label": "Bundled sample documents",
        "corpus_source": "sample",
        "document_count": len(sorted(path for path in settings.documents_dir.rglob('*') if path.is_file())),
        "chunk_count": len(chunks),
    }

    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=build_generator(configured, backend="extractive", model=None),
        top_k=configured.top_k,
        strict_grounding=configured.strict_grounding,
    )
    question = "How quickly must a security incident be reported?"
    result = pipeline.ask(question)
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Index ready for **Bundled sample documents**. Ask a question to inspect retrieval, citations, and grounding behavior.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": result.answer},
    ]
    st.session_state["last_result"] = result

    if include_evaluation:
        st.session_state["evaluation_report"] = evaluate_examples(
            pipeline,
            load_examples(settings.evaluation_examples_path),
        )
    else:
        st.session_state["evaluation_report"] = None


if __name__ == "__main__":
    main()
