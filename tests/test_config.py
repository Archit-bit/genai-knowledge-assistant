from pathlib import Path

from app.config import Settings


def test_settings_accepts_google_api_key_alias(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings.from_env(project_root=Path.cwd())

    assert settings.gemini_api_key == "test-gemini-key"


def test_settings_default_to_low_cost_backends(monkeypatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "EMBEDDING_BACKEND",
        "EMBEDDING_MODEL",
        "GENERATION_BACKEND",
        "GENERATION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(project_root=Path.cwd())

    assert settings.embedding_backend == "hashing"
    assert settings.generation_backend == "extractive"
