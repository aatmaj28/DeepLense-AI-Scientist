"""Env-driven, local-first model configuration (no network)."""

from __future__ import annotations

import pytest
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel

from dlens.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    GROQ_BASE_URL,
    ModelSettings,
    build_llm,
    build_model_from_env,
)


def _clear(monkeypatch):
    for var in (
        "DLENS_MODEL_PROVIDER", "DLENS_MODEL", "DLENS_MODEL_PORT",
        "DLENS_MODEL_BASE_URL", "DLENS_MODEL_API_KEY", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_are_local_ollama_qwen(monkeypatch):
    _clear(monkeypatch)
    s = ModelSettings.from_env()
    assert s.provider == DEFAULT_PROVIDER == "ollama"
    assert s.model == DEFAULT_MODEL == "qwen3:8b"
    model = build_model_from_env()
    # OllamaModel subclasses OpenAIChatModel; tag is qwen3:8b.
    assert "qwen3:8b" in str(getattr(model, "model_name", ""))


def test_env_overrides_provider_and_model(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("DLENS_MODEL", "Qwen/Qwen3-8B")
    s = ModelSettings.from_env()
    assert s.provider == "vllm" and s.model == "Qwen/Qwen3-8B"
    build_model_from_env()  # constructs without error / network


def test_openai_compatible_requires_base_url(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DLENS_MODEL_API_KEY", "shared-key")
    with pytest.raises(ValueError, match="DLENS_MODEL_BASE_URL"):
        build_model_from_env()
    # With a base URL it builds (shared-key / gateway path).
    monkeypatch.setenv("DLENS_MODEL_BASE_URL", "https://gateway.example/v1")
    build_model_from_env()


def test_unknown_provider_raises(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "nonsense")
    with pytest.raises(ValueError, match="Unknown DLENS_MODEL_PROVIDER"):
        build_model_from_env()


def test_describe_is_key_safe(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DLENS_MODEL_API_KEY", "super-secret")
    monkeypatch.setenv("DLENS_MODEL_BASE_URL", "https://gateway.example/v1")
    desc = ModelSettings.from_env().describe()
    assert "super-secret" not in desc
    assert "api_key=set" in desc


def test_openai_provider_defaults_to_gpt_4o_mini(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DLENS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = ModelSettings.from_env()
    assert s.provider == "openai"
    assert s.model == "gpt-4o-mini"  # provider-specific default, not the ollama tag
    assert s.api_key == "sk-test"
    build_model_from_env()  # constructs without a network call


def test_build_llm_routes_gemini_ids_to_google(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    model = build_llm("gemini-2.5-flash")
    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-2.5-flash"  # constructs without a network call


def test_build_llm_keeps_other_ids_on_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = build_llm("gpt-5.2")
    assert isinstance(model, OpenAIChatModel) and not isinstance(model, GoogleModel)
    assert model.model_name == "gpt-5.2"
    assert "api.openai.com" in model.base_url  # the OpenAI path is untouched


def test_build_llm_routes_groq_ids_to_groq_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    for mid, served in (("groq:openai/gpt-oss-120b", "openai/gpt-oss-120b"),
                        ("groq/compound", "groq/compound")):
        model = build_llm(mid)
        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == served  # the groq: prefix is not sent to Groq
        assert model.base_url.startswith(GROQ_BASE_URL)


def test_groq_without_its_key_refuses_rather_than_sending_the_openai_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_llm("groq:openai/gpt-oss-120b")
