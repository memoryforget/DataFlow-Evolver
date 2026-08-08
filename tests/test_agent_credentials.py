from __future__ import annotations

from dataflow_evolver.llm.agent_credentials import resolve_agent_endpoint
from dataflow_evolver.utils.config import Config


def test_generic_endpoint_is_shared_without_changing_backend_protocol(monkeypatch):
    monkeypatch.setenv("DF_AGENT_API_KEY", "generic-secret")
    cfg = Config(
        {
            "api_key_env": "DF_AGENT_API_KEY",
            "base_url": "https://gateway.example/v1",
        }
    )

    endpoint = resolve_agent_endpoint(cfg, "codex", provider="openai")
    assert endpoint.api_key_env == "DF_AGENT_API_KEY"
    assert endpoint.api_key == "generic-secret"
    assert endpoint.base_url == "https://gateway.example/v1"


def test_backend_override_wins_over_generic_value(monkeypatch):
    monkeypatch.setenv("CODEX_SPECIAL_KEY", "codex-secret")
    cfg = Config(
        {
            "api_key_env": "DF_AGENT_API_KEY",
            "base_url": "https://generic.example/",
            "codex": {
                "api_key_env": "CODEX_SPECIAL_KEY",
                "base_url": "https://codex.example/v1",
            },
        }
    )

    endpoint = resolve_agent_endpoint(cfg, "codex", provider="openai")
    assert endpoint.api_key_env == "CODEX_SPECIAL_KEY"
    assert endpoint.api_key == "codex-secret"
    assert endpoint.base_url == "https://codex.example/v1"


def test_legacy_provider_credentials_remain_fallbacks(monkeypatch):
    monkeypatch.delenv("DF_AGENT_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example/")

    endpoint = resolve_agent_endpoint(Config({}), "claude")
    assert endpoint.api_key_env == "ANTHROPIC_API_KEY"
    assert endpoint.api_key == "legacy-secret"
    assert endpoint.base_url == "https://anthropic.example/"
