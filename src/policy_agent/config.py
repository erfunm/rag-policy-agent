"""One Settings object, loaded once: YAML file first, environment variables override.

Env vars use the ``POLICY_AGENT_`` prefix and ``__`` for nesting, e.g.
``POLICY_AGENT_LLM__PROVIDER=anthropic``. ``POLICY_AGENT_CONFIG`` points at a different YAML file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("configs/settings.yaml")

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
Provider = Literal["fake", "anthropic", "openai"]


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    policies_dir: Path = Path("data/policies")
    cases_path: Path = Path("data/cases.json")
    cache_dir: Path = Path(".cache")
    mode: RetrievalMode = "hybrid_rerank"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    candidate_pool: int = 20
    top_k: int = 5
    max_chunk_words: int = 180
    # Evidence floor. With reranking on, the best cross-encoder score must reach min_rerank_score;
    # otherwise the best dense cosine similarity must reach min_relevance. Tuned on the golden set.
    min_relevance: float = 0.7
    min_rerank_score: float = -3.0


class LLMSettings(BaseModel):
    provider: Provider = "fake"
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str | None = None
    max_tokens: int = 4096
    effort: Literal["low", "medium", "high"] | None = "medium"
    # Server-side refusal fallback (Claude API beta); disable for proxies that reject beta params.
    anthropic_fallbacks: bool = True
    timeout_s: float = 60.0
    # Optional LLM-as-judge faithfulness grader used by `policy-agent eval --judge`.
    judge_model: str = "claude-haiku-5-5"
    # USD per million tokens: (input, output). Unknown models are costed at 0 and flagged in the report.
    prices: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "claude-sonnet-5-5": (2.0, 10.0),
            "claude-opus-5-5": (4.0, 20.0),
            "claude-haiku-5-5": (0.10, 0.50),
            "fake-llm": (0.0, 0.0),
        }
    )


class AgentSettings(BaseModel):
    max_steps: int = 6
    tool_timeout_s: float = 10.0
    escalation_message: str = "Insufficient evidence found. Please escalate to Compliance."


class GuardrailSettings(BaseModel):
    redact_input_pii: bool = True
    # Remove sentences that look like instructions from retrieved text before the model sees them.
    # Turn off to measure the model's own injection resistance (spotlighting only).
    strip_injections: bool = True


class ObservabilitySettings(BaseModel):
    audit_log_path: Path = Path("logs/audit.jsonl")
    tracing_enabled: bool = False
    service_name: str = "policy-agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLICY_AGENT_", env_nested_delimiter="__", extra="ignore")

    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get("POLICY_AGENT_CONFIG", DEFAULT_CONFIG_PATH))
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
