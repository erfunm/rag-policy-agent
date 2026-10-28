"""Provider-agnostic LLM layer. ``make_llm`` picks the adapter from settings."""

from __future__ import annotations

from policy_agent.config import LLMSettings, Provider
from policy_agent.llm.base import LLMClient


def make_llm(settings: LLMSettings, provider: Provider | None = None) -> LLMClient:
    provider = provider or settings.provider
    if provider == "anthropic":
        from policy_agent.llm.anthropic import AnthropicClient

        return AnthropicClient(settings)
    if provider == "openai":
        from policy_agent.llm.openai_compat import OpenAICompatClient

        return OpenAICompatClient(settings)
    from policy_agent.llm.fake import FakeLLM

    return FakeLLM()
