"""Build the client for a model in configs/models.yaml - always behind the cache.

This is the only place real clients are created, so no caller can skip the
cache by accident.
"""

from __future__ import annotations

from pathlib import Path

from claimssql.config import CACHE_DIR, EnvSettings, ModelConfig, load_env
from claimssql.llm.base import CacheMiss, LLMClient, LLMResponse, Message
from claimssql.llm.cache import CachedClient, DiskCache
from claimssql.llm.groq_client import GroqClient
from claimssql.llm.ollama_client import OllamaClient


class CacheOnlyClient:
    """Stands in for the real client when no new calls are allowed (the verify
    stage): it looks like the model for the cache key, but any call that reaches
    it means the answer was not in the cache."""

    def __init__(self, model: ModelConfig) -> None:
        self.provider = model.provider
        self.model = model.model
        self.options = dict(model.options)

    def chat(
        self,
        messages: list[Message],
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> LLMResponse:
        raise CacheMiss(
            f"no cached answer for {self.model} (temperature {temperature:g}, seed {seed})"
        )


def make_client(
    model: ModelConfig,
    prompt_version: str,
    cache_dir: Path | None = None,
    env: EnvSettings | None = None,
    cache_only: bool = False,
) -> CachedClient:
    inner: LLMClient
    if cache_only:
        # No network, no API key: answers can only come from the cache.
        return CachedClient(CacheOnlyClient(model), DiskCache(cache_dir or CACHE_DIR),
                            prompt_version)
    env = env or load_env()
    if model.provider == "ollama":
        inner = OllamaClient(model.model, env.ollama_host, model.options)
    else:
        inner = GroqClient(model.model, env.groq_api_key, model.options)
    return CachedClient(inner, DiskCache(cache_dir or CACHE_DIR), prompt_version)
