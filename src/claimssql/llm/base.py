"""What every LLM client looks like to the rest of the code.

The generator only knows this interface, so the real clients (Ollama, Groq),
the caching wrapper and the fake client used in tests are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False  # True when the answer came from the disk cache


class LLMError(Exception):
    """A call failed for good (after retries). Never contains the API key."""


class RateLimitStop(LLMError):
    """The provider asks for a long wait (e.g. a daily limit). Not worth sleeping
    through: stop the run cleanly and resume later - the cache keeps the progress."""


class CacheMiss(LLMError):
    """A cache-only client was asked for a call that is not in the cache."""


class DegenerateOutput(LLMError):
    """The model got stuck repeating itself and the server aborted the reply.

    Unlike a network error this is a property of the model's output: the same
    call (same prompt, temperature and seed) fails the same way again. So it is
    a result, not a failure to retry: the cache stores it, and the generator
    records the generation as DEGENERATE_OUTPUT.
    """

    def __init__(self, message: str, latency_ms: float = 0.0, cached: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.latency_ms = latency_ms
        self.cached = cached


class LLMClient(Protocol):
    provider: str
    model: str
    options: dict[str, Any]

    def chat(
        self,
        messages: list[Message],
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> LLMResponse: ...


def check_options(provider: str, options: dict[str, Any], allowed: set[str]) -> None:
    """Fail early on a misspelt option in models.yaml instead of silently ignoring it."""
    if unknown := sorted(set(options) - allowed):
        raise ValueError(
            f"unknown {provider} option(s) in models.yaml: {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )
