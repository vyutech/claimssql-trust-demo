"""Client for Groq's OpenAI-compatible API, with polite retrying.

The free tier limits tokens per minute, so 429 ("slow down") is a normal
answer here, not an error: we wait as long as Groq says (retry-after) and try
again. Server errors (5xx) and network errors are retried with exponential
backoff plus random jitter, so repeated runs do not hammer the API in step.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from claimssql.llm.base import LLMError, LLMResponse, Message, RateLimitStop, check_options

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ALLOWED_OPTIONS = {"reasoning_effort", "max_tokens"}
MAX_RETRIES = 8
BASE_DELAY_S = 2.0
MAX_DELAY_S = 120.0
# A per-minute limit clears within a minute. A longer wait means a daily limit:
# stop instead of sleeping for hours.
MAX_RATE_LIMIT_WAIT_S = 90.0


def retry_delay(attempt: int, retry_after: str | None, rng: random.Random) -> float:
    """Seconds to wait before retry number `attempt` (0-based)."""
    jitter = rng.uniform(0, 1)
    if retry_after:
        try:
            # The server knows when the limit resets; add jitter so we are not early.
            return max(float(retry_after), 0.0) + jitter
        except ValueError:
            pass  # not a number of seconds: fall back to our own backoff
    return min(BASE_DELAY_S * 2**attempt, MAX_DELAY_S) + jitter


def _too_long(retry_after: str | None) -> bool:
    try:
        return retry_after is not None and float(retry_after) > MAX_RATE_LIMIT_WAIT_S
    except ValueError:
        return False


class GroqClient:
    provider = "groq"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        options: dict[str, Any] | None = None,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set: put it in .env locally, or add it as a "
                           "secret in the Hugging Face Space settings")
        self.model = model
        self.options = dict(options or {})
        check_options(self.provider, self.options, ALLOWED_OPTIONS)
        self._api_key = api_key
        self._http = http or httpx.Client(timeout=120.0)
        self._sleep = sleep
        self._rng = rng or random.Random()

    def build_body(
        self, messages: list[Message], temperature: float, seed: int, json_mode: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if "reasoning_effort" in self.options:
            body["reasoning_effort"] = self.options["reasoning_effort"]
        if "max_tokens" in self.options:
            body["max_completion_tokens"] = self.options["max_tokens"]
        return body

    def chat(
        self,
        messages: list[Message],
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> LLMResponse:
        body = self.build_body(messages, temperature, seed, json_mode)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_problem = "no attempt made"
        for attempt in range(MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                response = self._http.post(GROQ_URL, headers=headers, json=body)
            except httpx.HTTPError as exc:  # network trouble: worth retrying
                last_problem = f"{type(exc).__name__}: {exc}"
                retry_after = None
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    return self._to_response(response.json(), latency_ms)
                if (rejected := self._rejected_json(response, latency_ms)) is not None:
                    return rejected
                last_problem = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code != 429 and response.status_code < 500:
                    break  # other 4xx: our request is wrong, retrying cannot help
                retry_after = response.headers.get("retry-after")
                if response.status_code == 429 and _too_long(retry_after):
                    raise RateLimitStop(
                        f"groq rate limit: asked to wait {retry_after} s "
                        f"(more than {MAX_RATE_LIMIT_WAIT_S:g} s) - {response.text[:200]}"
                    )
            if attempt < MAX_RETRIES:
                self._sleep(retry_delay(attempt, retry_after, self._rng))
        raise LLMError(f"groq call failed: {last_problem}")

    @staticmethod
    def _to_response(data: dict[str, Any], latency_ms: float) -> LLMResponse:
        usage = data.get("usage") or {}
        return LLMResponse(
            text=data["choices"][0]["message"].get("content") or "",
            latency_ms=latency_ms,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    @staticmethod
    def _rejected_json(response: httpx.Response, latency_ms: float) -> LLMResponse | None:
        """In JSON mode Groq answers HTTP 400 when the MODEL wrote invalid JSON.

        That is a malformed model output, not a failed call, so hand the text
        back and let the generator count it (retry, then PARSE_ERROR).
        """
        if response.status_code != 400:
            return None
        try:
            error = response.json().get("error", {})
        except ValueError:
            return None
        if error.get("code") != "json_validate_failed":
            return None
        return LLMResponse(
            text=error.get("failed_generation") or "",
            latency_ms=latency_ms,
            prompt_tokens=0,  # Groq reports no usage for a rejected generation
            completion_tokens=0,
        )
