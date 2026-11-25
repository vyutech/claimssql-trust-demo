"""Client for a local Ollama server: POST {OLLAMA_HOST}/api/chat."""

from __future__ import annotations

import time
from typing import Any

import httpx

from claimssql.llm.base import DegenerateOutput, LLMError, LLMResponse, Message, check_options

ALLOWED_OPTIONS = {"think", "num_ctx", "max_tokens"}
# Ollama stops a reply that keeps repeating the same tokens and answers HTTP 500
# with this text (seen on Kaggle: qwen on H-022, sample seed 3).
_DEGENERATE_MARKERS = ("token repeat limit", "prediction aborted")


def is_degenerate(status_code: int, body: str) -> bool:
    """True for the 'model looped' error; False for every other server error."""
    text = body.lower()
    return status_code == 500 and any(marker in text for marker in _DEGENERATE_MARKERS)
# CPU-only generation is slow (a few tokens per second), so allow a long wait.
DEFAULT_TIMEOUT_S = 1800.0


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        host: str,
        options: dict[str, Any] | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = dict(options or {})
        check_options(self.provider, self.options, ALLOWED_OPTIONS)
        self._http = http or httpx.Client(timeout=DEFAULT_TIMEOUT_S)

    def build_body(
        self, messages: list[Message], temperature: float, seed: int, json_mode: bool
    ) -> dict[str, Any]:
        run_options: dict[str, Any] = {"temperature": temperature, "seed": seed}
        if "num_ctx" in self.options:
            # Explicit, because Ollama's default (4096) is close to the prompt size
            # and a too-small window silently cuts the start of the prompt.
            run_options["num_ctx"] = self.options["num_ctx"]
        if "max_tokens" in self.options:
            run_options["num_predict"] = self.options["max_tokens"]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # false: thinking models otherwise reason for minutes on CPU before
            # answering. Some models have no "off" and take a level ("low") instead.
            "think": self.options.get("think", False),
            "options": run_options,
        }
        if json_mode:
            body["format"] = "json"
        return body

    def chat(
        self,
        messages: list[Message],
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> LLMResponse:
        body = self.build_body(messages, temperature, seed, json_mode)
        start = time.perf_counter()
        try:
            response = self._http.post(f"{self.host}/api/chat", json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            if is_degenerate(exc.response.status_code, detail):
                raise DegenerateOutput(
                    f"ollama HTTP {exc.response.status_code}: {detail}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                ) from exc
            raise LLMError(f"ollama HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {type(exc).__name__}: {exc}") from exc
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )
