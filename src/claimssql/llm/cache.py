"""Disk cache for LLM calls: one JSON file per call, so re-runs cost zero calls.

Key = sha256 of everything that can change the answer: provider, model, prompt
version, messages, temperature, seed, plus the model options and JSON mode
(owner decision, P4: changing a setting must never return answers produced
under the old setting).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claimssql.llm.base import DegenerateOutput, LLMClient, LLMResponse, Message


def cache_key(
    provider: str,
    model: str,
    prompt_version: str,
    messages: list[Message],
    temperature: float,
    seed: int,
    options: dict[str, Any],
    json_mode: bool,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "messages": messages,
        "temperature": float(temperature),  # 0 and 0.0 must give the same key
        "seed": seed,
        "options": options,
        "json_mode": json_mode,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Two-character sub-folders keep any one folder from holding thousands of files.
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if path.exists():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entry = None  # unreadable file: treat as a miss and overwrite it
            if entry is not None:
                self.hits += 1
                return entry
        self.misses += 1
        return None

    def put(self, key: str, entry: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)  # atomic: an interrupted run cannot leave half a file


class CachedClient:
    """Wraps a real client. Same interface; answers from disk when it can."""

    def __init__(self, inner: LLMClient, cache: DiskCache, prompt_version: str) -> None:
        self.inner = inner
        self.cache = cache
        self.prompt_version = prompt_version
        self.provider = inner.provider
        self.model = inner.model
        self.options = inner.options

    def chat(
        self,
        messages: list[Message],
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> LLMResponse:
        key = cache_key(
            self.provider, self.model, self.prompt_version, messages, temperature, seed,
            self.options, json_mode,
        )  # fmt: skip
        if (entry := self.cache.get(key)) is not None:
            if "degenerate" in entry:  # a stored "model looped" result: give it back as such
                raise DegenerateOutput(entry["degenerate"]["message"],
                                       latency_ms=entry["degenerate"]["latency_ms"], cached=True)
            # latency_ms is the original call's latency, so timing statistics
            # stay meaningful on a cached re-run.
            return LLMResponse(**entry["response"], cached=True)

        record = {
            "key": key,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "request": {
                "provider": self.provider,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "temperature": temperature,
                "seed": seed,
                "options": self.options,
                "json_mode": json_mode,
                "messages": messages,
            },
        }
        try:
            response = self.inner.chat(messages, temperature, seed, json_mode)
        except DegenerateOutput as exc:
            # Deterministic for this exact call, so it is stored: a rerun gets the
            # same result at once instead of looping again. Every other error
            # (network, timeout, rate limit) is NOT cached and is retried next run.
            record["degenerate"] = {"message": exc.message, "latency_ms": exc.latency_ms}
            self.cache.put(key, record)
            raise
        stored = asdict(response)
        stored.pop("cached")
        self.cache.put(key, {**record, "response": stored})
        return response
