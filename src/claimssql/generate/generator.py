"""Ask one model one question, the 7 ways the pipeline needs.

    1 x forced  greedy   (temperature 0)            -> config C0
    1 x abstain greedy   (temperature 0)            -> configs C1-C4
    k x abstain sampled  (temperature 0.7, seeds 1..k) -> V3 self-consistency

Each reply is cleaned (<think> blocks and code fences removed) and parsed as
{"answerable", "sql", "confidence", "reason"}. A reply that cannot be parsed
gets ONE retry; if that fails too, the generation is marked PARSE_ERROR.
A reply the server aborted because the model looped is marked
DEGENERATE_OUTPUT straight away (no retry).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from claimssql.config import GenerationConfig
from claimssql.generate.prompts import RETRY_MESSAGE, build_abstain, build_forced
from claimssql.llm.base import DegenerateOutput, LLMClient, Message

PARSE_ERROR = "PARSE_ERROR"
# The model looped and the server aborted the reply (see llm.base.DegenerateOutput).
DEGENERATE_OUTPUT = "DEGENERATE_OUTPUT"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


class ParsedOutput(BaseModel):
    """The JSON object the prompts ask for."""

    model_config = ConfigDict(extra="ignore")  # extra keys are harmless

    answerable: bool
    sql: str | None = None
    confidence: int = Field(ge=0, le=100)
    reason: str = ""

    @model_validator(mode="after")
    def _answer_needs_sql(self) -> ParsedOutput:
        if self.answerable and not (self.sql and self.sql.strip()):
            raise ValueError('"answerable" is true but "sql" is empty')
        return self


class ParseFailure(Exception):
    pass


def clean_output(text: str) -> str:
    """Remove reasoning blocks and a surrounding code fence."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():
        # A closing tag without its opening tag: the answer is what follows it.
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = text.strip()
    if match := _CODE_FENCE.match(text):
        text = match.group(1).strip()
    return text


def parse_output(text: str) -> ParsedOutput:
    cleaned = clean_output(text)
    if not cleaned:
        raise ParseFailure("empty reply")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ParseFailure(f"not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ParseFailure("JSON is not an object")
    try:
        return ParsedOutput.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "object"
        raise ParseFailure(f"{where}: {first['msg']}") from exc


@dataclass
class Attempt:
    raw: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    parse_problem: str | None = None  # why this attempt's reply was rejected


@dataclass
class Generation:
    prompt: str  # "forced" | "abstain"
    mode: str  # "greedy" | "sample"
    temperature: float
    seed: int
    raw: str = ""  # the last reply, exactly as the model wrote it
    parsed: dict[str, Any] | None = None
    error: str | None = None  # None, PARSE_ERROR or DEGENERATE_OUTPUT
    attempts: list[Attempt] = field(default_factory=list)
    # Totals over all attempts (a retry costs time and tokens too).
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_one(
    client: LLMClient,
    messages: list[Message],
    prompt: str,
    mode: str,
    temperature: float,
    seed: int,
    max_parse_retries: int = 1,
) -> Generation:
    generation = Generation(prompt=prompt, mode=mode, temperature=temperature, seed=seed)
    conversation = list(messages)
    for _ in range(max_parse_retries + 1):
        try:
            response = client.chat(conversation, temperature=temperature, seed=seed,
                                   json_mode=True)  # fmt: skip
        except DegenerateOutput as exc:
            # No retry: there is no reply to correct, and the same call loops again.
            generation.attempts.append(Attempt(
                raw="", latency_ms=exc.latency_ms, prompt_tokens=0, completion_tokens=0,
                cached=exc.cached, parse_problem=exc.message,
            ))  # fmt: skip
            generation.latency_ms += exc.latency_ms
            generation.error = DEGENERATE_OUTPUT
            return generation
        attempt = Attempt(
            raw=response.text,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cached=response.cached,
        )
        generation.attempts.append(attempt)
        generation.raw = response.text
        generation.latency_ms += response.latency_ms
        generation.prompt_tokens += response.prompt_tokens
        generation.completion_tokens += response.completion_tokens
        try:
            generation.parsed = parse_output(response.text).model_dump()
            return generation
        except ParseFailure as failure:
            attempt.parse_problem = str(failure)
        # Retry: show the model its own reply and ask again. Different messages
        # mean a different cache key, and at temperature 0 an identical request
        # would only produce the identical bad reply.
        conversation = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": RETRY_MESSAGE},
        ]
    generation.error = PARSE_ERROR
    return generation


def generate_all(
    question: str,
    client: LLMClient,
    schema_context: str,
    config: GenerationConfig,
) -> list[Generation]:
    """The 2 + k generations for one question on one model, in a fixed order."""
    forced = build_forced(schema_context, question)
    abstain = build_abstain(schema_context, question)
    retries = config.max_parse_retries
    greedy = (config.greedy_temperature, config.greedy_seed)
    generations = [
        generate_one(client, forced, "forced", "greedy", *greedy, retries),
        generate_one(client, abstain, "abstain", "greedy", *greedy, retries),
    ]
    for seed in config.sample_seeds:
        generations.append(
            generate_one(client, abstain, "abstain", "sample", config.sample_temperature, seed,
                         retries)  # fmt: skip
        )
    return generations
