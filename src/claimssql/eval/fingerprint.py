"""A fingerprint of everything that decides the LLM cache keys.

Generation runs on Kaggle and verification on the laptop; they only fit
together if both compute the same cache key for the same call. The key depends
on the model name and options, the prompt version, the prompt text (which
contains the rendered catalogue) and the question.

The fingerprint holds the real cache key of one fixed probe call per benchmark
model, plus hashes of the schema context and of all question texts. If two
machines print the same fingerprint, a cache written on one is found on the other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from claimssql.config import REPO_ROOT, load_experiment, load_models
from claimssql.generate.prompts import PROMPT_VERSION, build_forced
from claimssql.llm.cache import cache_key
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH, load_catalogue
from claimssql.schema.context_builder import render_context

DEFAULT_QUESTIONS = REPO_ROOT / "benchmark" / "questions.jsonl"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(
    catalogue_path: Path | None = None, questions_path: Path | None = None
) -> dict[str, Any]:
    context = render_context(load_catalogue(catalogue_path or DEFAULT_CATALOGUE_PATH))
    lines = (questions_path or DEFAULT_QUESTIONS).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line)["question"] for line in lines if line.strip()]
    generation = load_experiment().generation

    probe_keys = {}
    if questions:
        # The first call the runner makes for the first question: forced, greedy.
        messages = build_forced(context, questions[0])
        for model in load_models().models:
            if model.use == "benchmark":
                probe_keys[model.key] = cache_key(
                    model.provider, model.model, PROMPT_VERSION, messages,
                    generation.greedy_temperature, generation.greedy_seed, model.options, True,
                )  # fmt: skip
    return {
        "prompt_version": PROMPT_VERSION,
        "context_sha256": _sha256(context),
        "questions_sha256": _sha256("\n".join(questions)),
        "n_questions": len(questions),
        "probe_keys": probe_keys,
    }


def differences(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    """Human-readable list of what differs between two fingerprints."""
    out = []
    for name in ("prompt_version", "context_sha256", "questions_sha256", "n_questions"):
        if ours.get(name) != theirs.get(name):
            out.append(f"{name}: {theirs.get(name)!r} vs here {ours.get(name)!r}")
    ours_keys, theirs_keys = ours.get("probe_keys", {}), theirs.get("probe_keys", {})
    for model in sorted(set(ours_keys) | set(theirs_keys)):
        if ours_keys.get(model) != theirs_keys.get(model):
            out.append(f"cache key for model '{model}' differs")
    return out
