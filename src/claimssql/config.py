"""Load and validate configs/models.yaml, configs/experiment.yaml and .env.

Model names live only in models.yaml; the rest of the code asks for a model by
its short `key`. Pydantic validates the files at load time so a typo fails
early with a clear message instead of halfway through a benchmark run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# src/claimssql/config.py -> repo root is two levels above the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_TABLES_DIR = REPO_ROOT / "results" / "tables"
RESULTS_RUNS_DIR = REPO_ROOT / "results" / "runs"
CACHE_DIR = REPO_ROOT / "results" / "cache"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class ModelConfig(BaseModel):
    key: str = Field(min_length=1)
    provider: Literal["ollama", "groq"]
    model: str = Field(min_length=1)
    # benchmark = in the full runs; demo = live demo and small tests only.
    use: Literal["benchmark", "demo"] = "benchmark"
    # Per-model call settings (think, num_ctx, reasoning_effort, max_tokens).
    options: dict[str, bool | int | float | str] = Field(default_factory=dict)
    # Facts about the model for the report's model table (maker, size, hardware).
    # Not sent to the model and not part of the cache key.
    info: dict[str, bool | int | float | str] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _keys_unique(self) -> ModelsConfig:
        keys = [m.key for m in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate model keys in models.yaml: {keys}")
        return self

    def get(self, key: str) -> ModelConfig:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"no model with key {key!r} in models.yaml")


class GenerationConfig(BaseModel):
    greedy_temperature: float = Field(ge=0)
    greedy_seed: int
    k_samples: int = Field(ge=1)
    sample_temperature: float = Field(ge=0)
    sample_seeds: list[int]
    max_parse_retries: int = Field(ge=0)

    @model_validator(mode="after")
    def _one_seed_per_sample(self) -> GenerationConfig:
        if len(self.sample_seeds) != self.k_samples:
            raise ValueError("sample_seeds must have exactly k_samples entries")
        return self


class VerifyConfig(BaseModel):
    dry_run_timeout_s: float = Field(gt=0)
    result_row_cap: int = Field(ge=1)
    float_round_dp: int = Field(ge=0)


class PolicyConfig(BaseModel):
    # None until tau is tuned on the dev split (P6).
    tau: float | None = Field(default=None, ge=0, le=1)


class BenchmarkConfig(BaseModel):
    split_seed: int
    dev_fraction: float = Field(gt=0, lt=1)
    composition: dict[str, int] = Field(min_length=1)  # group -> target count


class MetricsConfig(BaseModel):
    rs_penalties: list[float] = Field(min_length=1)
    bootstrap_resamples: int = Field(default=2000, ge=1)
    bootstrap_seed: int = 20261117


class ExperimentConfig(BaseModel):
    generation: GenerationConfig
    verify: VerifyConfig
    policy: PolicyConfig
    benchmark: BenchmarkConfig
    metrics: MetricsConfig


class EnvSettings(BaseModel):
    groq_api_key: str | None
    ollama_host: str

    def __repr__(self) -> str:  # never let the key leak into logs or tracebacks
        key = "set" if self.groq_api_key else "missing"
        return f"EnvSettings(groq_api_key=<{key}>, ollama_host={self.ollama_host!r})"

    __str__ = __repr__


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_models(path: Path | None = None) -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(path or CONFIG_DIR / "models.yaml"))


def load_experiment(path: Path | None = None) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_read_yaml(path or CONFIG_DIR / "experiment.yaml"))


def load_env(env_file: Path | None = None) -> EnvSettings:
    """Read secrets from .env (real environment variables take precedence)."""
    load_dotenv(env_file or REPO_ROOT / ".env")
    return EnvSettings(
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip() or None,
        ollama_host=(os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/"),
    )
