"""The live pipeline behind the demo app and the API: one question in, an
answer or an abstention out.

    1 greedy call with the abstain prompt (the answer to verify)
    k sampled calls (k = 3 in the demo, to save the free tier's tokens)
    V1 schema check and V2 dry run on the greedy SQL
    V3 self-consistency over the k samples
    decide() with config C4

With k = 3, agreement can only be 0, 1/3, 2/3 or 1, so the benchmark's tau
values do not carry over. The demo answers when at least 2 of the 3 samples
agree with the greedy result (tau = 2/3). This is a demo setting, not a
benchmark result.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from claimssql.config import (
    CACHE_DIR,
    REPO_ROOT,
    GenerationConfig,
    ModelConfig,
    load_experiment,
    load_models,
)
from claimssql.eval.compare import execute_sql
from claimssql.generate.generator import Generation, generate_one
from claimssql.generate.prompts import PROMPT_VERSION, build_abstain
from claimssql.llm.base import LLMClient
from claimssql.llm.factory import make_client
from claimssql.policy.decide import decide
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH, Catalogue, load_catalogue
from claimssql.schema.context_builder import render_context
from claimssql.verify.dry_run import dry_run
from claimssql.verify.schema_check import Schema, check_sql, schema_from_catalogue
from claimssql.verify.self_consistency import self_consistency

DEMO_K = 3
DEMO_TAU = 2 / 3
RESULT_ROWS = 100  # rows of an answer shown to the user
DEFAULT_DEMO_DB = REPO_ROOT / "data" / "demo.duckdb"

# What each abstain reason means, for people who are not reading the code.
REASON_TEXT = {
    "MODEL_REFUSED": "The model said this question cannot be answered from this database.",
    "SCHEMA_UNKNOWN": "The SQL used a table or column that does not exist in the database.",
    "EXEC_ERROR": "The SQL could not be run on the database.",
    "LOW_AGREEMENT": "The model's answers disagreed with each other, so the answer was not "
                     "trusted.",
    "PARSE_ERROR": "The model's reply could not be read.",
    "DEGENERATE_OUTPUT": "The model got stuck repeating itself and produced no answer.",
}  # fmt: skip

Progress = Callable[[str], None]


@dataclass
class Answer:
    question: str
    answered: bool
    reason_code: str | None
    reason_text: str | None
    sql: str | None  # the SQL given as the answer (None when abstaining)
    model_sql: str | None  # what the model wrote, even if the system abstained
    model_reason: str | None  # the model's own one-line explanation
    confidence: int | None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    more_rows: bool = False
    k: int = DEMO_K
    tau: float = DEMO_TAU
    agreement: float | None = None
    cluster_sizes: list[int] = field(default_factory=list)
    unknown_tables: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)
    exec_error: str | None = None
    model: str = ""
    calls: int = 0
    cached_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sql_of(generation: Generation) -> str | None:
    parsed = generation.parsed
    if generation.error or not parsed or not parsed.get("answerable"):
        return None
    sql = parsed.get("sql")
    return sql if sql and sql.strip() else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return float(value) if isinstance(value, Decimal) else str(value)


def ask(
    question: str,
    client: LLMClient,
    con: duckdb.DuckDBPyConnection,
    schema: Schema,
    context: str,
    generation: GenerationConfig,
    k: int = DEMO_K,
    tau: float = DEMO_TAU,
    timeout_s: float = 30.0,
    progress: Progress | None = None,
) -> Answer:
    """Run the pipeline. LLM errors (e.g. RateLimitStop) are left to the caller."""
    say = progress or (lambda message: None)
    messages = build_abstain(context, question)
    retries = generation.max_parse_retries

    say("Asking the model for an answer")
    greedy = generate_one(client, messages, "abstain", "greedy", generation.greedy_temperature,
                          generation.greedy_seed, retries)  # fmt: skip
    samples = []
    for number, seed in enumerate(generation.sample_seeds[:k], start=1):
        say(f"Asking again for a consistency check ({number} of {k})")
        samples.append(generate_one(client, messages, "abstain", "sample",
                                    generation.sample_temperature, seed, retries))  # fmt: skip

    say("Checking the SQL")
    sql = _sql_of(greedy)
    v1 = check_sql(sql, schema).to_dict() if sql else None
    v2 = dry_run(con, sql, timeout_s=timeout_s).to_dict() if sql else None
    v3 = self_consistency(
        con, [_sql_of(s) for s in samples], greedy_hash=(v2 or {}).get("result_hash"),
        timeout_s=timeout_s, sample_errors=[s.error for s in samples],
    ).to_dict()  # fmt: skip
    record = {"abstain": greedy.to_dict(), "v1": v1, "v2": v2, "v3": v3}
    decision = decide(record, "C4", tau)

    parsed = greedy.parsed or {}
    attempts = [a for g in [greedy, *samples] for a in g.attempts]
    answer = Answer(
        question=question,
        answered=decision.answered,
        reason_code=decision.reason_code,
        reason_text=REASON_TEXT.get(decision.reason_code) if decision.reason_code else None,
        sql=decision.sql,
        model_sql=parsed.get("sql"),
        model_reason=parsed.get("reason"),
        confidence=parsed.get("confidence"),
        k=k,
        tau=tau,
        agreement=v3["agreement"] if sql else None,
        cluster_sizes=v3["cluster_sizes"],
        unknown_tables=(v1 or {}).get("unknown_tables", []),
        unknown_columns=(v1 or {}).get("unknown_columns", []),
        exec_error=(v1 or {}).get("parse_error") or (v2 or {}).get("error"),
        model=getattr(client, "model", ""),
        calls=len(attempts),
        cached_calls=sum(a.cached for a in attempts),
    )
    if decision.answered:
        say("Running the query")
        result = execute_sql(con, decision.sql, timeout_s=timeout_s, max_rows=RESULT_ROWS)
        answer.columns = result.columns
        answer.rows = [[_jsonable(v) for v in row] for row in result.rows]
        answer.more_rows = result.truncated
    return answer


def demo_model(models: list[ModelConfig] | None = None) -> ModelConfig:
    """The model marked `use: demo` in configs/models.yaml."""
    demo = [m for m in (models or load_models().models) if m.use == "demo"]
    if not demo:
        raise ValueError("no model with use: demo in configs/models.yaml")
    return demo[0]


class DemoService:
    """Everything `ask` needs, opened once and shared by the app or the API."""

    def __init__(
        self,
        client: LLMClient,
        con: duckdb.DuckDBPyConnection,
        catalogue: Catalogue,
        generation: GenerationConfig,
        db_path: Path | None = None,
    ) -> None:
        self.client = client
        self.con = con
        self.catalogue = catalogue
        self.schema = schema_from_catalogue(catalogue)
        self.context = render_context(catalogue)
        self.generation = generation
        self.db_path = db_path

    @classmethod
    def from_env(cls) -> DemoService:
        """Settings from the environment, with project defaults:
        CLAIMSSQL_DB (default data/demo.duckdb), CLAIMSSQL_CACHE_DIR (default
        results/cache), GROQ_API_KEY (from .env locally, a Space secret online)."""
        db_path = Path(os.getenv("CLAIMSSQL_DB") or DEFAULT_DEMO_DB)
        cache_dir = Path(os.getenv("CLAIMSSQL_CACHE_DIR") or CACHE_DIR)
        con = duckdb.connect(str(db_path), read_only=True)
        client = make_client(demo_model(), PROMPT_VERSION, cache_dir=cache_dir)
        return cls(client, con, load_catalogue(DEFAULT_CATALOGUE_PATH),
                   load_experiment().generation, db_path)  # fmt: skip

    def ask(self, question: str, progress: Progress | None = None) -> Answer:
        # A cursor per question: DuckDB connections must not be shared by threads.
        cursor = self.con.cursor()
        try:
            return ask(question, self.client, cursor, self.schema, self.context,
                       self.generation, progress=progress)  # fmt: skip
        finally:
            cursor.close()
