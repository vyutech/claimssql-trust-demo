"""Turn logged records into scored cases - offline, no model calls.

For one record (question x model) and one configuration C0-C4:
  1. decide() says whether the system answers, and with which SQL;
  2. an answer is correct when its result hash equals the gold hash;
  3. the case falls into one of the five TrustSQL outcome regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from claimssql.policy.decide import CONFIGS, decide

ANSWERABLE_CORRECT = "answerable_correct"
ANSWERABLE_WRONG = "answerable_wrong"
ANSWERABLE_ABSTAINED = "answerable_abstained"
INFEASIBLE_ABSTAINED = "infeasible_abstained"
INFEASIBLE_ANSWERED = "infeasible_answered"
REGIONS = (
    ANSWERABLE_CORRECT, ANSWERABLE_WRONG, ANSWERABLE_ABSTAINED,
    INFEASIBLE_ABSTAINED, INFEASIBLE_ANSWERED,
)  # fmt: skip
# Regions where the system gave an answer it should not have given.
HARMFUL = (ANSWERABLE_WRONG, INFEASIBLE_ANSWERED)


@dataclass(frozen=True)
class ScoredCase:
    question_id: str
    model_key: str
    group: str
    split: str | None
    answerable: bool
    config: str
    answered: bool
    reason_code: str | None
    sql: str | None
    correct: bool
    region: str


def verifier_outputs(record: dict[str, Any], config: str) -> tuple[dict | None, dict | None]:
    """(V1, V2) of the SQL this config would answer with: C0 uses the forced
    prompt's SQL, C1-C4 the abstain prompt's greedy SQL."""
    if config == "C0":
        return record.get("forced_v1"), record.get("forced_v2")
    return record.get("v1"), record.get("v2")


def region_of(answerable: bool, answered: bool, correct: bool) -> str:
    if answerable:
        if not answered:
            return ANSWERABLE_ABSTAINED
        return ANSWERABLE_CORRECT if correct else ANSWERABLE_WRONG
    return INFEASIBLE_ANSWERED if answered else INFEASIBLE_ABSTAINED


def score_record(record: dict[str, Any], config: str, tau: float | None = None) -> ScoredCase:
    decision = decide(record, config, tau)
    _, v2 = verifier_outputs(record, config)
    answer_hash = (v2 or {}).get("result_hash")
    gold_hash = record.get("gold_result_hash")
    # An answer whose SQL failed (no hash) can never be correct.
    correct = bool(
        decision.answered and record["answerable"] and answer_hash and answer_hash == gold_hash
    )
    return ScoredCase(
        question_id=record["question_id"],
        model_key=record["model_key"],
        group=record["group"],
        split=record.get("split"),
        answerable=record["answerable"],
        config=config,
        answered=decision.answered,
        reason_code=decision.reason_code,
        sql=decision.sql,
        correct=correct,
        region=region_of(record["answerable"], decision.answered, correct),
    )


def score_records(
    records: list[dict[str, Any]], config: str, tau: float | None = None
) -> list[ScoredCase]:
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}")
    return [score_record(record, config, tau) for record in records]
