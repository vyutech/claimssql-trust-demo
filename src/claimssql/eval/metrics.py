"""The numbers of the report, computed from scored cases and logged records.

Definitions (TrustSQL-style):
  EX        answerable questions answered correctly / answerable questions
  coverage  questions answered / all questions
  risk      harmful answers / questions answered
            (harmful = wrong answer, or any answer to an infeasible question)
  RS_c      100 * mean score, where a case scores
              +1  answerable-correct or infeasible-abstained
               0  answerable-abstained
              -c  answerable-wrong or infeasible-answered
            c is the penalty for a harmful answer: with c = 10 one wrong answer
            cancels ten right ones.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from claimssql.eval.score import (
    ANSWERABLE_CORRECT,
    HARMFUL,
    INFEASIBLE_ABSTAINED,
    REGIONS,
    ScoredCase,
    score_record,
    score_records,
)

PENALTIES = (0, 1, 10)
# With k = 5 samples, agreement can only be 0, 0.2, ..., 1.0, so these six
# thresholds are every distinct behaviour of C4.
TAU_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SELECTION_PENALTY = 10  # tau is chosen by RS_10 on the dev split
FALLBACK_TAU = 0.6  # used only when a model has no dev records


def region_counts(cases: list[ScoredCase]) -> dict[str, int]:
    counts = Counter(case.region for case in cases)
    return {region: counts.get(region, 0) for region in REGIONS}


def reliability_score(cases: list[ScoredCase], penalty: float) -> float | None:
    if not cases:
        return None
    total = 0.0
    for case in cases:
        if case.region in (ANSWERABLE_CORRECT, INFEASIBLE_ABSTAINED):
            total += 1
        elif case.region in HARMFUL:
            total -= penalty
        # answerable-abstained scores 0
    return 100 * total / len(cases)


def summarise(cases: list[ScoredCase]) -> dict[str, Any]:
    """n, EX, coverage, risk and RS_c for one set of cases."""
    counts = region_counts(cases)
    n = len(cases)
    answerable = sum(case.answerable for case in cases)
    answered = sum(case.answered for case in cases)
    harmful = sum(counts[region] for region in HARMFUL)
    out: dict[str, Any] = {
        "n": n,
        "n_answerable": answerable,
        "answered": answered,
        "ex": counts[ANSWERABLE_CORRECT] / answerable if answerable else None,
        "coverage": answered / n if n else None,
        "risk": harmful / answered if answered else None,  # undefined when nothing is answered
    }
    for penalty in PENALTIES:
        out[f"rs_{penalty}"] = reliability_score(cases, penalty)
    return out


def risk_coverage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per tau: what C4 does at that threshold."""
    return [{"tau": tau, **summarise(score_records(records, "C4", tau))} for tau in TAU_GRID]


def choose_tau(dev_records: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """tau with the highest RS_10 on the dev records; ties go to the lowest tau
    (the one that answers more questions). Returns (tau, the rows compared)."""
    rows = risk_coverage(dev_records)
    if not dev_records:
        return FALLBACK_TAU, rows
    key = f"rs_{SELECTION_PENALTY}"
    best = max(rows, key=lambda row: (row[key], -row["tau"]))
    return best["tau"], rows


# -------------------------------------------------------------- signals ----


def auroc(is_wrong: list[bool], signal: list[float]) -> float | None:
    """How well a signal ranks wrong answers above right ones (1.0 = perfectly,
    0.5 = no better than chance). None when all answers are right or all wrong."""
    if len(set(is_wrong)) < 2:
        return None
    return float(roc_auc_score(is_wrong, signal))


def signal_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AUROC of each abstention signal, over the questions the model answered
    under the abstain prompt (the C1 answers)."""
    wrong: list[bool] = []
    signals: dict[str, list[float]] = {
        "1 - agreement": [], "100 - confidence": [], "sql length": [],
    }  # fmt: skip
    for record in records:
        case = score_record(record, "C1")
        if not case.answered:
            continue
        wrong.append(not case.correct)
        parsed = record["abstain"]["parsed"]
        signals["1 - agreement"].append(1 - (record.get("v3") or {}).get("agreement", 0.0))
        signals["100 - confidence"].append(100 - parsed.get("confidence", 0))
        signals["sql length"].append(len(case.sql or ""))
    return [
        {"signal": name, "n": len(wrong), "n_wrong": sum(wrong), "auroc": auroc(wrong, values)}
        for name, values in signals.items()
    ]


# --------------------------------------------- failed generations ----


def generation_failures(records: list[dict[str, Any]]) -> dict[str, int]:
    """How often a model's replies were unusable, over all 7 generations per
    question: unreadable (PARSE_ERROR) or looped and aborted (DEGENERATE_OUTPUT)."""
    generations = [g for r in records for g in _generations(r)]
    greedy = [g for r in records for g in (r["forced"], r["abstain"])]

    def count(items: list[dict[str, Any]], code: str) -> int:
        return sum(g.get("error") == code for g in items)

    return {
        "questions": len(records),
        "generations": len(generations),
        "parse_error": count(generations, "PARSE_ERROR"),
        "degenerate_output": count(generations, "DEGENERATE_OUTPUT"),
        "degenerate_greedy": count(greedy, "DEGENERATE_OUTPUT"),
        "questions_with_degenerate": sum(
            any(g.get("error") == "DEGENERATE_OUTPUT" for g in _generations(r)) for r in records
        ),
    }


def abstain_reasons(cases: list[ScoredCase], reason_codes: tuple[str, ...]) -> dict[str, int]:
    """Abstentions per reason code (an abstention has exactly one)."""
    counts = Counter(case.reason_code for case in cases if not case.answered)
    return {code: counts.get(code, 0) for code in reason_codes}


# ------------------------------------------------------ latency and cost ----


def _generations(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [record["forced"], record["abstain"], *record.get("samples", [])]


def latency_cost(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-call latency percentiles and per-question cost for one model.
    Latencies are those of the original calls (the cache keeps them)."""
    if not records:
        return {"n_questions": 0}
    call_ms = [a["latency_ms"] for r in records for g in _generations(r) for a in g["attempts"]]
    calls = [sum(len(g["attempts"]) for g in _generations(r)) for r in records]
    c1_ms = [r["abstain"]["latency_ms"] for r in records]
    c4_ms = [r["abstain"]["latency_ms"] + sum(s["latency_ms"] for s in r.get("samples", []))
             for r in records]  # fmt: skip
    prompt = [sum(g["prompt_tokens"] for g in _generations(r)) for r in records]
    completion = [sum(g["completion_tokens"] for g in _generations(r)) for r in records]
    return {
        "n_questions": len(records),
        "calls_per_question": float(np.mean(calls)),
        "latency_p50_ms": float(np.percentile(call_ms, 50)),
        "latency_p95_ms": float(np.percentile(call_ms, 95)),
        # What a user would wait for one answer: C1-C3 need 1 call, C4 needs 1 + k.
        "c1_latency_per_question_ms": float(np.mean(c1_ms)),
        "c4_latency_per_question_ms": float(np.mean(c4_ms)),
        "prompt_tokens_per_question": float(np.mean(prompt)),
        "completion_tokens_per_question": float(np.mean(completion)),
    }
