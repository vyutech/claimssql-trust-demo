"""Confidence intervals for the report, by bootstrap over questions.

The test split is one sample of 160 questions. To see how much each number
would move with a different sample, we resample the questions with replacement
many times (2,000 by default) and recompute every metric each time. The middle
95% of those values is the confidence interval.

Three choices worth knowing:
  * Stratified: questions are resampled inside each of the 7 groups, so every
    resample keeps the benchmark's composition (as many easy, T1, ... questions
    as the real test split).
  * Paired: every model and config is scored on the SAME resampled questions.
    A difference such as "C4 minus C0" is computed inside each resample, so
    question difficulty cancels out and the interval is about the difference.
  * tau is fixed at each model's dev-chosen value; it is not re-chosen per
    resample. The intervals describe the system as evaluated.

If a metric is undefined in a resample (risk when nothing was answered, AUROC
when all answers are right), that resample is left out for that metric and the
number of resamples used is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata

from claimssql.eval.score import HARMFUL, score_record
from claimssql.policy.decide import CONFIGS

PENALTIES = (0, 1, 10)
METRICS = ("ex", "coverage", "risk", "rs_0", "rs_1", "rs_10",
           "harmful", "answerable_wrong", "infeasible_answered")  # fmt: skip
SIGNALS = ("1 - agreement", "100 - confidence", "sql length")
AUROC_COMPARISON = ("1 - agreement", "100 - confidence")  # reported as first minus second


@dataclass(frozen=True)
class Estimate:
    value: float | None  # on the real data
    low: float | None  # 2.5th percentile over resamples
    high: float | None  # 97.5th percentile
    used: int  # resamples in which the metric was defined

    def excludes_zero(self) -> bool | None:
        if self.low is None or self.high is None:
            return None
        return self.low > 0 or self.high < 0


@dataclass
class CaseArrays:
    """One model x config, one entry per question (same question order for all)."""

    answerable: np.ndarray
    answered: np.ndarray
    correct: np.ndarray
    answerable_wrong: np.ndarray
    infeasible_answered: np.ndarray


def question_order(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(question ids, their groups) in first-seen order; every model must cover them."""
    ids: list[str] = []
    groups: dict[str, str] = {}
    for record in records:
        if record["question_id"] not in groups:
            ids.append(record["question_id"])
            groups[record["question_id"]] = record["group"]
    return ids, [groups[i] for i in ids]


def case_arrays(
    records: list[dict[str, Any]], taus: dict[str, float], ids: list[str]
) -> dict[tuple[str, str], CaseArrays]:
    position = {qid: i for i, qid in enumerate(ids)}
    out: dict[tuple[str, str], CaseArrays] = {}
    for model in dict.fromkeys(r["model_key"] for r in records):
        model_records = [r for r in records if r["model_key"] == model]
        if {r["question_id"] for r in model_records} != set(ids):
            raise ValueError(f"model '{model}' does not cover the same questions as the others")
        for config in CONFIGS:
            arrays = {name: np.zeros(len(ids), dtype=bool) for name in
                      ("answerable", "answered", "correct", "answerable_wrong",
                       "infeasible_answered")}  # fmt: skip
            for record in model_records:
                case = score_record(record, config, taus[model])
                i = position[record["question_id"]]
                arrays["answerable"][i] = case.answerable
                arrays["answered"][i] = case.answered
                arrays["correct"][i] = case.correct
                if case.region in HARMFUL:
                    kind = "answerable_wrong" if case.answerable else "infeasible_answered"
                    arrays[kind][i] = True
            out[(model, config)] = CaseArrays(**arrays)
    return out


def stratified_indices(groups: list[str], resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(resamples x n) question positions, drawn with replacement inside each group."""
    labels = np.array(groups)
    parts = []
    for group in dict.fromkeys(groups):
        members = np.flatnonzero(labels == group)
        parts.append(rng.choice(members, size=(resamples, len(members)), replace=True))
    return np.concatenate(parts, axis=1)


def metric_values(arrays: CaseArrays, index: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric for each row of `index` (one row = one resample)."""
    n = index.shape[1]
    answerable = arrays.answerable[index].sum(axis=1)
    answered = arrays.answered[index].sum(axis=1)
    correct = arrays.correct[index].sum(axis=1)
    wrong = arrays.answerable_wrong[index].sum(axis=1)
    infeasible_answered = arrays.infeasible_answered[index].sum(axis=1)
    infeasible_abstained = (~arrays.answerable & ~arrays.answered)[index].sum(axis=1)
    harmful = wrong + infeasible_answered
    with np.errstate(divide="ignore", invalid="ignore"):
        values = {
            "ex": np.where(answerable > 0, correct / answerable, np.nan),
            "coverage": answered / n,
            "risk": np.where(answered > 0, harmful / answered, np.nan),
            "harmful": harmful.astype(float),
            "answerable_wrong": wrong.astype(float),
            "infeasible_answered": infeasible_answered.astype(float),
        }
    for penalty in PENALTIES:
        values[f"rs_{penalty}"] = 100 * (correct + infeasible_abstained - penalty * harmful) / n
    return values


def estimate(point: float, samples: np.ndarray, level: float = 0.95) -> Estimate:
    defined = samples[~np.isnan(samples)]
    if np.isnan(point) or defined.size == 0:
        return Estimate(None if np.isnan(point) else float(point), None, None, int(defined.size))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(defined, [tail, 100 - tail])
    return Estimate(float(point), float(low), float(high), int(defined.size))


def fast_auroc(is_wrong: np.ndarray, signal: np.ndarray) -> float:
    """AUROC from ranks (ties share a rank); NaN when only one class is present.
    Same value as sklearn's roc_auc_score, but fast enough for 2,000 resamples."""
    positives = int(is_wrong.sum())
    negatives = is_wrong.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(signal)
    return float((ranks[is_wrong].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))  # fmt: skip


def signal_arrays(
    records: list[dict[str, Any]], ids: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Per model: who answered under C1, which of those answers were wrong, and
    the three signals, all in question order."""
    position = {qid: i for i, qid in enumerate(ids)}
    out: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        arrays = out.setdefault(record["model_key"], {
            "answered": np.zeros(len(ids), dtype=bool), "wrong": np.zeros(len(ids), dtype=bool),
            **{name: np.zeros(len(ids)) for name in SIGNALS},
        })  # fmt: skip
        case = score_record(record, "C1")
        if not case.answered:
            continue
        i = position[record["question_id"]]
        parsed = record["abstain"]["parsed"]
        arrays["answered"][i] = True
        arrays["wrong"][i] = not case.correct
        arrays["1 - agreement"][i] = 1 - (record.get("v3") or {}).get("agreement", 0.0)
        arrays["100 - confidence"][i] = 100 - parsed.get("confidence", 0)
        arrays["sql length"][i] = len(case.sql or "")
    return out


def bootstrap_report(
    records: list[dict[str, Any]],
    taus: dict[str, float],
    resamples: int = 2000,
    seed: int = 0,
    cross_model: list[tuple[tuple[str, str], tuple[str, str]]] | None = None,
) -> dict[str, Any]:
    """All intervals for one set of records (normally the test split).

    Returns {"metrics": {(model, config): {metric: Estimate}},
             "differences": [(label_a, label_b, metric, Estimate)],
             "auroc": {(model, signal): Estimate}, "n": ..., "resamples": ...}.
    `cross_model`: extra RS_10 differences, e.g. (("qwen", "C4"), ("mid_oss", "C0")).
    For every model it also adds AUROC(1 - agreement) minus AUROC(100 - confidence).
    """
    ids, groups = question_order(records)
    cases = case_arrays(records, taus, ids)
    rng = np.random.default_rng(seed)
    index = stratified_indices(groups, resamples, rng)
    identity = np.arange(len(ids))[None, :]

    points = {key: metric_values(arrays, identity) for key, arrays in cases.items()}
    samples = {key: metric_values(arrays, index) for key, arrays in cases.items()}
    metrics = {
        key: {name: estimate(points[key][name][0], samples[key][name]) for name in METRICS}
        for key in cases
    }

    def difference(a: tuple[str, str], b: tuple[str, str], name: str) -> Estimate:
        # Same resamples for a and b: subtracting row by row is the paired part.
        return estimate(points[a][name][0] - points[b][name][0],
                        samples[a][name] - samples[b][name])  # fmt: skip

    differences = []
    for model in dict.fromkeys(m for m, _ in cases):
        for name in ("rs_10", "harmful"):
            differences.append((f"{model} C4", f"{model} C0", name,
                                difference((model, "C4"), (model, "C0"), name)))  # fmt: skip
    for a, b in cross_model or []:
        if a in cases and b in cases:
            differences.append((f"{a[0]} {a[1]}", f"{b[0]} {b[1]}", "rs_10",
                                difference(a, b, "rs_10")))  # fmt: skip

    auroc = {}
    for model, arrays in signal_arrays(records, ids).items():
        answered, wrong = arrays["answered"], arrays["wrong"]
        points, values = {}, {}
        for name in SIGNALS:
            points[name] = fast_auroc(wrong[answered], arrays[name][answered])
            values[name] = np.empty(resamples)
            for row in range(resamples):
                picked = index[row][answered[index[row]]]
                values[name][row] = fast_auroc(wrong[picked], arrays[name][picked])
            auroc[(model, name)] = estimate(points[name], values[name])
        # Does self-consistency beat the model's own confidence? Paired: both
        # AUROCs come from the same resampled questions.
        first, second = AUROC_COMPARISON
        differences.append((f"{model} AUROC({first})", f"{model} AUROC({second})", "auroc",
                            estimate(points[first] - points[second],
                                     values[first] - values[second])))  # fmt: skip

    return {"metrics": metrics, "differences": differences, "auroc": auroc,
            "n": len(ids), "resamples": resamples}  # fmt: skip
