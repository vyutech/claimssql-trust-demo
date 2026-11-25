"""The report's second batch of tables: confidence intervals, data for the
figures, and report-ready exports. Called by eval/report.py.

  e2_rs_ci.csv                 EX, coverage, risk and RS_c with 95% intervals
  e3_harmful_ci.csv            harmful-answer counts with 95% intervals
  e4_auroc_ci.csv              AUROC of the three signals with 95% intervals
  differences_ci.csv           paired differences (C4 - C0, and small C4 vs mid-size C0)
  e4_roc_points.csv            ROC curve points for the signals
  e7_infeasible_by_group.csv   infeasible questions answered, by trap group and config
  table6_1_models.csv          the models: maker, size, settings, hardware, speed
  table7_1_latency_cost.csv    latency and cost per model, in seconds

Intervals are on the test split only (see eval/bootstrap.py for the method).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

from claimssql.config import ModelConfig
from claimssql.eval import metrics
from claimssql.eval.bootstrap import SIGNALS, Estimate, bootstrap_report, signal_arrays
from claimssql.eval.bootstrap import question_order as bootstrap_order
from claimssql.eval.score import score_record
from claimssql.policy.decide import CONFIGS

Table = tuple[list[str], list[tuple]]
INFEASIBLE_GROUPS = ("T1", "T2", "T3", "T4")
RATE_METRICS = ("ex", "coverage", "risk", "rs_0", "rs_1", "rs_10")
COUNT_METRICS = ("harmful", "answerable_wrong", "infeasible_answered")


def _r(value: float | None, digits: int = 4) -> Any:
    return "" if value is None else round(value, digits)


def _triple(est: Estimate, digits: int = 4) -> tuple:
    return _r(est.value, digits), _r(est.low, digits), _r(est.high, digits)


def _with_bounds(names: tuple[str, ...]) -> list[str]:
    return [column for name in names for column in (name, f"{name}_lo", f"{name}_hi")]


def cross_model_pairs(models: list[str], configs: list[ModelConfig]) -> list[tuple]:
    """RS_10 comparisons across models (RQ4). "Largest" = biggest size_gb in
    models.yaml; no pairs if sizes are not recorded.

      each smaller model's C4 against the largest model's C0
          (can a small verified model match a bigger unverified one?)
      the largest model's C4 against each smaller model's C4
          (with full verification on both, how much does size still add?)
    """
    size = {c.key: c.info.get("size_gb") for c in configs if c.key in models}
    known = {k: v for k, v in size.items() if isinstance(v, (int, float))}
    if len(known) < 2:
        return []
    largest = max(known, key=known.get)
    smaller = [model for model in models if model != largest]
    return ([((model, "C4"), (largest, "C0")) for model in smaller]
            + [((largest, "C4"), (model, "C4")) for model in smaller])  # fmt: skip


def ci_tables(
    test_records: list[dict[str, Any]],
    taus: dict[str, float],
    resamples: int,
    seed: int,
    cross_model: list[tuple],
) -> dict[str, Table]:
    if not test_records:
        return {}
    result = bootstrap_report(test_records, taus, resamples, seed, cross_model)
    n, used = result["n"], result["resamples"]

    e2, e3 = [], []
    for (model, config), estimates in result["metrics"].items():
        tau = taus[model] if config == "C4" else ""
        rates = [value for name in RATE_METRICS for value in _triple(estimates[name])]
        e2.append((model, config, tau, n, *rates, estimates["risk"].used))
        counts = [value for name in COUNT_METRICS for value in _triple(estimates[name], 1)]
        e3.append((model, config, tau, n, *counts))

    differences = []
    for label_a, label_b, name, est in result["differences"]:
        excludes = est.excludes_zero()
        digits = 4 if name == "auroc" else 2
        differences.append((f"{label_a} minus {label_b}", name, *_triple(est, digits), est.used,
                            "" if excludes is None else ("yes" if excludes else "no")))  # fmt: skip

    signals = {(m, s): row for m in {r["model_key"] for r in test_records}
               for row in metrics.signal_table([r for r in test_records if r["model_key"] == m])
               for s in [row["signal"]]}  # fmt: skip
    auroc = []
    for (model, signal), est in result["auroc"].items():
        row = signals[(model, signal)]
        auroc.append((model, signal, row["n"], row["n_wrong"], *_triple(est), est.used))

    return {
        "e2_rs_ci.csv": (
            ["model", "config", "tau", "n", *_with_bounds(RATE_METRICS), "risk_resamples_used"],
            e2),
        "e3_harmful_ci.csv": (["model", "config", "tau", "n", *_with_bounds(COUNT_METRICS)], e3),
        "differences_ci.csv": (
            ["comparison", "metric", "difference", "lo", "hi", "resamples_used",
             "interval_excludes_zero"], differences),
        "e4_auroc_ci.csv": (
            ["model", "signal", "n_answered", "n_wrong", "auroc", "auroc_lo", "auroc_hi",
             "resamples_used"], auroc),
        "bootstrap_info.csv": (
            ["split", "questions", "resamples", "seed", "level", "method"],
            [("test", n, used, seed, 0.95,
              "paired, stratified by group, percentile interval, tau fixed")]),
    }  # fmt: skip


def roc_points(test_records: list[dict[str, Any]]) -> Table:
    """Points of each signal's ROC curve (higher signal = more likely wrong)."""
    ids, _ = bootstrap_order(test_records)
    rows = []
    for model, arrays in signal_arrays(test_records, ids).items():
        answered, wrong = arrays["answered"], arrays["wrong"]
        if len(set(wrong[answered].tolist())) < 2:
            continue  # all right or all wrong: there is no curve
        for signal in SIGNALS:
            fpr, tpr, thresholds = roc_curve(wrong[answered], arrays[signal][answered])
            for x, y, threshold in zip(fpr, tpr, thresholds, strict=True):
                shown = "" if not np.isfinite(threshold) else round(float(threshold), 4)
                rows.append((model, signal, round(float(x), 4), round(float(y), 4), shown))
    return ["model", "signal", "false_positive_rate", "true_positive_rate", "threshold"], rows


def infeasible_by_group(
    records: list[dict[str, Any]], taus: dict[str, float], splits: tuple[str, ...]
) -> Table:
    rows = []
    for split in splits:
        for model in dict.fromkeys(r["model_key"] for r in records):
            for group in INFEASIBLE_GROUPS:
                subset = [r for r in records if r["model_key"] == model and r["group"] == group
                          and (split == "all" or r.get("split") == split)]  # fmt: skip
                answered = [sum(score_record(r, config, taus[model]).answered for r in subset)
                            for config in CONFIGS]  # fmt: skip
                rows.append((split, model, group, len(subset), *answered))
    return ["split", "model", "group", "n", *(f"answered_{c}" for c in CONFIGS)], rows


def _settings(options: dict[str, Any]) -> str:
    return "; ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}"
                     for k, v in options.items())  # fmt: skip


def model_tables(
    records: list[dict[str, Any]], configs: list[ModelConfig]
) -> dict[str, Table]:
    by_key = {c.key: c for c in configs}
    models_rows, cost_rows = [], []
    for model in dict.fromkeys(r["model_key"] for r in records):
        model_records = [r for r in records if r["model_key"] == model]
        stats = metrics.latency_cost(model_records)
        calls = [a["latency_ms"] for r in model_records
                 for g in (r["forced"], r["abstain"], *r.get("samples", []))
                 for a in g["attempts"]]  # fmt: skip
        mean_s = round(float(np.mean(calls)) / 1000, 2)
        info = by_key[model].info if model in by_key else {}
        options = by_key[model].options if model in by_key else model_records[0]["model_options"]
        models_rows.append((
            model, info.get("maker", ""), model_records[0]["model"], info.get("parameters", ""),
            info.get("quantization", ""), info.get("size_gb", ""), _settings(options),
            info.get("hardware", ""), mean_s,
        ))  # fmt: skip
        cost_rows.append((
            model, model_records[0]["model"], stats["n_questions"],
            round(stats["calls_per_question"], 2), mean_s,
            round(stats["latency_p50_ms"] / 1000, 2), round(stats["latency_p95_ms"] / 1000, 2),
            round(stats["c1_latency_per_question_ms"] / 1000, 2),
            round(stats["c4_latency_per_question_ms"] / 1000, 2),
            round(stats["prompt_tokens_per_question"]),
            round(stats["completion_tokens_per_question"]),
        ))  # fmt: skip
    return {
        "table6_1_models.csv": (
            ["model_key", "maker", "ollama_tag", "parameters", "quantization", "size_gb",
             "settings", "hardware", "mean_seconds_per_call"], models_rows),
        "table7_1_latency_cost.csv": (
            ["model_key", "ollama_tag", "questions", "calls_per_question",
             "mean_seconds_per_call", "p50_seconds_per_call", "p95_seconds_per_call",
             "seconds_per_question_c1_to_c3", "seconds_per_question_c4",
             "prompt_tokens_per_question", "completion_tokens_per_question"], cost_rows),
    }  # fmt: skip


def build_stats_tables(
    records: list[dict[str, Any]],
    taus: dict[str, float],
    configs: list[ModelConfig],
    resamples: int,
    seed: int,
    splits: tuple[str, ...] = ("test", "all"),
) -> dict[str, Table]:
    test = [r for r in records if r.get("split") == "test"]
    models = list(dict.fromkeys(r["model_key"] for r in records))
    tables = ci_tables(test, taus, resamples, seed, cross_model_pairs(models, configs))
    tables["e4_roc_points.csv"] = roc_points(test)
    tables["e7_infeasible_by_group.csv"] = infeasible_by_group(records, taus, splits)
    tables.update(model_tables(records, configs))
    return tables
