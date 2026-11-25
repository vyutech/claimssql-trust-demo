"""Turn one run file into the report's tables. No model calls, no database.

Reads results/runs/<run_id>.jsonl and writes to results/tables/:

  tau_selection.csv              per model: C4 at every tau on the DEV split, and the chosen tau
  e1_ex_by_model_difficulty.csv  execution accuracy by model, difficulty and config
  e2_rs_by_config_penalty.csv    coverage, risk and Reliability Score by model and config
  e3_taxonomy.csv                kinds of harmful answers by model and config
  e3_generation_failures.csv     unusable replies per model: PARSE_ERROR and
                                 DEGENERATE_OUTPUT (the model looped)
  abstain_reasons.csv            abstentions by reason code, model and config
  e4_signal_auroc.csv            how well each abstention signal spots wrong answers
  e4_risk_coverage.csv           C4 at every tau (risk-coverage curve)
  e5_latency_cost.csv            latency, calls and tokens per model
  outcomes_confusion.csv         the five outcome regions by model and config
  failures.jsonl                 every harmful answer, with its SQL and label

tau is chosen per model on the dev split only (highest RS_10); every other
table reports the test split, with the all-questions numbers for reference.

It also writes the tables of eval/report_stats.py: 95% confidence intervals
(e2_rs_ci, e3_harmful_ci, e4_auroc_ci, differences_ci), data for the figures
(e4_roc_points, e7_infeasible_by_group) and the report exports
(table6_1_models, table7_1_latency_cost).

Usage:
    python -m claimssql.eval.report --run-id full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from claimssql.config import RESULTS_RUNS_DIR, RESULTS_TABLES_DIR, load_experiment, load_models
from claimssql.db.load_synpuf import write_csv
from claimssql.eval import metrics
from claimssql.eval.report_stats import build_stats_tables
from claimssql.eval.run_benchmark import read_run
from claimssql.eval.score import HARMFUL, REGIONS, score_records, verifier_outputs
from claimssql.eval.taxonomy import LABELS, LowCardinality, label_case, normalise_low_cardinality
from claimssql.policy.decide import CONFIGS, REASON_CODES
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH, load_catalogue

REPORT_SPLITS = ("test", "all")
FAILURE_KEYS = ["questions", "generations", "parse_error", "degenerate_output",
                "degenerate_greedy", "questions_with_degenerate"]  # fmt: skip
DIFFICULTIES = ("easy", "medium", "hard")
RS_COLUMNS = [f"rs_{penalty}" for penalty in metrics.PENALTIES]


def _fmt(value: Any) -> Any:
    """4 decimals for fractions and scores; empty cell for 'not defined'."""
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def in_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return records if split == "all" else [r for r in records if r.get("split") == split]


def by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(record["model_key"], []).append(record)
    return out


def build_tables(
    records: list[dict[str, Any]], low_cardinality: LowCardinality
) -> tuple[dict[str, tuple[list[str], list[tuple]]], list[dict[str, Any]], dict[str, float]]:
    """All tables as {file name: (header, rows)}, the failure lines, and tau per model."""
    models = by_model(records)
    tables: dict[str, tuple[list[str], list[tuple]]] = {}

    # tau per model, from that model's dev records only.
    taus: dict[str, float] = {}
    tau_rows = []
    for model, model_records in models.items():
        tau, rows = metrics.choose_tau(in_split(model_records, "dev"))
        taus[model] = tau
        for row in rows:
            tau_rows.append((model, row["tau"], row["n"], _fmt(row["coverage"]),
                             _fmt(row["risk"]), *(_fmt(row[c]) for c in RS_COLUMNS),
                             "yes" if row["tau"] == tau else ""))  # fmt: skip
    tables["tau_selection.csv"] = (
        ["model", "tau", "dev_n", "coverage", "risk", *RS_COLUMNS, "chosen"], tau_rows)  # fmt: skip

    e1, e2, e3, auroc_rows, rc_rows, confusion = [], [], [], [], [], []
    reason_rows, failure_rows = [], []
    failures: dict[tuple[str, str], dict[str, Any]] = {}
    for split in REPORT_SPLITS:
        for model, model_records in models.items():
            subset = in_split(model_records, split)
            scored = {config: score_records(subset, config, taus[model]) for config in CONFIGS}

            for group in (*DIFFICULTIES, "all"):
                ex = []
                for config in CONFIGS:
                    cases = [c for c in scored[config] if c.answerable
                             and (group == "all" or c.group == group)]  # fmt: skip
                    ex.append(_fmt(metrics.summarise(cases)["ex"]))
                n = sum(r["answerable"] and (group == "all" or r["group"] == group)
                        for r in subset)  # fmt: skip
                e1.append((split, model, group, n, *ex))

            for config in CONFIGS:
                cases = scored[config]
                summary = metrics.summarise(cases)
                scores = [_fmt(summary[column]) for column in RS_COLUMNS]
                e2.append((split, model, config, taus[model] if config == "C4" else "",
                           summary["n"], _fmt(summary["coverage"]), _fmt(summary["risk"]),
                           _fmt(summary["ex"]), *scores))  # fmt: skip
                counts = metrics.region_counts(cases)
                confusion.append((split, model, config, len(cases),
                                  *(counts[region] for region in REGIONS)))  # fmt: skip

                labels = dict.fromkeys(LABELS, 0)
                records_by_id = {r["question_id"]: r for r in subset}
                for case in cases:
                    if case.region not in HARMFUL:
                        continue
                    record = records_by_id[case.question_id]
                    label = label_case(record, case, low_cardinality)
                    labels[label] += 1
                    if split == "all":  # one failure line per model x question
                        line = failures.setdefault((model, case.question_id), {
                            "model": model, "question_id": case.question_id,
                            "group": case.group, "split": case.split,
                            "question": record["question"], "configs": {},
                        })  # fmt: skip
                        _, v2 = verifier_outputs(record, config)
                        line["configs"][config] = {
                            "label": label, "sql": case.sql,
                            "error": (v2 or {}).get("error"),
                        }  # fmt: skip
                e3.append((split, model, config, sum(labels.values()),
                           *(labels[label] for label in LABELS)))  # fmt: skip
                reasons = metrics.abstain_reasons(cases, REASON_CODES)
                reason_rows.append((split, model, config, sum(reasons.values()),
                                    *(reasons[code] for code in REASON_CODES)))  # fmt: skip

            failed = metrics.generation_failures(subset)
            failure_rows.append((split, model, *(failed[key] for key in FAILURE_KEYS)))

            for row in metrics.signal_table(subset):
                auroc_rows.append((split, model, row["signal"], row["n"], row["n_wrong"],
                                   _fmt(row["auroc"])))  # fmt: skip
            for row in metrics.risk_coverage(subset):
                rc_rows.append((split, model, row["tau"], row["n"], row["answered"],
                                _fmt(row["coverage"]), _fmt(row["risk"]),
                                *(_fmt(row[c]) for c in RS_COLUMNS)))  # fmt: skip

    cost_keys = ["n_questions", "calls_per_question", "latency_p50_ms", "latency_p95_ms",
                 "c1_latency_per_question_ms", "c4_latency_per_question_ms",
                 "prompt_tokens_per_question", "completion_tokens_per_question"]  # fmt: skip
    cost_rows = []
    for model, model_records in models.items():
        stats = metrics.latency_cost(model_records)
        cost_rows.append((model, model_records[0]["model"],
                          *(_fmt(round(stats[k], 1)) if isinstance(stats.get(k), float)
                            else _fmt(stats.get(k)) for k in cost_keys)))  # fmt: skip

    ex_columns = [f"ex_{config}" for config in CONFIGS]
    tables["e1_ex_by_model_difficulty.csv"] = (
        ["split", "model", "difficulty", "n_answerable", *ex_columns], e1)
    tables["e2_rs_by_config_penalty.csv"] = (
        ["split", "model", "config", "tau", "n", "coverage", "risk", "ex", *RS_COLUMNS], e2)
    tables["e3_taxonomy.csv"] = (["split", "model", "config", "n_harmful", *LABELS], e3)
    tables["e4_signal_auroc.csv"] = (
        ["split", "model", "signal", "n_answered", "n_wrong", "auroc"], auroc_rows)
    tables["e4_risk_coverage.csv"] = (
        ["split", "model", "tau", "n", "answered", "coverage", "risk", *RS_COLUMNS], rc_rows)
    tables["e5_latency_cost.csv"] = (["model_key", "model", *cost_keys], cost_rows)
    tables["outcomes_confusion.csv"] = (["split", "model", "config", "n", *REGIONS], confusion)
    tables["abstain_reasons.csv"] = (
        ["split", "model", "config", "n_abstained", *REASON_CODES], reason_rows)
    tables["e3_generation_failures.csv"] = (["split", "model", *FAILURE_KEYS], failure_rows)
    return tables, list(failures.values()), taus


def load_low_cardinality(json_path: Path, catalogue_path: Path) -> LowCardinality:
    """Distinct values per low-cardinality column, without the DOUBLE (amount) columns."""
    raw = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    skip = {
        (table.name.lower(), column.upper())
        for table in load_catalogue(catalogue_path).tables
        for column, col_type in table.column_types().items()
        if col_type.upper() == "DOUBLE"
    }
    return normalise_low_cardinality(raw, skip)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the report tables for one run.")
    parser.add_argument("--run-id", default="main")
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_TABLES_DIR)
    parser.add_argument("--low-cardinality", type=Path,
                        default=RESULTS_TABLES_DIR / "low_cardinality_values.json")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument("--resamples", type=int,
                        help="bootstrap resamples (default: experiment.yaml)")
    parser.add_argument("--seed", type=int, help="bootstrap seed (default: experiment.yaml)")
    args = parser.parse_args(argv)

    run_path = args.runs_dir / f"{args.run_id}.jsonl"
    records = read_run(run_path)
    if not records:
        print(f"No records in {run_path}. Run the verify stage first.", file=sys.stderr)
        return 1

    low_cardinality = load_low_cardinality(args.low_cardinality, args.catalogue)
    tables, failures, taus = build_tables(records, low_cardinality)
    settings = load_experiment().metrics
    tables.update(build_stats_tables(
        records, taus, load_models().models,
        resamples=args.resamples or settings.bootstrap_resamples,
        seed=settings.bootstrap_seed if args.seed is None else args.seed,
    ))  # fmt: skip
    for name, (header, rows) in tables.items():
        write_csv(args.out_dir / name, header, rows)
    failures_path = args.out_dir / "failures.jsonl"
    failures_path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in failures),
        encoding="utf-8", newline="\n",
    )  # fmt: skip

    models = by_model(records)
    print(f"{len(records)} record(s), {len(models)} model(s) from {run_path}")
    for model, model_records in models.items():
        dev = len(in_split(model_records, "dev"))
        note = "" if dev else "  (no dev records: fallback tau)"
        print(f"  {model}: {len(model_records)} questions, tau = {taus[model]}{note}")
    print(f"Wrote {len(tables)} tables and failures.jsonl ({len(failures)} lines) "
          f"to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
