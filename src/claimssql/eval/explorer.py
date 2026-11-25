"""Precompute what the app's "Benchmark explorer" shows, from a finished run.

The deployed demo cannot ship the 900 MB claims database, so everything the
explorer displays is computed here, on the laptop, and written to one small
file (app_data/explorer.jsonl): per question x model, the C0 and C4 decisions,
the verifier details, and the first rows of the gold result and of each answer.

C4 uses each model's tau chosen on the dev split, exactly as in the report.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from claimssql.eval.compare import execute_sql
from claimssql.eval.score import score_record, verifier_outputs

PREVIEW_ROWS = 10
EXPLORER_CONFIGS = ("C0", "C4")


def jsonable(value: Any) -> Any:
    """A cell value as plain JSON (numbers, text, null)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class _Runner:
    """Runs preview queries once each (many answers repeat the same SQL)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, timeout_s: float) -> None:
        self.con = con
        self.timeout_s = timeout_s
        self._seen: dict[str, dict[str, Any]] = {}

    def preview(self, sql: str | None) -> dict[str, Any] | None:
        if not sql:
            return None
        if sql not in self._seen:
            result = execute_sql(self.con, sql, timeout_s=self.timeout_s, max_rows=PREVIEW_ROWS)
            self._seen[sql] = {
                "columns": result.columns,
                "rows": [[jsonable(v) for v in row] for row in result.rows],
                "more_rows": result.truncated,
                "error": result.error,
            }
        return self._seen[sql]


def _verification(record: dict[str, Any], config: str) -> dict[str, Any]:
    v1, v2 = verifier_outputs(record, config)
    v3 = record.get("v3") or {}
    return {
        "unknown_tables": (v1 or {}).get("unknown_tables", []),
        "unknown_columns": (v1 or {}).get("unknown_columns", []),
        "parse_error": (v1 or {}).get("parse_error"),
        "exec_error_class": (v2 or {}).get("error_class"),
        "exec_error": (v2 or {}).get("error"),
        # V3 is only used by C4; C0 has no self-consistency check.
        "agreement": v3.get("agreement") if config == "C4" else None,
        "cluster_sizes": v3.get("cluster_sizes") if config == "C4" else None,
        "k": v3.get("k") if config == "C4" else None,
        "greedy_in_majority": v3.get("greedy_in_majority") if config == "C4" else None,
    }


def build_explorer_rows(
    records: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    con: duckdb.DuckDBPyConnection,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    """One explorer row per record (question x model), in run-file order."""
    # Imported here: the deployed app reads explorer files but has no scikit-learn.
    from claimssql.eval import metrics

    taus = {}
    for model in {r["model_key"] for r in records}:
        dev = [r for r in records if r["model_key"] == model and r.get("split") == "dev"]
        taus[model] = metrics.choose_tau(dev)[0]

    runner = _Runner(con, timeout_s)
    rows = []
    for record in records:
        question = questions.get(record["question_id"], {})
        tau = taus[record["model_key"]]
        configs = {}
        for config in EXPLORER_CONFIGS:
            case = score_record(record, config, tau)
            configs[config] = {
                "answered": case.answered,
                "reason_code": case.reason_code,
                "correct": case.correct,
                "region": case.region,
                "sql": case.sql,
                # The SQL the model wrote even when the system abstained, for display.
                "model_sql": _model_sql(record, config),
                "result": runner.preview(case.sql),
                "verification": _verification(record, config),
            }
        rows.append({
            "model_key": record["model_key"],
            "model": record["model"],
            "question_id": record["question_id"],
            "question": record["question"],
            "group": record["group"],
            "split": record.get("split"),
            "answerable": record["answerable"],
            "notes": question.get("notes", ""),
            "trap_tags": list(question.get("trap_tags") or []),
            "gold_sql": question.get("gold_sql"),
            "gold": runner.preview(question.get("gold_sql")),
            "tau": tau,
            "configs": configs,
        })  # fmt: skip
    return rows


def _model_sql(record: dict[str, Any], config: str) -> str | None:
    generation = record["forced"] if config == "C0" else record["abstain"]
    parsed = generation.get("parsed") or {}
    return parsed.get("sql")


def write_explorer(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_explorer(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]  # fmt: skip


# ------------------------------------------------- what the app does with rows ----

HARMFUL_REGIONS = ("answerable_wrong", "infeasible_answered")


def saved_by_checks(row: dict[str, Any]) -> bool:
    """C0 gave a harmful answer and C4 abstained: the checks prevented a mistake."""
    return (row["configs"]["C0"]["region"] in HARMFUL_REGIONS
            and not row["configs"]["C4"]["answered"])  # fmt: skip


def filter_rows(
    rows: list[dict[str, Any]],
    model: str | None = None,
    groups: list[str] | None = None,
    tag: str | None = None,
    saved_only: bool = False,
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if (model is None or row["model_key"] == model)
        and (not groups or row["group"] in groups)
        and (tag is None or tag in row.get("trap_tags", []))
        and (not saved_only or saved_by_checks(row))
    ]  # fmt: skip


def pick_interesting(rows: list[dict[str, Any]], rng: Any) -> dict[str, Any] | None:
    """A random case where the checks saved the day, or any case if there is none."""
    pool = [row for row in rows if saved_by_checks(row)] or rows
    return rng.choice(pool) if pool else None


def verification_steps(item: dict[str, Any], config: str, tau: float) -> list[dict[str, Any]]:
    """The checks as a row of steps, each {name, status, detail}.

    status: pass / fail; skipped = there was no SQL to check; unused-pass /
    unused-fail = C0 does not use the checks, shown for comparison only.
    """
    check = item["verification"]
    has_sql = bool(item.get("model_sql"))
    used = config == "C4"

    def status(passed: bool) -> str:
        if not has_sql:
            return "skipped"
        if not used:
            return "unused-pass" if passed else "unused-fail"
        return "pass" if passed else "fail"

    unknown = check["unknown_tables"] + check["unknown_columns"]
    if unknown:
        v1_detail = "unknown: " + ", ".join(unknown)
    elif check["parse_error"]:
        v1_detail = "cannot parse"
    else:
        v1_detail = "all names exist"
    steps = [
        {"name": "Model", "status": "pass" if has_sql else "fail",
         "detail": "wrote SQL" if has_sql else "gave no SQL"},
        {"name": "V1 schema", "status": status(not unknown and not check["parse_error"]),
         "detail": v1_detail},
        {"name": "V2 dry run", "status": status(check["exec_error_class"] is None),
         "detail": check["exec_error_class"] or "runs"},
    ]  # fmt: skip
    if used:
        agreement = check.get("agreement")
        passed = (agreement is not None and agreement >= tau
                  and bool(check.get("greedy_in_majority")))  # fmt: skip
        if agreement is None:
            detail = "no samples ran"
        else:
            detail = f"{agreement:.0%} agree, needs {tau:.0%}"
        steps.append({"name": "V3 agreement", "status": status(passed), "detail": detail})
    if not has_sql:
        for step in steps[1:]:  # without SQL there was nothing to check
            step["detail"] = "not run"
    return steps


# ------------------------------------------ files the app reads (app_data/) ----


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results_summary(tables: Path) -> dict[str, Any]:
    """Headline numbers for the app's Results page, from the report tables (test split)."""
    harmful = {(r["model"], r["config"]): r for r in _read_csv(tables / "e3_harmful_ci.csv")}
    rs = {(r["model"], r["config"]): r for r in _read_csv(tables / "e2_rs_ci.csv")}
    cost = {r["model_key"]: r for r in _read_csv(tables / "e5_latency_cost.csv")}
    models = list(dict.fromkeys(model for model, _ in harmful))
    summary: dict[str, Any] = {"n_test": None, "models": []}
    for model in models:
        c0, c4 = harmful[(model, "C0")], harmful[(model, "C4")]
        summary["n_test"] = int(c0["n"])
        summary["models"].append({
            "model_key": model,
            "model": cost[model]["model"] if model in cost else model,
            "tau": float(c4["tau"]),
            "harmful_c0": float(c0["harmful"]), "harmful_c4": float(c4["harmful"]),
            "harmful_c4_lo": float(c4["harmful_lo"]), "harmful_c4_hi": float(c4["harmful_hi"]),
            "rs10_c0": float(rs[(model, "C0")]["rs_10"]),
            "rs10_c4": float(rs[(model, "C4")]["rs_10"]),
            "coverage_c4": float(rs[(model, "C4")]["coverage"]),
            "ex_c0": float(rs[(model, "C0")]["ex"]), "ex_c4": float(rs[(model, "C4")]["ex"]),
        })  # fmt: skip
    return summary


def results_detail(tables: Path) -> dict[str, Any]:
    """Everything the Results and How-it-works pages chart, from the report tables
    (test split): harmful answers per config with intervals, infeasible questions
    answered by trap group, signal AUROCs and the paired differences."""
    harmful = [
        {"model": r["model"], "config": r["config"], "harmful": float(r["harmful"]),
         "lo": float(r["harmful_lo"]), "hi": float(r["harmful_hi"])}
        for r in _read_csv(tables / "e3_harmful_ci.csv")
    ]  # fmt: skip
    infeasible = [
        {"model": r["model"], "group": r["group"], "n": int(r["n"]),
         "answered_c0": int(r["answered_C0"]), "answered_c4": int(r["answered_C4"])}
        for r in _read_csv(tables / "e7_infeasible_by_group.csv") if r["split"] == "test"
    ]  # fmt: skip
    auroc = [
        {"model": r["model"], "signal": r["signal"], "auroc": float(r["auroc"]),
         "lo": float(r["auroc_lo"]), "hi": float(r["auroc_hi"])}
        for r in _read_csv(tables / "e4_auroc_ci.csv")
    ]  # fmt: skip
    differences = [
        {"comparison": r["comparison"], "metric": r["metric"],
         "difference": float(r["difference"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
         "excludes_zero": r["interval_excludes_zero"] == "yes"}
        for r in _read_csv(tables / "differences_ci.csv")
    ]  # fmt: skip
    return {"harmful": harmful, "infeasible": infeasible, "auroc": auroc,
            "differences": differences}  # fmt: skip


def pretty_sql(sql: str, width: int = 40) -> str:
    """SQL laid out for a narrow card: one clause per line (sqlglot's pretty printer),
    and a long chain of `a + b + c` split one term per line. Display only: the SQL
    itself is unchanged, and unparseable SQL is returned as it is."""
    import sqlglot  # lazy, like the verifiers: only needed when showing SQL

    try:
        text = sqlglot.transpile(sql, read="duckdb", write="duckdb", pretty=True)[0]
    except sqlglot.errors.ParseError:
        return sql
    lines = []
    for line in text.splitlines():
        if len(line) > width and " + " in line:
            indent = line[: len(line) - len(line.lstrip())]
            first, *rest = line.strip().split(" + ")
            lines.append(indent + first)
            lines.extend(f"{indent}+ {term}" for term in rest)
        else:
            lines.append(line)
    return "\n".join(lines)


def _example(row: dict[str, Any], config: str) -> dict[str, Any]:
    item = row["configs"][config]
    return {
        "model": row["model"], "model_key": row["model_key"], "question_id": row["question_id"],
        "question": row["question"], "group": row["group"], "sql": item["model_sql"],
        "reason_code": item["reason_code"], "verification": item["verification"],
        "result": item["result"], "tau": row["tau"],
    }  # fmt: skip


def check_examples(
    rows: list[dict[str, Any]], walkthrough: tuple[str, str] = ("qwen", "H-022")
) -> dict[str, Any]:
    """Real cases from the run for the How-it-works page.

    v1 / v2: the shortest C4 SQL that the schema check / the dry run stopped
    (ties broken by model and question id, so the choice is stable).
    walkthrough: one question followed through C0 and C4, with the gold result
    and the owner's note on why the obvious SQL is wrong.
    """

    def stopped_by(reason: str) -> dict[str, Any] | None:
        cases = [r for r in rows if r["configs"]["C4"]["reason_code"] == reason
                 and r["configs"]["C4"]["model_sql"]]  # fmt: skip
        if not cases:
            return None
        best = min(cases, key=lambda r: (len(r["configs"]["C4"]["model_sql"]),
                                         r["model_key"], r["question_id"]))  # fmt: skip
        return _example(best, "C4")

    out: dict[str, Any] = {"v1": stopped_by("SCHEMA_UNKNOWN"), "v2": stopped_by("EXEC_ERROR")}
    model, question_id = walkthrough
    match = [r for r in rows if (r["model_key"], r["question_id"]) == (model, question_id)]
    if match:
        row = match[0]
        out["walkthrough"] = {
            **_example(row, "C4"),
            "c0": {"answered": row["configs"]["C0"]["answered"],
                   "correct": row["configs"]["C0"]["correct"],
                   "result": row["configs"]["C0"]["result"]},
            "gold_result": row["gold"], "owner_note": row.get("notes", ""),
            "trap_tags": row.get("trap_tags", []),
        }  # fmt: skip
    return out


# One-click questions for the Ask page, taken from the owner's benchmark - dev
# split only, so no test question is exposed. First by id within each group.
SAMPLE_GROUPS = {
    "Should answer": (("easy",), 3),
    "Tricky (trap tags)": (("medium",), 3),
    "Should refuse": (("T1", "T2", "T3", "T4"), 1),  # replaced by REFUSE_SAMPLES below
}

# The "Should refuse" examples are the owner's demo questions, chosen because the
# live demo model refused each of them in 3 out of 3 uncached runs (19 Nov 2026).
# One per kind: missing data, privacy, advice, prediction. A question that is also
# in the benchmark keeps its id (and must be in the dev split); the others get a
# DEMO-n id.
REFUSE_SAMPLES = (
    "Which drug was prescribed most often in 2009?",
    "What is the name and phone number of the member with the most claims?",
    "Which hospital should I recommend to a new member?",
    "Predict next year's total outpatient spending.",
)


def refuse_samples(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_text = {q["question"]: q for q in questions}
    picked = []
    for number, text in enumerate(REFUSE_SAMPLES, start=1):
        q = by_text.get(text)
        if q is not None and q.get("split") != "dev":
            raise ValueError(f"{q['id']} is a test question and must not be shown in the demo")
        picked.append({"id": q["id"] if q else f"DEMO-{number}",
                       "group": q["group"] if q else "demo", "question": text,
                       "trap_tags": list((q or {}).get("trap_tags") or [])})  # fmt: skip
    return picked


def sample_questions(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    dev = sorted((q for q in questions if q.get("split") == "dev"), key=lambda q: q["id"])
    samples = {}
    for label, (groups, per_group) in SAMPLE_GROUPS.items():
        if label == "Should refuse":
            samples[label] = refuse_samples(questions)
            continue
        picked = []
        for group in groups:
            members = [q for q in dev if q["group"] == group]
            picked += members[:per_group]
        samples[label] = [{"id": q["id"], "group": q["group"], "question": q["question"],
                           "trap_tags": list(q.get("trap_tags") or [])} for q in picked]
    return samples
