"""Run the benchmark in two stages that share the LLM disk cache.

    generate   Ask every model every question (7 calls each). Needs only
               questions.jsonl and the catalogue - no database - so it can run
               on a Kaggle GPU. All answers land in the cache.
    verify     Re-read those answers from the cache (it makes NO model calls and
               fails loudly if one is missing), run V1/V2/V3 against the
               database, and append one JSON line per question x model to
               results/runs/<run_id>.jsonl in the shape decide() expects.
    all        generate, then verify (for small local runs).

To move work from Kaggle to the laptop, copy results/cache/. The cache key
includes the prompt text, so both machines must be on the same commit.

The generate stage must run where there is no database and possibly an older
sqlglot/duckdb, so this module imports the database and verifier code only
inside the verify stage.

Both stages are resumable: generate skips whatever is cached, verify skips
question x model pairs already in the run file.

Usage:
    python -m claimssql.eval.run_benchmark --stage generate --run-id full
    python -m claimssql.eval.run_benchmark --stage verify   --run-id full
    python -m claimssql.eval.run_benchmark --stage all --run-id smoke --models qwen \\
        --split dev --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claimssql.config import (
    CACHE_DIR,
    REPO_ROOT,
    RESULTS_RUNS_DIR,
    ExperimentConfig,
    ModelConfig,
    load_experiment,
    load_models,
)
from claimssql.generate.generator import DEGENERATE_OUTPUT, Generation, generate_all
from claimssql.generate.prompts import PROMPT_VERSION
from claimssql.llm.base import CacheMiss, LLMError, RateLimitStop
from claimssql.llm.factory import make_client
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH, load_catalogue
from claimssql.schema.context_builder import render_context

if TYPE_CHECKING:  # names used only in type hints: no import at run time
    import duckdb

    from claimssql.verify.schema_check import Schema

DEFAULT_QUESTIONS = REPO_ROOT / "benchmark" / "questions.jsonl"
GROUPS = ("easy", "medium", "hard", "T1", "T2", "T3", "T4")
EXIT_OK, EXIT_ERRORS, EXIT_RATE_LIMIT = 0, 1, 2


# ------------------------------------------------------------- selection ----


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]  # fmt: skip


def select_questions(
    questions: list[dict[str, Any]],
    groups: list[str] | None = None,
    split: str = "all",
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter in file order: by id, group and split, then keep the first `limit`."""
    selected = [
        q for q in questions
        if (not ids or q["id"] in ids)
        and (not groups or q["group"] in groups)
        and (split == "all" or q.get("split") == split)
    ]  # fmt: skip
    return selected[:limit] if limit else selected


def select_models(models: list[ModelConfig], keys: list[str] | None) -> list[ModelConfig]:
    """The named models, or by default every model marked use: benchmark."""
    if not keys:
        return [m for m in models if m.use == "benchmark"]
    known = {m.key: m for m in models}
    if unknown := sorted(set(keys) - set(known)):
        raise ValueError(f"not in models.yaml: {', '.join(unknown)}")
    return [known[key] for key in keys]


# -------------------------------------------------------------- generate ----


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def stage_generate(
    questions: list[dict[str, Any]],
    models: list[ModelConfig],
    context: str,
    experiment: ExperimentConfig,
    cache_dir: Path,
    summary_path: Path | None = None,
) -> int:
    """Fill the cache. If `summary_path` is given, also write per-model counts
    there as JSON (the Kaggle notebook prints that file)."""
    exit_code = EXIT_OK
    summary: dict[str, dict[str, Any]] = {}
    # Model by model: a server loads each model once instead of swapping per question.
    for model in models:
        client = make_client(model, PROMPT_VERSION, cache_dir=cache_dir)
        total, calls, live_calls, live_seconds = len(questions), 0, 0, 0.0
        live_questions, live_wall = 0, 0.0
        done, status, degenerate = 0, "complete", 0
        print(f"[{model.key}] generate: {total} question(s), {model.model}", flush=True)
        for index, question in enumerate(questions, start=1):
            started = time.perf_counter()
            try:
                generations = generate_all(
                    question["question"], client, context, experiment.generation
                )
            except RateLimitStop as exc:
                print(f"[{model.key}] STOPPED at {question['id']} ({index - 1}/{total} done): "
                      f"{exc}\n[{model.key}] Progress is in the cache. Run the same command "
                      "again later to continue.", flush=True)
                exit_code = EXIT_RATE_LIMIT
                status = "stopped by rate limit"
                break
            except LLMError as exc:
                print(f"[{model.key}] {index}/{total} {question['id']}: FAILED - {exc}",
                      flush=True)
                exit_code = max(exit_code, EXIT_ERRORS)
                status = "finished with failed questions"
                continue
            done += 1
            wall = time.perf_counter() - started
            new_calls = [a for g in generations for a in g.attempts if not a.cached]
            degenerate += sum(g.error == DEGENERATE_OUTPUT for g in generations)
            calls += sum(len(g.attempts) for g in generations)
            live_calls += len(new_calls)
            live_seconds += sum(a.latency_ms for a in new_calls) / 1000
            if new_calls:
                live_questions += 1
                live_wall += wall
            # ETA from the questions that needed real calls (cached ones take no time).
            per_question = live_wall / live_questions if live_questions else 0.0
            per_call = f"{live_seconds / live_calls:.1f} s/call" if live_calls else "all cached"
            print(f"[{model.key}] {index}/{total} {question['id']} | calls {calls} "
                  f"(new {live_calls}, cached {calls - live_calls}) | {per_call} | "
                  f"ETA {_clock(per_question * (total - index))}", flush=True)
        print(f"[{model.key}] cache hits {client.cache.hits}, misses {client.cache.misses}",
              flush=True)
        summary[model.key] = {
            "model": model.model,
            "status": status,
            "questions": total,
            "questions_done": done,
            "calls": calls,
            "new_calls": live_calls,
            "cache_hits": client.cache.hits,
            "seconds_per_new_call": round(live_seconds / live_calls, 2) if live_calls else None,
            "degenerate_outputs": degenerate,  # generations where the model looped
        }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return exit_code


# ---------------------------------------------------------------- verify ----


def _sql_of(generation: Generation) -> str | None:
    parsed = generation.parsed
    if generation.error or not parsed or not parsed.get("answerable"):
        return None
    sql = parsed.get("sql")
    return sql if sql and sql.strip() else None


def build_record(
    run_id: str,
    question: dict[str, Any],
    model: ModelConfig,
    generations: list[Generation],
    con: duckdb.DuckDBPyConnection,
    schema: Schema,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    """One run-file line: the generations plus every verifier output."""
    # Imported here, not at the top: only the verify stage needs sqlglot/duckdb.
    from claimssql.verify.dry_run import dry_run
    from claimssql.verify.schema_check import check_sql
    from claimssql.verify.self_consistency import self_consistency

    forced, abstain, samples = generations[0], generations[1], generations[2:]
    timeout = experiment.verify.dry_run_timeout_s
    max_rows = experiment.verify.result_row_cap

    def verify_greedy(generation: Generation) -> tuple[dict | None, dict | None]:
        sql = _sql_of(generation)
        if sql is None:
            return None, None
        return (check_sql(sql, schema).to_dict(),
                dry_run(con, sql, timeout_s=timeout, max_rows=max_rows).to_dict())

    forced_v1, forced_v2 = verify_greedy(forced)  # needed to score C0
    v1, v2 = verify_greedy(abstain)
    v3 = self_consistency(
        con,
        [_sql_of(sample) for sample in samples],
        greedy_hash=v2["result_hash"] if v2 else None,
        timeout_s=timeout,
        max_rows=max_rows,
        sample_errors=[sample.error for sample in samples],
    ).to_dict()

    return {
        "run_id": run_id,
        "question_id": question["id"],
        "question": question["question"],
        "group": question["group"],
        "answerable": question["answerable"],
        "split": question.get("split"),
        "gold_result_hash": question.get("gold_result_hash"),
        "model_key": model.key,
        "provider": model.provider,
        "model": model.model,
        "model_options": model.options,
        "prompt_version": PROMPT_VERSION,
        "forced": forced.to_dict(),
        "abstain": abstain.to_dict(),
        "samples": [sample.to_dict() for sample in samples],
        "forced_v1": forced_v1,
        "forced_v2": forced_v2,
        "v1": v1,
        "v2": v2,
        "v3": v3,
    }


def read_run(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]  # fmt: skip


def stage_verify(
    run_id: str,
    questions: list[dict[str, Any]],
    models: list[ModelConfig],
    context: str,
    experiment: ExperimentConfig,
    cache_dir: Path,
    con: duckdb.DuckDBPyConnection,
    schema: Schema,
    run_path: Path,
) -> int:
    done = {(r["question_id"], r["model_key"]) for r in read_run(run_path)}
    run_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with run_path.open("a", encoding="utf-8", newline="\n") as out:
        for model in models:
            # cache_only: this stage must not call a model. A missing answer
            # means generate has not finished (or the cache was not copied).
            client = make_client(model, PROMPT_VERSION, cache_dir=cache_dir, cache_only=True)
            for index, question in enumerate(questions, start=1):
                if (question["id"], model.key) in done:
                    skipped += 1
                    continue
                try:
                    generations = generate_all(
                        question["question"], client, context, experiment.generation
                    )
                except CacheMiss as exc:
                    print(f"\nERROR: generations missing for question {question['id']} on "
                          f"model '{model.key}': {exc}\nRun --stage generate for this model "
                          "first, or copy its results/cache/ here (same commit).",
                          file=sys.stderr)
                    return EXIT_ERRORS
                record = build_record(run_id, question, model, generations, con, schema,
                                      experiment)  # fmt: skip
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()  # one complete line per pair: safe to interrupt and resume
                written += 1
                print(f"[{model.key}] verify {index}/{len(questions)} {question['id']}",
                      flush=True)
    print(f"verify: wrote {written} record(s), skipped {skipped} already in {run_path}")
    return EXIT_OK


# ------------------------------------------------------------------- CLI ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the benchmark: generate (LLM calls) and/or verify (database)."
    )
    parser.add_argument("--stage", choices=["generate", "verify", "all"], required=True)
    parser.add_argument("--run-id", default="main", help="names results/runs/<run-id>.jsonl")
    parser.add_argument("--models", nargs="+",
                        help="model keys (default: every model with use: benchmark)")
    parser.add_argument("--groups", nargs="+", choices=GROUPS)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--ids", nargs="+", help="only these question ids")
    parser.add_argument("--limit", type=int, help="first N questions after filtering")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--db-path", type=Path, help="default: data/claims.duckdb")
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_RUNS_DIR)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    experiment = load_experiment()
    try:
        models = select_models(load_models().models, args.models)
    except ValueError as exc:
        parser.error(str(exc))
    questions = select_questions(
        load_questions(args.questions), args.groups, args.split, args.ids, args.limit
    )
    if not questions or not models:
        print("Nothing to do: no questions or no models selected.", file=sys.stderr)
        return EXIT_ERRORS
    catalogue = load_catalogue(args.catalogue)
    context = render_context(catalogue)
    print(f"{len(questions)} question(s) x {len(models)} model(s) "
          f"[{', '.join(m.key for m in models)}], prompt {PROMPT_VERSION}", flush=True)

    exit_code = EXIT_OK
    if args.stage in ("generate", "all"):
        exit_code = stage_generate(
            questions, models, context, experiment, args.cache_dir,
            summary_path=args.runs_dir / f"{args.run_id}.generate.json",
        )  # fmt: skip
        if exit_code != EXIT_OK and args.stage == "all":
            print("generate did not finish; skipping verify.", file=sys.stderr)
            return exit_code
    if args.stage in ("verify", "all"):
        from claimssql.db.connection import get_connection
        from claimssql.verify.schema_check import schema_from_catalogue

        con = get_connection(read_only=True, db_path=args.db_path)
        try:
            exit_code = stage_verify(
                args.run_id, questions, models, context, experiment, args.cache_dir, con,
                schema_from_catalogue(catalogue), args.runs_dir / f"{args.run_id}.jsonl",
            )  # fmt: skip
        finally:
            con.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
