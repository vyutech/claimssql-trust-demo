"""Validate benchmark/questions.yaml and build benchmark/questions.jsonl from it.

questions.yaml is the file the owner edits; this tool only reads it.
questions.jsonl is generated: the same records in the project's record format,
plus the two machine-written fields (gold_result_hash, split).

What it does, in order:
  1. checks every record (fields, unique ids, group vs. sql, tags);
  2. runs each sql read-only with a timeout, computes gold_result_hash and saves
     the first rows to benchmark/gold_previews/<id>.csv;
  3. carries each record's split over from the existing questions.jsonl and
     assigns dev/test (stratified by group) only to records that have none;
  4. writes questions.jsonl in the YAML's order and prints a report.

It reports problems and never fixes them. Because a split must never change
once assigned, questions.jsonl is the memory for splits: keep it in git.

Usage:
    python -m claimssql.eval.validate_benchmark --no-split   # while writing questions
    python -m claimssql.eval.validate_benchmark              # final run, assigns splits
    python -m claimssql.eval.validate_benchmark --dry-run    # report only, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from claimssql.config import REPO_ROOT, RESULTS_TABLES_DIR, load_experiment
from claimssql.db.connection import DEFAULT_DB_PATH, get_connection
from claimssql.db.load_synpuf import write_csv
from claimssql.eval.compare import MAX_ROWS, execute_sql, result_hash

BENCHMARK_DIR = REPO_ROOT / "benchmark"
DEFAULT_QUESTIONS = BENCHMARK_DIR / "questions.yaml"
DEFAULT_OUTPUT = BENCHMARK_DIR / "questions.jsonl"
DEFAULT_TRAP_TAGS = BENCHMARK_DIR / "trap_tags.yaml"
DEFAULT_PREVIEWS = BENCHMARK_DIR / "gold_previews"

ANSWERABLE_GROUPS = ("easy", "medium", "hard")
INFEASIBLE_GROUPS = ("T1", "T2", "T3", "T4")
PREVIEW_ROWS = 20

Group = Literal["easy", "medium", "hard", "T1", "T2", "T3", "T4"]


class YamlQuestion(BaseModel):
    """One record as the owner writes it in questions.yaml."""

    # extra="forbid": a misspelt field name is an error, not silently dropped.
    model_config = ConfigDict(extra="forbid")

    # The id becomes a file name in gold_previews/, hence the restricted characters.
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    group: Group
    question: str
    sql: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> YamlQuestion:
        if not self.question.strip():
            raise ValueError("question is empty")
        has_sql = bool(self.sql and self.sql.strip())
        if self.group in ANSWERABLE_GROUPS and not has_sql:
            raise ValueError(f"group {self.group!r} is answerable but the record has no sql")
        if self.group in INFEASIBLE_GROUPS and self.sql is not None:
            raise ValueError(f"group {self.group!r} is infeasible, so it must have no sql")
        return self


class QuestionRecord(BaseModel):
    """One line of questions.jsonl - the record format the rest of the pipeline reads."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    group: Group
    answerable: bool
    gold_sql: str | None = None
    gold_result_hash: str | None = None
    trap_tags: list[str] = Field(default_factory=list)
    notes: str = ""
    split: Literal["dev", "test"] | None = None


def to_record(source: YamlQuestion) -> QuestionRecord:
    """YAML record -> JSONL record. Only surrounding whitespace is removed
    (a YAML block always ends in a newline); the text itself is not touched."""
    return QuestionRecord(
        id=source.id,
        question=source.question.strip(),
        group=source.group,
        answerable=source.group in ANSWERABLE_GROUPS,
        gold_sql=source.sql.strip() if source.sql else None,
        trap_tags=list(source.tags),
        notes=(source.notes or "").strip(),
    )


@dataclass
class Entry:
    """One record of questions.yaml plus what the validator found out about it."""

    number: int  # position in the file, starting at 1
    line: int  # line in questions.yaml where the record starts
    raw_id: str | None = None  # the id as written, even if the record is invalid
    record: QuestionRecord | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gold_error: str | None = None

    @property
    def label(self) -> str:
        return self.raw_id or f"record {self.number}"


class BenchmarkFileError(Exception):
    """questions.yaml as a whole cannot be read (bad YAML, or not a list)."""


def load_trap_tags(path: Path = DEFAULT_TRAP_TAGS) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(tag): str(meaning) for tag, meaning in data.items()}


def _pydantic_errors(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        message = err["msg"].removeprefix("Value error, ")
        out.append(f"{where}: {message}" if where else message)
    return out


def parse_yaml(text: str, allowed_tags: set[str]) -> list[Entry]:
    """Check every record of questions.yaml. Invalid ones keep record=None."""
    loader = yaml.SafeLoader(text)
    try:
        # Working on nodes (not plain objects) gives each record's line number,
        # so an error can point at the place to fix.
        root = loader.get_single_node()
        if root is None:
            return []  # empty file, or comments only
        if not isinstance(root, yaml.SequenceNode):
            raise BenchmarkFileError(
                "questions.yaml must be a list of records (each starting with '- id: ...')"
            )
        items = [
            (node.start_mark.line + 1, loader.construct_object(node, deep=True))
            for node in root.value
        ]
    except yaml.YAMLError as exc:
        raise BenchmarkFileError(f"questions.yaml is not valid YAML: {exc}") from exc
    finally:
        loader.dispose()

    entries = []
    for number, (line, data) in enumerate(items, start=1):
        entry = Entry(number=number, line=line)
        entries.append(entry)
        if not isinstance(data, dict):
            entry.errors.append("not a record (expected fields like 'id: ...')")
            continue
        if isinstance(data.get("id"), str):
            entry.raw_id = data["id"]
        try:
            source = YamlQuestion.model_validate(data)
        except ValidationError as exc:
            entry.errors += _pydantic_errors(exc)
            continue
        if unknown := sorted(set(source.tags) - allowed_tags):
            entry.errors.append(f"unknown tags: {', '.join(unknown)}")
            continue
        entry.record = to_record(source)
        if not entry.record.answerable and not entry.record.notes:
            entry.warnings.append("infeasible record has no reason in notes")

    # A duplicated id makes every record that uses it invalid: we cannot know
    # which one is meant, and previews, hashes and splits are keyed by id.
    counts = Counter(e.raw_id for e in entries if e.raw_id)
    for entry in entries:
        if entry.raw_id and counts[entry.raw_id] > 1:
            entry.errors.append(f"duplicate id {entry.raw_id!r}")
            entry.record = None
    return entries


Previous = dict[str, tuple[str, QuestionRecord]]


def load_previous(text: str) -> Previous:
    """id -> (original line, record) from the existing questions.jsonl."""
    previous: Previous = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            record = QuestionRecord.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            continue  # an unreadable old line carries no split we could trust
        previous[record.id] = (raw, record)
    return previous


def apply_previous(entries: list[Entry], previous: Previous) -> None:
    """Carry split (always) and the old hash (if the SQL is unchanged) forward."""
    for entry in entries:
        if entry.record is None or entry.record.id not in previous:
            continue
        old = previous[entry.record.id][1]
        update: dict = {"split": old.split}
        if old.gold_sql == entry.record.gold_sql:
            # Kept only so run_gold_sql can notice a result that changed.
            update["gold_result_hash"] = old.gold_result_hash
        entry.record = entry.record.model_copy(update=update)


def write_preview(path: Path, columns: list[str], rows: list[tuple]) -> None:
    write_csv(path, columns, [tuple(row) for row in rows[:PREVIEW_ROWS]])


def run_gold_sql(
    entries: list[Entry],
    con: duckdb.DuckDBPyConnection,
    timeout_s: float,
    max_rows: int = MAX_ROWS,
    preview_dir: Path | None = None,
) -> None:
    """Execute each valid record's SQL; set the hash, collect warnings and failures."""
    for entry in entries:
        record = entry.record
        if record is None or not record.answerable:
            continue
        result = execute_sql(con, record.gold_sql, timeout_s=timeout_s, max_rows=max_rows)
        if not result.ok:
            entry.gold_error = result.error
            # No result means no hash: a stale one would be scored against.
            entry.record = record.model_copy(update={"gold_result_hash": None})
            continue

        if not result.rows:
            entry.warnings.append("gold result is empty (0 rows)")
        elif len(result.rows) == 1 and len(result.rows[0]) == 1 and result.rows[0][0] is None:
            entry.warnings.append("gold result is a single NULL")
        if result.truncated:
            entry.warnings.append(
                f"gold result has more than {max_rows:,} rows; hash covers a truncated result"
            )
        new_hash = result_hash(result.rows)
        if record.gold_result_hash and record.gold_result_hash != new_hash:
            entry.warnings.append("gold result changed since the last run (same SQL)")
        entry.record = record.model_copy(update={"gold_result_hash": new_hash})
        if preview_dir is not None:
            write_preview(preview_dir / f"{record.id}.csv", result.columns, result.rows)


def assign_splits(entries: list[Entry], seed: int, dev_fraction: float) -> int:
    """Give dev/test to valid records without a split. Returns how many were assigned.

    Stratified: inside each group, enough new records go to dev to bring the
    group to round(dev_fraction * group size). Existing splits are never
    touched. Deterministic for a given seed and set of ids.
    """
    assigned = 0
    by_group: dict[str, list[Entry]] = {}
    for entry in entries:
        if entry.record:
            by_group.setdefault(entry.record.group, []).append(entry)

    for group, members in sorted(by_group.items()):
        unassigned = sorted(
            (m for m in members if m.record.split is None), key=lambda m: m.record.id
        )
        if not unassigned:
            continue
        target_dev = round(dev_fraction * len(members))
        have_dev = sum(m.record.split == "dev" for m in members)
        need = min(max(target_dev - have_dev, 0), len(unassigned))
        # Seeded per group so adding questions to one group cannot reshuffle another.
        rng = random.Random(f"{seed}:{group}")
        dev_ids = {m.record.id for m in rng.sample(unassigned, need)}
        for member in unassigned:
            split = "dev" if member.record.id in dev_ids else "test"
            member.record = member.record.model_copy(update={"split": split})
            assigned += 1
    return assigned


def render_jsonl(entries: list[Entry], previous: Previous) -> tuple[str, list[str]]:
    """New questions.jsonl text, in YAML order, plus notes about what happened.

    A record that is invalid right now but was valid before keeps its previous
    line, so a typo in questions.yaml cannot throw away an assigned split.
    """
    out, notes = [], []
    for entry in entries:
        duplicate = any("duplicate id" in error for error in entry.errors)
        if entry.record is not None:
            out.append(json.dumps(entry.record.model_dump(), ensure_ascii=False))
        elif entry.raw_id in previous and not duplicate:
            out.append(previous[entry.raw_id][0])
            notes.append(f"{entry.raw_id}: invalid now; previous version kept in questions.jsonl")
    current_ids = {e.raw_id for e in entries if e.raw_id}
    for old_id, (_, old) in previous.items():
        if old_id not in current_ids:
            lost = f" (its split '{old.split}' is discarded)" if old.split else ""
            notes.append(
                f"{old_id}: no longer in questions.yaml; removed from questions.jsonl{lost}"
            )
    return "\n".join(out) + ("\n" if out else ""), notes


def write_atomically(path: Path, text: str) -> None:
    """Write to a temporary file, then swap it in, so a crash cannot corrupt the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def composition(entries: list[Entry], targets: dict[str, int]) -> list[tuple]:
    """Rows (group, target, count, difference, dev, test, no_split) plus a total row."""
    records = [entry.record for entry in entries if entry.record]
    rows = []
    for group, target in targets.items():
        members = [r for r in records if r.group == group]
        dev = sum(r.split == "dev" for r in members)
        test = sum(r.split == "test" for r in members)
        rows.append(
            (group, target, len(members), len(members) - target, dev, test,
             len(members) - dev - test)
        )  # fmt: skip
    totals = [sum(row[i] for row in rows) for i in range(1, 7)]
    return [*rows, ("total", *totals)]


COMPOSITION_HEADER = ["group", "target", "count", "difference", "dev", "test", "no_split"]


def print_report(entries: list[Entry], rows: list[tuple], notes: list[str]) -> None:
    print("Composition (valid records only)")
    print(f"  {'group':<8}{'target':>7}{'count':>7}{'diff':>6}{'dev':>5}{'test':>6}"
          f"{'no split':>10}")
    for group, target, count, diff, dev, test, no_split in rows:
        print(f"  {group:<8}{target:>7}{count:>7}{diff:>+6}{dev:>5}{test:>6}{no_split:>10}")

    invalid = [entry for entry in entries if entry.errors]
    failed = [entry for entry in entries if entry.gold_error]
    warned = [entry for entry in entries if entry.warnings]
    print(f"\nInvalid records: {len(invalid)}")
    for entry in invalid:
        for error in entry.errors:
            print(f"  {entry.label} (questions.yaml line {entry.line}): {error}")
    print(f"\nGold SQL failures: {len(failed)}")
    for entry in failed:
        print(f"  {entry.label} (questions.yaml line {entry.line}): {entry.gold_error}")
    print(f"\nWarnings: {sum(len(entry.warnings) for entry in warned) + len(notes)}")
    for entry in warned:
        for warning in entry.warnings:
            print(f"  {entry.label}: {warning}")
    for note in notes:
        print(f"  {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                        help="the owner-edited YAML file (only read, never written)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="the generated JSONL file")
    parser.add_argument("--trap-tags", type=Path, default=DEFAULT_TRAP_TAGS)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--previews-dir", type=Path, default=DEFAULT_PREVIEWS)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_TABLES_DIR)
    parser.add_argument("--timeout", type=float, help="seconds per gold query "
                        "(default: verify.dry_run_timeout_s in experiment.yaml)")
    parser.add_argument("--no-split", action="store_true",
                        help="do not assign dev/test yet (use while still writing questions)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only: no file is written")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # not available when output is captured
        sys.stdout.reconfigure(encoding="utf-8")
    experiment = load_experiment()
    try:
        entries = parse_yaml(
            args.questions.read_text(encoding="utf-8"), set(load_trap_tags(args.trap_tags))
        )
    except BenchmarkFileError as exc:
        print(f"ERROR: {exc}\nNothing was written.", file=sys.stderr)
        return 1

    old_text = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
    previous = load_previous(old_text)
    apply_previous(entries, previous)

    if any(entry.record and entry.record.answerable for entry in entries):
        con = get_connection(read_only=True, db_path=args.db_path)
        try:
            run_gold_sql(
                entries,
                con,
                timeout_s=args.timeout or experiment.verify.dry_run_timeout_s,
                max_rows=experiment.verify.result_row_cap,
                preview_dir=None if args.dry_run else args.previews_dir,
            )
        finally:
            con.close()

    assigned = 0
    if not args.no_split:
        assigned = assign_splits(
            entries, experiment.benchmark.split_seed, experiment.benchmark.dev_fraction
        )

    rows = composition(entries, experiment.benchmark.composition)
    new_text, notes = render_jsonl(entries, previous)
    print_report(entries, rows, notes)

    if args.dry_run:
        print("\nDry run: nothing written.")
    else:
        if new_text != old_text or not args.output.exists():
            write_atomically(args.output, new_text)
        write_csv(args.out_dir / "benchmark_composition.csv", COMPOSITION_HEADER, rows)
        state = "updated" if new_text != old_text else "unchanged"
        print(f"\n{args.output}: {state}; {assigned} split(s) assigned")
        print(f"Wrote {args.out_dir / 'benchmark_composition.csv'}")

    has_problems = any(entry.errors or entry.gold_error for entry in entries)
    return 1 if has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
