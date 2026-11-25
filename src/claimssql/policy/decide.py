"""The answer-or-abstain decision for the five configurations C0-C4.

Everything is derived from ONE logged record per question and model - the model
is never called again. A record holds:

    record["forced"]   the forced-prompt greedy generation   (generator output)
    record["abstain"]  the abstain-prompt greedy generation
        each: {"parsed": {"answerable", "sql", ...} or None,
               "error": None, "PARSE_ERROR" or "DEGENERATE_OUTPUT"}
    record["v1"]  SchemaCheckResult.to_dict()  for the abstain greedy SQL (or None)
    record["v2"]  DryRunResult.to_dict()       for the abstain greedy SQL (or None)
    record["v3"]  SelfConsistencyResult.to_dict() over the k samples    (or None)

    Config  Rule
    C0      answer with the forced greedy SQL
    C1      answer with the abstain greedy SQL unless the model refused
    C2      C1 and V1 passes
    C3      C2 and V2 passes
    C4      C3 and the greedy result is in the V3 majority cluster and agreement >= tau

Each config adds one check to the previous one, so the first failing check gives
the reason code.
"""

from __future__ import annotations

from typing import Any, NamedTuple

CONFIGS = ("C0", "C1", "C2", "C3", "C4")

MODEL_REFUSED = "MODEL_REFUSED"
SCHEMA_UNKNOWN = "SCHEMA_UNKNOWN"
EXEC_ERROR = "EXEC_ERROR"
LOW_AGREEMENT = "LOW_AGREEMENT"
PARSE_ERROR = "PARSE_ERROR"
DEGENERATE_OUTPUT = "DEGENERATE_OUTPUT"  # the model looped; the server aborted the reply
REASON_CODES = (
    MODEL_REFUSED, SCHEMA_UNKNOWN, EXEC_ERROR, LOW_AGREEMENT, PARSE_ERROR, DEGENERATE_OUTPUT,
)  # fmt: skip


class Decision(NamedTuple):
    answered: bool
    sql: str | None  # the SQL given as the answer; None when abstaining
    reason_code: str | None  # why it abstained; None when answering


def _abstain(reason: str) -> Decision:
    return Decision(False, None, reason)


def _generation_gate(generation: dict[str, Any] | None) -> Decision:
    """Shared first step: is there SQL to answer with at all?"""
    if generation and generation.get("error") == DEGENERATE_OUTPUT:
        return _abstain(DEGENERATE_OUTPUT)  # there is no reply at all
    if not generation or generation.get("error") or not generation.get("parsed"):
        return _abstain(PARSE_ERROR)  # the reply was never readable JSON
    parsed = generation["parsed"]
    sql = parsed.get("sql")
    if not parsed.get("answerable") or not (sql and sql.strip()):
        # Also reached under the forced prompt: models sometimes refuse even
        # when told to always answer (seen in the P4 smoke test).
        return _abstain(MODEL_REFUSED)
    return Decision(True, sql, None)


def decide(record: dict[str, Any], config_name: str, tau: float | None = None) -> Decision:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown config {config_name!r}; expected one of {CONFIGS}")

    if config_name == "C0":
        return _generation_gate(record.get("forced"))

    decision = _generation_gate(record.get("abstain"))  # C1
    if not decision.answered or config_name == "C1":
        return decision

    v1 = record.get("v1") or {}  # C2: schema check
    if v1.get("parse_error"):
        # The SQL is not even well-formed: that is broken SQL, not an unknown name.
        return _abstain(EXEC_ERROR)
    if not v1.get("passed"):
        return _abstain(SCHEMA_UNKNOWN)
    if config_name == "C2":
        return decision

    v2 = record.get("v2") or {}  # C3: dry run
    if not v2.get("ok"):
        return _abstain(EXEC_ERROR)
    if config_name == "C3":
        return decision

    if tau is None:  # C4: self-consistency
        raise ValueError("C4 needs tau")
    v3 = record.get("v3") or {}
    if not v3.get("greedy_in_majority") or v3.get("agreement", 0.0) < tau:
        return _abstain(LOW_AGREEMENT)
    return decision
