"""The two prompts of the pipeline.

  forced   the model must write a query (baseline C0: no way to say "I can't")
  abstain  the model may refuse (C1-C4)

Both share the same schema text and rules and differ in one rule only, so any
difference in behaviour comes from being allowed to refuse. There are no
examples in the prompts: nothing from the benchmark can leak into them.

PROMPT_VERSION is part of the cache key. Change ANY text in this file and bump
it, otherwise old cached answers would be reused for the new prompt.
"""

from __future__ import annotations

from claimssql.llm.base import Message

PROMPT_VERSION = "v1"

INTRO = "You translate questions about a healthcare claims database into SQL."

OUTPUT_FORMAT = (
    '{"answerable": true or false, "sql": "<the query>" or null, '
    '"confidence": <integer from 0 to 100>, "reason": "<one short sentence>"}'
)

SHARED_RULES = [
    "Write DuckDB SQL: one read-only query (SELECT, optionally starting with WITH).",
    "Use only the tables and columns listed in the schema. Do not invent tables or columns.",
    "Columns whose names end in _DT hold dates as integers in YYYYMMDD form, not as DATE values.",
    "confidence is how sure you are, from 0 to 100, that your answer is correct.",
    "Reply with exactly one JSON object and nothing else, in this form: " + OUTPUT_FORMAT,
]

FORCED_RULE = (
    'You must always write a query: set "answerable" to true and put your best query '
    'in "sql", even if you are unsure.'
)

ABSTAIN_RULE = (
    "First decide whether the question can be answered from this database. If it cannot, "
    'set "answerable" to false and "sql" to null, and say why in "reason". Otherwise set '
    '"answerable" to true and put the query in "sql".'
)

# Sent once when a reply could not be parsed (see generator.py).
RETRY_MESSAGE = (
    "Your reply could not be read. Reply again with exactly one JSON object and nothing "
    "else, in this form: " + OUTPUT_FORMAT
)


def _system_text(schema_context: str, mode_rule: str) -> str:
    rules = "\n".join(f"- {rule}" for rule in [*SHARED_RULES, mode_rule])
    # Schema first: it is identical for every question, so a local server can
    # reuse its work on that part of the prompt between calls.
    return f"{INTRO}\n\nDATABASE SCHEMA\n{schema_context}\n\nRULES\n{rules}"


def _messages(schema_context: str, question: str, mode_rule: str) -> list[Message]:
    return [
        {"role": "system", "content": _system_text(schema_context, mode_rule)},
        {"role": "user", "content": f"Question: {question.strip()}"},
    ]


def build_forced(schema_context: str, question: str) -> list[Message]:
    return _messages(schema_context, question, FORCED_RULE)


def build_abstain(schema_context: str, question: str) -> list[Message]:
    return _messages(schema_context, question, ABSTAIN_RULE)
