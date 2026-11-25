"""Render the catalogue as the compact schema text that goes into the prompt.

One line per column or numbered family, code meanings inline. Only entries with
include: true are rendered. `notes` are never rendered: they hold the owner's
trap notes, and showing them to the model would give the traps away.

Usage:
    python -m claimssql.schema.context_builder            # print context + token estimate
    python -m claimssql.schema.context_builder --count-only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from claimssql.schema.catalogue import (
    DEFAULT_CATALOGUE_PATH,
    Catalogue,
    ColumnEntry,
    TableEntry,
    load_catalogue,
)

# Measured in P2: the real context is 8,319 characters and the two local models
# count 2,748 and 2,859 prompt tokens for it, i.e. about 3 characters per token
# (column names and codes split into more tokens than plain English's ~4).
CHARS_PER_TOKEN = 3


def render_column(entry: ColumnEntry) -> str:
    members = entry.members()
    if len(members) > 1:
        # Spell out both ends so the model sees real column names.
        name = f"{members[0]} .. {members[-1]} ({len(members)} columns)"
    else:
        name = entry.name
    line = f"  {name} {entry.type}"
    if entry.description.strip():
        line += f": {entry.description.strip()}"
    if entry.codes:
        line += " [" + ", ".join(f"{code}={meaning}" for code, meaning in entry.codes.items()) + "]"
    return line


def render_table(table: TableEntry) -> str:
    lines = [f"TABLE {table.name}"]
    if table.grain.strip():
        lines.append(f"grain: {table.grain.strip()}")
    if table.description.strip():
        lines.append(table.description.strip())
    lines.append("columns:")
    lines += [render_column(entry) for entry in table.columns if entry.include]
    return "\n".join(lines)


def render_context(catalogue: Catalogue) -> str:
    return "\n\n".join(render_table(table) for table in catalogue.tables)


def approx_tokens(text: str) -> int:
    """Rough token count (characters / CHARS_PER_TOKEN). An estimate, not a tokeniser."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the schema context for the prompt.")
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument("--count-only", action="store_true", help="print only the size")
    args = parser.parse_args()

    context = render_context(load_catalogue(args.catalogue))
    if not args.count_only:
        sys.stdout.reconfigure(encoding="utf-8")  # codebook labels contain en dashes
        print(context)
        print()
    print(f"{len(context):,} characters, {len(context.splitlines())} lines, "
          f"about {approx_tokens(context):,} tokens (estimate: characters / {CHARS_PER_TOKEN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
