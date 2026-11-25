"""Read variable labels and code tables out of the CMS DE-SynPUF codebook.

Source: "CMS Linkable 2008-2010 Medicare DE-SynPUF Codebook" (de-10-codebook.pdf).
Every variable has an entry that looks like this in the PDF text:

    BEN-4.
    Label: DESYNPUF: Sex
    Variable Name: BENE_SEX_IDENT_CD
    Categories
    Code Label 2008 Freq. ...
    1 Male 1,033,995  44.44 ...
    2 Female 1,292,861  55.56 ...
    Total ...

Numbered families share one entry ("IP-21-30.", "Variable Name: ICD9_DGNS_CD_1 -
ICD9_DGNS_CD_10") and carry a "Label prefix:" line. A few entries list codes as
"A  =  Allowed" lines instead of a Categories table.

The labels are used verbatim as catalogue descriptions (minus the "DESYNPUF:"
prefix and footnote digits) - nothing is paraphrased, so the codebook can be
cited as the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Codebook section prefix -> our table name. PDE (drug events) is out of scope.
SECTION_TABLES = {
    "BEN": "beneficiary",
    "IP": "inpatient_claims",
    "OP": "outpatient_claims",
    "CAR": "carrier_claims",
}

_ENTRY_HEADER = re.compile(r"^(BEN|IP|OP|CAR|PDE)-(\d+)(?:-(\d+))?\.\s*$", re.M)
_DASH = r"[–—-]"  # en dash, em dash or hyphen between range ends
# A footnote number glued to the end of a word: "CLM_FROM_DT2", "start date1".
# Family members end in "_<n>" and labels in " <n>", so those are left alone.
_FOOTNOTE = re.compile(r"(?<=[A-Za-z\)])\d+$")
_CATEGORY_ROW = re.compile(
    r"^(\S+)\s+(.+?)\s+[\d,]+\s+[\d.]+\s+[\d,]+\s+[\d.]+\s+[\d,]+\s+[\d.]+\s*$"
)
_EQUALS_ROW = re.compile(r"^\s*(\S{1,2})\s+=\s+(\S.*?)\s*$")
_FAMILY_MEMBER = re.compile(r"^(.+_)(\d+)$")


@dataclass(frozen=True)
class CodebookEntry:
    section: str  # BEN / IP / OP / CAR / PDE
    entry_id: str  # e.g. "BEN-4" or "IP-21-30"
    columns: tuple[str, ...]  # every column the entry covers, family expanded
    label: str  # short description, cleaned
    codes: dict[str, str] = field(default_factory=dict)

    @property
    def table(self) -> str | None:
        return SECTION_TABLES.get(self.section)


def extract_text(pdf_path: Path) -> str:
    """Text layer of the PDF. pypdf is imported here so only this step needs it."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _clean_label(text: str) -> str:
    text = text.replace("DESYNPUF:", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return _FOOTNOTE.sub("", text).strip()


def _clean_name(text: str) -> str:
    name = _FOOTNOTE.sub("", text.strip())
    return name.replace("(T)", "")  # the codebook prints "..._LBLTY_AM(T)"


def expand_names(raw: str) -> tuple[str, ...]:
    """'ICD9_DGNS_CD_1 - ICD9_DGNS_CD_10' -> the ten names; a single name -> itself."""
    parts = [_clean_name(p) for p in re.split(rf"\s+{_DASH}\s+", raw.strip()) if p.strip()]
    if len(parts) == 1:
        return (parts[0],)
    first, last = _FAMILY_MEMBER.match(parts[0]), _FAMILY_MEMBER.match(parts[-1])
    if not first or not last or first.group(1) != last.group(1):
        raise ValueError(f"cannot read variable range: {raw!r}")
    prefix = first.group(1)
    return tuple(f"{prefix}{n}" for n in range(int(first.group(2)), int(last.group(2)) + 1))


def _parse_codes(block: str) -> dict[str, str]:
    codes: dict[str, str] = {}
    in_table = False
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("Categories"):
            in_table = True
            continue
        if in_table:
            if stripped.startswith("Total"):
                in_table = False
            elif match := _CATEGORY_ROW.match(stripped):
                codes[match.group(1)] = match.group(2).strip()
            continue
        if match := _EQUALS_ROW.match(line):
            # First definition wins; later tables in the same entry repeat codes.
            codes.setdefault(match.group(1), match.group(2))
    return codes


def _parse_entry(section: str, entry_id: str, block: str) -> CodebookEntry | None:
    # Label and variable name can each wrap onto a second line.
    label = re.search(r"^Label:\s*(.*?)(?=^Variable Name:)", block, re.M | re.S)
    name = re.search(
        r"^Variable Name:\s*(.*?)(?=^(?:Type:|Source|Categories|Summary|Codebook)|\n\s*\n)",
        block,
        re.M | re.S,
    )
    if not label or not name:
        return None  # the table of contents repeats entry ids without content
    prefix = re.search(r"^Label prefix:\s*(.*)$", block, re.M)
    columns = expand_names(re.sub(r"\s+", " ", name.group(1)))
    return CodebookEntry(
        section=section,
        entry_id=entry_id,
        columns=columns,
        # Families: "Label prefix" is the clean shared description.
        label=_clean_label(prefix.group(1) if prefix else label.group(1)),
        codes=_parse_codes(block),
    )


def parse_codebook(text: str) -> list[CodebookEntry]:
    headers = list(_ENTRY_HEADER.finditer(text))
    entries = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        entry_id = header.group(0).strip().rstrip(".")
        entry = _parse_entry(header.group(1), entry_id, text[header.end() : end])
        if entry:
            entries.append(entry)
    return entries


def index_by_column(entries: list[CodebookEntry]) -> dict[tuple[str, str], CodebookEntry]:
    """(table, COLUMN) -> entry, for the four in-scope tables."""
    index: dict[tuple[str, str], CodebookEntry] = {}
    for entry in entries:
        if entry.table:
            for column in entry.columns:
                index[(entry.table, column.upper())] = entry
    return index
