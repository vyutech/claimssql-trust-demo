"""Shared pieces of the demo app's pages: data loading, colours and small widgets.

Colour has one meaning everywhere: green = answered correctly, amber = refused
safely, red = wrong answer. The app's logic (which checks passed, which cases are
interesting) lives in claimssql.eval.explorer; this module only displays it.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from claimssql.api.pipeline import REASON_TEXT, DemoService
from claimssql.config import REPO_ROOT
from claimssql.eval.explorer import read_explorer, verification_steps

APP_DATA = REPO_ROOT / "app_data"
# Set to the public repository to show a GitHub button; empty hides it (and the sidebar).
GITHUB_URL = ""

GREEN, AMBER, RED = "#22C55E", "#F59E0B", "#EF4444"
GROUP_NAMES = {
    "easy": "Easy", "medium": "Medium", "hard": "Hard",
    "T1": "T1 · missing concept", "T2": "T2 · not in data", "T3": "T3 · ambiguous",
    "T4": "T4 · not a SQL question",
}  # fmt: skip
# Badge per check status: (colour, icon). "unused" = shown for comparison, not applied.
STEP_STYLE = {
    "pass": ("green", ":material/check_circle:"),
    "fail": ("red", ":material/cancel:"),
    "skipped": ("gray", ":material/remove:"),
    "unused-pass": ("gray", ":material/check:"),
    "unused-fail": ("gray", ":material/close:"),
}

# Small, scoped styling (the owner asked for it): the Home header and the tiles.
# Only .st-key-<key> classes of containers created with that key are targeted.
CSS = """
<style>
.st-key-hero {
  background: radial-gradient(120% 140% at 0% 0%, rgba(34,197,94,0.22) 0%,
              rgba(34,197,94,0.06) 45%, rgba(10,13,12,0) 75%);
  border: 1px solid rgba(34,197,94,0.25);
  border-radius: 20px;
  padding: 2.6rem 2.4rem 2.2rem 2.4rem;
}
[class*="st-key-tile"] {
  background: linear-gradient(180deg, #161D1A 0%, #121715 100%);
  border: 1px solid #232B28;
  border-radius: 16px;
  padding: 1.4rem 1.5rem;
}
[class*="st-key-step"] {
  background: #141917;
  border: 1px solid #232B28;
  border-radius: 16px;
  padding: 1.2rem 1.3rem;
}
/* How it works walkthrough: the question strip ... */
.walk-question {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.4rem 1rem;
  background: #121816;
  border-left: 3px solid #22C55E;
  border-radius: 0 10px 10px 0;
  padding: 0.85rem 1.2rem;
}
.walk-label { color: #22C55E; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; }
.walk-text { color: #E7ECEA; font-size: 1.15rem; font-weight: 500; }
/* ... and the four cards: 4 per row, 2 on medium screens, 1 on phones. */
/* Streamlit wraps each child of a horizontal container in a div; that wrapper is
   the flex item, so the widths go on it. */
.st-key-walk_cards { flex-wrap: wrap; align-items: stretch; }
.st-key-walk_cards > div { flex: 1 1 calc(25% - 0.75rem); min-width: 0; }
@media (max-width: 1100px) {
  .st-key-walk_cards > div { flex-basis: calc(50% - 0.5rem); }
}
@media (max-width: 640px) {
  .st-key-walk_cards > div { flex-basis: 100%; }
}
[class*="st-key-walk_card_"] {
  width: 100%;
  background: #151B19;
  border: 1px solid #26302C;
  border-radius: 14px;
  padding: 1rem 1.1rem;
}
.walk-head {
  display: flex; align-items: center; gap: 0.6rem;
  padding-bottom: 0.65rem;
  border-bottom: 1px solid #26302C;
}
.walk-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.5rem; height: 1.5rem; border-radius: 50%;
  background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.55);
  color: #22C55E; font-size: 0.8rem; font-weight: 700;
}
.walk-title { font-weight: 700; color: #E7ECEA; }
/* Glossary group headings: thin green bar on the left, space above each group. */
[class*="st-key-ghead"] {
  border-left: 3px solid #22C55E;
  padding-left: 0.75rem;
  margin-top: 1.25rem;
}
[class*="st-key-ghead"] p { font-size: 1.2rem; line-height: 1.6; margin: 0; }
/* Streamlit gives markdown a -16px bottom margin, which would cut the bar short. */
[class*="st-key-ghead"] [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
/* Glossary term cards: smaller and tighter than the step cards. */
[class*="st-key-term"] {
  background: #141917;
  border: 1px solid #232B28;
  border-radius: 12px;
  padding: 0.75rem 1rem;
  gap: 0.25rem;
}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def tile(key: str, value: str, label: str, note: str = "", color: str = "green") -> None:
    """A large headline number in a styled card (styled by the .st-key-tile* CSS)."""
    with st.container(key=f"tile_{key}", height="stretch"):
        st.markdown(f"# :{color}[{value}]")
        st.markdown(f"**{label}**")
        if note:
            st.caption(note)


@st.cache_data
def load_explorer() -> list[dict[str, Any]]:
    return read_explorer(APP_DATA / "explorer.jsonl")


@st.cache_data
def load_json(name: str) -> dict[str, Any]:
    path = APP_DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource
def load_service() -> DemoService:
    return DemoService.from_env()


def reason_text(code: str | None) -> str:
    return REASON_TEXT.get(code or "", code or "")


def outcome_badge(item: dict[str, Any], answerable: bool) -> None:
    """Green = correct, amber = refused, red = wrong (or answered a trick question)."""
    if item["answered"]:
        if not answerable:
            st.badge("Wrong: answered a trick question", icon=":material/error:", color="red")
        elif item["correct"]:
            st.badge("Answered correctly", icon=":material/check_circle:", color="green")
        else:
            st.badge("Wrong answer", icon=":material/cancel:", color="red")
    else:
        label = "Refused safely" if not answerable else "Refused"
        st.badge(label, icon=":material/do_not_disturb_on:", color="orange")


def steps_markdown(steps: list[dict[str, Any]]) -> str:
    """The checks as a row of coloured badges joined by arrows."""
    parts = []
    for step in steps:
        color, icon = STEP_STYLE[step["status"]]
        parts.append(f":{color}-badge[{icon} {step['name']}: {step['detail']}]")
    return " → ".join(parts)


def verification_row(item: dict[str, Any], config: str, tau: float) -> None:
    st.markdown(steps_markdown(verification_steps(item, config, tau)))
    if config == "C0":
        st.caption("C0 does not use the checks; grey badges show what they would have said.")


def result_table(preview: dict[str, Any] | None, alt: str) -> None:
    if not preview:
        return
    if preview.get("error"):
        st.caption(f"The query failed: {preview['error']}")
        return
    st.dataframe(pd.DataFrame(preview["rows"], columns=preview["columns"]), hide_index=True,
                 height="content", alt=alt)  # fmt: skip
    if preview.get("more_rows"):
        st.caption("First rows only.")


def sidebar() -> None:
    """Only a GitHub button, and only once a real URL is set; otherwise no sidebar."""
    if GITHUB_URL:
        with st.sidebar:
            st.link_button("Code on GitHub", GITHUB_URL, icon=":material/code:",
                           width="stretch")  # fmt: skip
