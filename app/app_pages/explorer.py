"""Explorer page: what each model did on each benchmark question, with the checks."""

import random

import streamlit as st

from claimssql.eval.explorer import filter_rows, pick_interesting, saved_by_checks
from ui_common import (
    GROUP_NAMES,
    load_explorer,
    outcome_badge,
    reason_text,
    result_table,
    verification_row,
)

CONFIG_TITLES = {"C0": "C0 · no checks", "C4": "C4 · all checks"}
CONFIG_NOTES = {"C0": "The model must answer; nothing is checked.",
                "C4": "The model may refuse; the SQL must pass V1, V2 and V3."}  # fmt: skip

rows = load_explorer()
if not rows:
    st.info(
        "No explorer data yet. Build it with `python scripts/build_demo_db.py`.",
        icon=":material/info:",
    )
    st.stop()

if "explorer_pick" not in st.session_state:
    st.session_state.explorer_pick = None  # (model, question id) chosen by the button

st.title("Explorer")
st.markdown(
    "Every benchmark question, as each model answered it without checks (C0) and with "
    "all checks (C4). Gold answers are computed on the full database."
)
# Small and grey on purpose, so the guide does not compete with the title; only the
# dots carry the theme colours ("orange" is set to amber in .streamlit/config.toml).
st.caption(
    "**How to read this page**\n"
    "- Pick a model and a question\n"
    "- Gold = the correct answer\n"
    "- C0 = no checks · C4 = all three checks\n"
    "- :green[●] correct · :red[●] wrong · :orange[●] refused (with the reason)"
)

models = list(dict.fromkeys(r["model_key"] for r in rows))
labels = {r["model_key"]: r["model"] for r in rows}
tags = sorted({tag for r in rows for tag in r.get("trap_tags", [])})
# The model choice starts here, not as a widget default: the "interesting case"
# button also sets it, and a widget may not get its value from both places.
if st.session_state.get("model") not in models:
    st.session_state.model = models[0]


def show_interesting() -> None:
    row = pick_interesting(rows, random.Random())
    if row:
        st.session_state.explorer_pick = (row["model_key"], row["question_id"])
        st.session_state.model = row["model_key"]
        st.session_state.groups = []
        st.session_state.tag = None
        st.session_state.saved_only = True


with st.container(horizontal=True, vertical_alignment="bottom"):
    model = st.segmented_control("Model", models, format_func=labels.get, key="model",
                                 required=True)  # fmt: skip
    groups = st.multiselect("Groups", list(GROUP_NAMES), format_func=GROUP_NAMES.get,
                            key="groups", placeholder="All groups")  # fmt: skip
    tag = st.selectbox("Trap tag", tags, index=None, key="tag", placeholder="Any tag")
    saved_only = st.toggle(
        "Only cases where the checks prevented a harmful answer",
        key="saved_only",
        help="Only cases where C0 gave a harmful answer and C4 refused.",
    )
    st.button(
        "Show me an interesting case",
        key="interesting",
        icon=":material/casino:",
        type="primary",
        on_click=show_interesting,
    )

choices = filter_rows(rows, model, groups, tag, saved_only)
model_rows = filter_rows(rows, model)
saved = sum(saved_by_checks(r) for r in model_rows)
st.caption(
    f"{len(choices)} matching questions. For {labels[model]}, across all {len(model_rows)} "
    f"questions, the checks turned {saved} harmful C0 answers into refusals."
)
if not choices:
    st.info("No question matches these filters.", icon=":material/filter_alt_off:")
    st.stop()

index = 0
if st.session_state.explorer_pick:
    picked = [
        i
        for i, r in enumerate(choices)
        if (r["model_key"], r["question_id"]) == st.session_state.explorer_pick
    ]
    index = picked[0] if picked else 0
row = st.selectbox("Question", choices, index=index,
                   format_func=lambda r: f"{r['question_id']} · {r['question']}")  # fmt: skip
st.session_state.explorer_pick = (row["model_key"], row["question_id"])

with st.container(key="step_question"):
    st.subheader(row["question"])
    with st.container(horizontal=True):
        st.badge(GROUP_NAMES.get(row["group"], row["group"]), color="blue")
        st.badge(f"{row['split']} split", color="gray")
        for trap in row.get("trap_tags", []):
            st.badge(trap.replace("_", " "), icon=":material/warning:", color="violet")
    if row["answerable"]:
        with st.expander("Gold answer", icon=":material/verified:"):
            st.code(row["gold_sql"], language="sql", wrap_lines=True)
            result_table(row["gold"], alt="Result of the gold SQL")
    else:
        st.markdown(f":material/block: **Cannot be answered from this database.** {row['notes']}")

for column, config in zip(st.columns(2, gap="medium"), ("C0", "C4"), strict=True):
    item = row["configs"][config]
    with column.container(key=f"step_{config.lower()}", height="stretch"):
        st.markdown(f"### {CONFIG_TITLES[config]}")
        st.caption(CONFIG_NOTES[config])
        outcome_badge(item, row["answerable"])
        if item["answered"]:
            st.code(item["sql"], language="sql", wrap_lines=True)
            result_table(item["result"], alt=f"Result of the {config} answer")
        else:
            st.markdown(f"**Why:** {reason_text(item['reason_code'])}")
            if item.get("model_sql"):
                with st.expander("SQL the model wrote (not used)", icon=":material/code:"):
                    st.code(item["model_sql"], language="sql", wrap_lines=True)
        st.markdown("**Checks**")
        verification_row(item, config, row["tau"])
