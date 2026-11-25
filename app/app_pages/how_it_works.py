"""How it works: the problem, the three checks with real examples, one question
walked through, and a glossary."""

from contextlib import contextmanager
from html import escape

import streamlit as st

from claimssql.eval.explorer import pretty_sql
from ui_common import load_json, reason_text

examples = load_json("examples.json")


@contextmanager
def card(number: int, title: str):
    """A walkthrough card: numbered header with a divider, then the content
    (styled by the .st-key-walk_card* and .walk-head CSS)."""
    with st.container(key=f"walk_card_{number}"):
        st.html(
            f'<div class="walk-head"><span class="walk-num">{number}</span>'
            f'<span class="walk-title">{escape(title)}</span></div>'
        )
        yield


st.title("How it works")
st.markdown(
    "Language models are good at writing SQL that *looks* right. On a real claims database "
    "they also invent column names, misread codes, or answer questions the data cannot "
    "answer, and they sound just as confident when they do. This system keeps the model, "
    "but checks every query before showing an answer, and **refuses** when a check fails."
)

st.subheader("The three safety checks")
cards = [
    ("v1", ":material/table_view:", "V1 · Does every table and column exist?",
     "The SQL is parsed and every name is looked up in the database schema. An invented "
     "column is caught before anything runs."),
    ("v2", ":material/play_circle:", "V2 · Does the SQL run?",
     "The query runs on a read-only copy with a time limit. Anything that errors out is "
     "refused."),
    ("v3", ":material/groups:", "V3 · Do 5 attempts agree?",
     "The model answers 5 more times with a little randomness. If their results disagree, "
     "the model is guessing, and the answer is refused."),
]  # fmt: skip
for column, (key, icon, title, text) in zip(st.columns(3, gap="medium"), cards, strict=True):
    with column.container(key=f"step_{key}", height="stretch"):
        st.markdown(f"## {icon}")
        st.markdown(f"**{title}**")
        st.markdown(text)
        example = examples.get(key)
        if key == "v3":
            example = examples.get("walkthrough")
        if not example:
            continue
        st.caption(f"Real example · {example['model']} · {example['question_id']}")
        st.markdown(f"*{example['question']}*")
        check = example["verification"]
        if key == "v1":
            names = ", ".join(check["unknown_tables"] + check["unknown_columns"])
            st.code(example["sql"], language="sql", wrap_lines=True)
            st.markdown(
                f":orange-badge[:material/do_not_disturb_on: Refused] `{names}` does not exist."
            )
        elif key == "v2":
            error = (check.get("exec_error") or "").split("\n")[0]
            st.code(example["sql"], language="sql", wrap_lines=True)
            st.markdown(
                f":orange-badge[:material/do_not_disturb_on: Refused] "
                f"{check['exec_error_class']} error"
            )
            st.caption(error[:180])
        else:
            st.markdown(
                f":orange-badge[:material/do_not_disturb_on: Refused] only "
                f"{check['agreement']:.0%} of the attempts agreed "
                f"(groups of {', '.join(str(s) for s in check['cluster_sizes'])})."
            )

walk = examples.get("walkthrough")
if walk:
    st.subheader("One question, followed all the way")
    st.caption(f"{walk['model']} on benchmark question {walk['question_id']}")
    gold = walk["gold_result"]["rows"][0][0] if walk.get("gold_result") else None
    c0 = walk["c0"]["result"]["rows"][0][0] if walk["c0"].get("result") else None
    check = walk["verification"]
    # The question as a full-width strip (styled by the .walk-question CSS).
    st.html(
        '<div class="walk-question"><span class="walk-label">QUESTION</span>'
        f'<span class="walk-text">{escape(walk["question"])}</span></div>'
    )
    # Four cards in a wrapping row: 4 per row, 2 on medium screens, 1 on phones
    # (the widths are set in the .st-key-walk_cards CSS, not with st.columns, which
    # can only switch from 4 straight to 1).
    with st.container(key="walk_cards", horizontal=True, gap="small"):
        with card(1, "The model's SQL"):
            # Laid out one clause and one flag per line; scrolls inside the card if long.
            st.code(pretty_sql(walk["sql"]), language="sql", height=320)
        with card(2, "Why it is wrong"):
            if c0 is not None and gold is not None:
                st.markdown(
                    f":red-badge[Returns {c0:,}] every member, not the "
                    f":green-badge[{gold:,}] who really have 3 or more."
                )
            st.caption(walk["owner_note"])
        with card(3, "The checks"):
            st.markdown(":green-badge[:material/check_circle: V1 · names exist]")
            st.markdown(":green-badge[:material/check_circle: V2 · it runs]")
            st.markdown(f":red-badge[:material/cancel: V3 · {check['agreement']:.0%} agree]")
            st.caption("The SQL is valid, so only the agreement check can catch this mistake.")
        with card(4, "The decision"):
            st.markdown("No checks (C0)  \n:red-badge[:material/cancel: Wrong answer]")
            st.markdown("All checks (C4)  \n:orange-badge[:material/do_not_disturb_on: Refused]")
            st.caption(reason_text(walk["reason_code"]))


# ------------------------------------------------------------- glossary ----
# Compact term cards, three per row, grouped under three headings. The coloured dots
# follow the app's meaning: red = harmful, amber = refused, green = correct.
def heading(key: str, icon: str, text: str) -> None:
    """A group heading: green, bold, with a thin green bar (.st-key-ghead_* CSS)."""
    with st.container(key=f"ghead_{key}"):
        # A bold paragraph sized by the CSS (a heading would add its own spacing).
        st.markdown(f":green[**{icon} {text}**]")


def term(key: str, name: str, text: str, example: str = "", dot: str = "") -> None:
    with st.container(key=f"term_{key}", height="stretch"):
        st.markdown(f":{dot}[●] **{name}**" if dot else f"**{name}**")
        st.markdown(text)
        if example:
            st.caption(example)


def rs_card() -> None:
    with st.container(key="term_rs", height="stretch"):
        st.markdown("**Reliability Score (RS₁₀)**")
        st.markdown(
            "Rewards right answers and safe refusals, and punishes a harmful answer ten "
            "times as hard. Example with 10 questions:"
        )
        st.markdown(
            "| Outcome | Qs | Points |\n|---|---:|---:|\n"
            "| Right answer | 6 | :green[+6] |\n"
            "| Justified refusal | 2 | :orange[+2] |\n"
            "| Unneeded refusal | 1 | 0 |\n"
            "| Wrong answer | 1 | :red[−10] |"
        )
        st.markdown("(6 + 2 + 0 − 10) / 10 × 100 = **−20**")
        st.caption("One wrong answer cancels out eight good ones.")


def row(cards: list) -> None:
    """Equal-width columns; height="stretch" cards make every card in the row as tall
    as the tallest. A card is the arguments for term(), or a function that draws one."""
    for column, card in zip(st.columns(3, gap="small"), cards, strict=True):
        with column:
            if callable(card):
                card()
            else:
                term(*card)


v1 = examples.get("v1")
invented = ", ".join(v1["verification"]["unknown_columns"]) if v1 else ""
coverage = [m["coverage_c4"] for m in load_json("results.json").get("models", [])]

st.space("small")
with st.expander("Glossary", icon=":material/menu_book:"):
    heading("problem", ":material/report:", "The problem")
    row([
        ("hallucination", "Hallucination",
         "The model makes something up, such as a column or a code meaning, that looks "
         "real but is not.",
         f"Example: a column called {invented}, which does not exist." if invented else ""),
        ("harmful", "Harmful answer",
         "A wrong answer to a real question, or any answer to a trap question.",
         "Example: 112,754 members when the right number is 35,857.", "red"),
        ("trap", "Trap question (T1–T4)",
         "A question the data cannot answer, so the right response is to refuse.",
         "T1 missing concept · T2 not in the data · T3 ambiguous · T4 not a database "
         "question"),
    ])  # fmt: skip

    heading("system", ":material/settings_suggest:", "What the system does")
    row([
        ("text_to_sql", "Text-to-SQL",
         "Turning a question in plain English into a database query that answers it."),
        ("abstention", "Abstention (refusal)",
         "The system says “I can't answer that” and gives the reason, instead of guessing.",
         "Reasons: the model refused, a name does not exist, the query fails, or the "
         "attempts disagree.", "orange"),
        ("agreement", "Self-consistency (agreement)",
         "The model answers 5 more times; if the results do not match, the answer is not "
         "trusted.",
         "Agreement = the share of attempts that gave the most common result."),
    ])  # fmt: skip

    heading("measured", ":material/straighten:", "How it's measured")
    with st.container(key="term_ladder"):
        st.markdown("**C0–C4: five ways to run the same model**")
        st.markdown(
            ":gray-badge[C0 · no checks] → :blue-badge[C1 · may refuse] → "
            ":blue-badge[C2 · + V1 names exist] → :blue-badge[C3 · + V2 query runs] → "
            ":blue-badge[C4 · + V3 attempts agree]"
        )
        st.caption("Each step keeps everything before it and adds one more check.")
    row([
        ("gold", "Gold answer",
         "The correct result, from a query the author wrote and checked by running it.",
         "", "green"),
        rs_card,
        ("coverage", "Coverage",
         "The share of questions the system answers instead of refusing.",
         f"With all checks: {min(coverage):.0%}–{max(coverage):.0%} of test questions."
         if coverage else ""),
    ])  # fmt: skip
