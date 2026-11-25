"""Try it page: the live pipeline on the 5,000-member demo database."""

import pandas as pd
import streamlit as st

from claimssql.llm.base import LLMError, RateLimitStop
from ui_common import STEP_STYLE, load_json, load_service

if "question" not in st.session_state:
    st.session_state.question = ""
if "answer" not in st.session_state:
    st.session_state.answer = None  # the last Answer, as a dict
if "run_now" not in st.session_state:
    st.session_state.run_now = False

st.title("Try it yourself")
st.caption(
    "Runs on a 5,000-member subset of the synthetic data, so numbers differ from the "
    "benchmark. One question takes up to two minutes on the free model tier. The demo "
    "answers when the SQL passes V1 and V2 and at least 2 of 3 extra samples agree "
    "(a demo setting, not a benchmark result)."
)


def use_sample(text: str) -> None:
    st.session_state.question = text
    st.session_state.run_now = True


samples = load_json("samples.json")
if samples:
    st.markdown("**Try an example question**")
    # Same colour meaning as everywhere: green = should answer, amber = should refuse.
    icons = {"Should answer": ":green[:material/check_circle:]",
             "Tricky (trap tags)": ":violet[:material/warning:]",
             "Should refuse": ":orange[:material/do_not_disturb_on:]"}  # fmt: skip
    for number, (column, (label, items)) in enumerate(
        zip(st.columns(len(samples)), samples.items(), strict=True)
    ):
        with column.container(key=f"step_sample_{number}", height="stretch"):
            st.markdown(f"{icons.get(label, '')} **{label}**")
            for sample in items:
                st.button(
                    sample["question"],
                    key=f"sample_{sample['id']}",
                    width="stretch",
                    type="tertiary",
                    on_click=use_sample,
                    args=(sample["question"],),
                )

st.text_area(
    "Your question",
    key="question",
    max_chars=1000,
    height=80,
    placeholder="How many members had diabetes in 2009?",
)
clicked = st.button("Ask", type="primary", icon=":material/send:",
                    disabled=not st.session_state.question.strip())  # fmt: skip

if clicked or st.session_state.run_now:
    st.session_state.run_now = False
    question = st.session_state.question.strip()
    try:
        service = load_service()
    except Exception as exc:  # noqa: BLE001 - shown to the user instead of a stack trace
        st.error(f"The demo is not set up: {exc}", icon=":material/error:")
        st.stop()
    try:
        with st.status("Working on it…", expanded=True) as status:
            answer = service.ask(question, progress=lambda message: status.write(
                f":material/progress_activity: {message}"))  # fmt: skip
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.answer = answer.to_dict()
    except RateLimitStop:
        st.error("The model provider's free daily limit has been reached. Please try again "
                 "tomorrow, or use the Explorer, which needs no model calls.",
                 title="Daily limit reached", icon=":material/hourglass_disabled:")  # fmt: skip
        st.stop()
    except LLMError as exc:
        st.error(f"The model could not be reached: {exc}", icon=":material/cloud_off:")
        st.stop()

answer = st.session_state.answer
if answer:
    with st.container(key="step_answer"):
        st.markdown(f"**Question:** {answer['question']}")
        if answer["answered"]:
            st.success(
                "The SQL passed every check.", title="Answered", icon=":material/check_circle:"
            )
            st.code(answer["sql"], language="sql", wrap_lines=True)
            st.dataframe(
                pd.DataFrame(answer["rows"], columns=answer["columns"]),
                hide_index=True,
                height="content",
                alt="Result of the answer",
            )
            if answer["more_rows"]:
                st.caption("First 100 rows only.")
        else:
            st.warning(answer["reason_text"], title="Refused", icon=":material/do_not_disturb_on:")
            if answer["model_sql"]:
                with st.expander("SQL the model wrote (not used)", icon=":material/code:"):
                    st.code(answer["model_sql"], language="sql", wrap_lines=True)
            if answer["model_reason"]:
                st.caption(f"The model's own explanation: {answer['model_reason']}")

        unknown = answer["unknown_tables"] + answer["unknown_columns"]
        has_sql = bool(answer["model_sql"])
        steps = [
            ("Model", has_sql, "wrote SQL" if has_sql else "gave no SQL"),
            ("V1 schema", not unknown, ", ".join(unknown) or "all names exist"),
            ("V2 dry run", not answer["exec_error"], "fails" if answer["exec_error"] else "runs"),
            ("V3 agreement", answer["agreement"] is not None and answer["answered"],
             "no samples ran" if answer["agreement"] is None
             else f"{answer['agreement']:.0%} agree, needs {answer['tau']:.0%}"),
        ]  # fmt: skip
        badges = []
        for position, (name, passed, detail) in enumerate(steps):
            state = "skipped" if position and not has_sql else ("pass" if passed else "fail")
            color, icon = STEP_STYLE[state]
            badges.append(f":{color}-badge[{icon} {name}: {detail}]")
        st.markdown("**Checks** " + " → ".join(badges))
        st.caption(
            f"Model {answer['model']} · {answer['calls']} calls "
            f"({answer['cached_calls']} from the cache)"
        )
