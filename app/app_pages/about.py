"""About page: the dataset, the models, the limitations and the author."""

import pandas as pd
import streamlit as st

from claimssql.config import load_models
from ui_common import load_json

st.title("About")
st.markdown(
    "**ClaimsSQL-Trust** is an academic project that measures how often small open language "
    "models write wrong SQL for questions about healthcare claims, and whether a set of "
    "checks lets them refuse instead of giving a confident wrong answer. It has three parts: "
    "a 200-question benchmark, a Text-to-SQL service that can refuse (this demo), and an "
    "evaluation that produces every table and figure in the report."
)

left, right = st.columns(2, gap="medium")
with left.container(key="step_dataset", height="stretch"):
    st.markdown("### :material/database: The data")
    st.markdown(
        "CMS **DE-SynPUF** Sample 1 (2008–2010): Medicare claims that are **fully "
        "synthetic**, published by CMS for training and software testing. No real patient "
        "appears in it, and its numbers are not valid for medical or policy conclusions.\n\n"
        "Four tables: members by year, inpatient, outpatient and doctor (carrier) claims. "
        "Codes and dates are kept exactly as CMS publishes them; handling them correctly is "
        "part of the test.\n\n"
        "The **Try it** page uses a 5,000-member subset so it fits a free server; the "
        "benchmark and Explorer use the full sample."
    )
with right.container(key="step_benchmark", height="stretch"):
    st.markdown("### :material/quiz: The benchmark")
    st.markdown(
        "**200 hand-written questions**: 120 answerable (40 easy, 40 medium, 40 hard), each "
        "with an execution-checked gold query, and 80 trick questions that the data cannot "
        "answer (missing concept, not in the data, ambiguous, not a SQL question).\n\n"
        "Questions are split once: 20% *dev*, used only to choose the agreement threshold "
        "τ, and 80% *test*, on which every reported number is measured."
    )

st.subheader("The models")
models = [m for m in load_models().models if m.use == "benchmark"]
st.dataframe(pd.DataFrame([
    {"Model": m.model, "Maker": m.info.get("maker", ""),
     "Parameters": m.info.get("parameters", ""), "Quantization": m.info.get("quantization", ""),
     "Size (GB)": m.info.get("size_gb", ""), "Hardware": m.info.get("hardware", "")}
    for m in models
]), hide_index=True, alt="The open models in the benchmark",
    column_config={"Hardware": st.column_config.TextColumn(width="large")})  # fmt: skip
st.caption(
    "All run through Ollama on a free Kaggle GPU, at temperature 0 for the main answer "
    "and 0.7 for the 5 extra attempts. The Try it page uses a hosted model on a free "
    "tier instead."
)

st.subheader("Limitations")
coverage = [m["coverage_c4"] for m in load_json("results.json").get("models", [])]
answered = f"{min(coverage):.0%}–{max(coverage):.0%}" if coverage else "a minority"
st.markdown(
    "- **Synthetic data, one database.** Results may differ on real claims or another "
    "schema.\n"
    "- **Small benchmark.** 160 test questions give wide confidence intervals; small "
    "differences between models are not reliable.\n"
    "- **One author wrote the questions and gold answers**, so they reflect one person's "
    "reading of the data.\n"
    f"- **Refusals have a cost.** With all checks the system answers only {answered} of "
    "the test questions; a correct answer can be refused when the attempts disagree.\n"
    "- **The checks cannot catch everything.** If all attempts make the same mistake, V3 "
    "agrees and a wrong answer gets through.\n"
    "- **Not for real decisions.** This is a research demo, not a medical or billing tool."
)

st.subheader("Author")
with st.container(key="step_author"):
    st.markdown("**Vyshnavi Vadla**  \nB.Tech CSE (Artificial Intelligence)")
