"""Home page: what the project is, the three steps and the headline numbers."""

import streamlit as st

from ui_common import load_json, tile

summary = load_json("results.json")
models = summary.get("models", [])

with st.container(key="hero"):
    st.title("ClaimsSQL-Trust")
    st.markdown(
        "### An AI that writes SQL for claims data, and knows when to say "
        ":green[“I can't answer that”]"
    )
    st.caption(
        "Ask a question about Medicare claims in plain English. Small open models "
        "write the database query; three safety checks decide whether to trust it."
    )
    with st.container(horizontal=True):
        if st.button("See how it works", type="primary", icon=":material/account_tree:"):
            st.switch_page("app_pages/how_it_works.py")
        if st.button("Try it yourself", icon=":material/chat:"):
            st.switch_page("app_pages/try_it.py")

st.space("medium")
st.subheader("Every question takes three steps")
steps = [
    ("ask", ":material/chat:", "1 · Ask",
     "You ask a question in plain English. The model writes a SQL query, or says it can't."),
    ("check", ":material/verified_user:", "2 · Check",
     "Three safety checks: do all tables and columns exist, does the query run, and do "
     "five more attempts give the same answer?"),
    ("decide", ":material/fact_check:", "3 · Answer or refuse",
     "If every check passes you get the answer. If any fails, the system refuses and says "
     "why, instead of guessing."),
]  # fmt: skip
for column, (key, icon, title, text) in zip(st.columns(3, gap="medium"), steps, strict=True):
    with column.container(key=f"step_{key}", height="stretch"):
        st.markdown(f"## {icon}")
        st.markdown(f"**{title}**")
        st.markdown(text)

st.space("medium")
if models:
    best_cut = max((m["harmful_c0"] - m["harmful_c4"]) / m["harmful_c0"] * 100
                   for m in models if m["harmful_c0"])  # fmt: skip
    cuts = ", ".join(f"{m['model']} {m['harmful_c0']:.0f} → {m['harmful_c4']:.0f}"
                     for m in models)  # fmt: skip
    first, second, third = st.columns(3, gap="medium")
    with first:
        tile(
            "cut",
            f"−{best_cut:.0f}%",
            "fewer harmful answers",
            f"Best case, with all three checks on. Harmful test answers: {cuts}.",
        )
    with second:
        tile(
            "questions",
            "200",
            "benchmark questions",
            "Hand-written; 80 of them cannot be answered from the data and should be refused.",
        )
    with third:
        tile(
            "models",
            str(len(models)),
            "open models",
            ", ".join(m["model"] for m in models) + ", run on a free GPU.",
        )
