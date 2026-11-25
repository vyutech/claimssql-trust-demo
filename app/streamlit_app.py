"""ClaimsSQL-Trust demo app (entry point).

Pages, in app/app_pages/ (top navigation):
  Home          what the project is, headline numbers, the three steps (default)
  How it works  the three checks with real examples, one question walked through
  Results       the research findings with interactive charts
  Explorer      what each model did on each benchmark question, with the checks
  Try it        the live pipeline on the 5,000-member demo database
  About         dataset, models, limitations, author

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable without installing it (needed on Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from ui_common import inject_css, sidebar  # noqa: E402

st.set_page_config(page_title="ClaimsSQL-Trust", page_icon=":material/verified_user:",
                   layout="wide")  # fmt: skip

page = st.navigation(
    [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True,
                help="What the project is, the headline numbers, and the three steps every "
                     "question goes through."),
        st.Page("app_pages/how_it_works.py", title="How it works",
                icon=":material/account_tree:", url_path="how-it-works",
                help="The three safety checks with real examples, and one question followed "
                     "from the model's SQL to the final decision."),
        st.Page("app_pages/results.py", title="Results", icon=":material/insights:",
                help="The research findings in plain language, with interactive charts of "
                     "harmful answers and trick questions."),
        st.Page("app_pages/explorer.py", title="Explorer", icon=":material/manage_search:",
                help="Browse every benchmark question: what each model answered with and "
                     "without the checks, and which check stopped a bad answer."),
        st.Page("app_pages/try_it.py", title="Try it", icon=":material/chat:",
                url_path="try-it",
                help="Ask your own question. The live pipeline answers only if every check "
                     "passes, on a 5,000-member subset of the data."),
        st.Page("app_pages/about.py", title="About", icon=":material/info:",
                help="The dataset, the three models, the limitations and the author."),
    ],
    position="top",
)  # fmt: skip
inject_css()
sidebar()
page.run()
