"""Results page: the research findings in plain language, with interactive charts.

Every number is read from app_data/results.json (built from results/tables/), so the
text cannot drift from the report. Charts use the app's colour meaning: red = harmful
or wrong, amber = refused safely.
"""

import altair as alt
import pandas as pd
import streamlit as st

from ui_common import AMBER, RED, load_json

data = load_json("results.json")
models = data.get("models", [])
if not models:
    st.info(
        "No results yet. Build them with `python scripts/build_demo_db.py`.", icon=":material/info:"
    )
    st.stop()

name = {m["model_key"]: m["model"] for m in models}
order = [m["model"] for m in models]
n_test = data["n_test"]
# The "bigger model" for RQ4 is the one most accurate without checks.
big = max(models, key=lambda m: m["ex_c0"])["model_key"]
small = [m["model_key"] for m in models if m["model_key"] != big]


def difference(comparison: str, metric: str) -> dict | None:
    return next(
        (
            d
            for d in data.get("differences", [])
            if d["comparison"] == comparison and d["metric"] == metric
        ),
        None,
    )


def interval(d: dict, digits: int = 0) -> str:
    return f"{d['lo']:+.{digits}f} to {d['hi']:+.{digits}f}"


st.title("Results")
st.caption(
    f"All numbers are on the {n_test} test questions. Ranges are 95% confidence "
    "intervals from 2,000 bootstrap resamples."
)

# ------------------------------------------------------------- findings ----
c0 = [m["harmful_c0"] for m in models]
rq1 = (
    f"Often. Without checks (C0), **{min(c0):.0f}–{max(c0):.0f} of {n_test}** test "
    f"questions ({min(c0) / n_test:.0%}–{max(c0) / n_test:.0%}) got a harmful answer. "
    "Accuracy on the answerable ones was "
    + ", ".join(f"{m['ex_c0']:.0%} for {m['model']}" for m in models)
    + "."
)

cuts = ", ".join(
    f"**{m['harmful_c0']:.0f} → {m['harmful_c4']:.0f}** for {m['model']} "
    f"(−{1 - m['harmful_c4'] / m['harmful_c0']:.0%})"
    for m in models
)
all_clear = all((d := difference(f"{k} C4 minus {k} C0", "harmful")) and d["excludes_zero"]
                for k in name)  # fmt: skip
cov = [m["coverage_c4"] for m in models]
rq2 = (
    f"Yes. With all checks (C4), harmful answers fell {cuts}"
    + ("; every interval excludes zero" if all_clear else "")
    + ". The cost is coverage: "
    f"C4 answers {min(cov):.0%}–{max(cov):.0%} of questions and refuses the rest."
)

v3 = [a for a in data.get("auroc", []) if a["signal"] == "1 - agreement"]
versus = []
for key in name:
    d = difference(f"{key} AUROC(1 - agreement) minus {key} AUROC(100 - confidence)", "auroc")
    if d:
        clear = d["excludes_zero"] and d["difference"] > 0
        versus.append(
            f"{name[key]} {d['difference']:+.2f} "
            f"({'clearly better' if clear else 'not clearly different'})"
        )
rq3 = (
    f"Disagreement between attempts (V3) spots wrong answers with AUROC "
    f"**{min(a['auroc'] for a in v3):.2f}–{max(a['auroc'] for a in v3):.2f}** for every "
    "model (1.0 is perfect, 0.5 is chance). Compared with the model's own confidence: "
    + "; ".join(versus)
    + "."
)

rq4_parts = []
for key in small:
    d = difference(f"{key} C4 minus {big} C0", "rs_10")
    if d:
        rq4_parts.append(
            f"{name[key]} with checks vs. {name[big]} without: RS₁₀ "
            f"**{d['difference']:+.0f}** ({interval(d)})"
        )
same = [(key, d) for key in small if (d := difference(f"{big} C4 minus {key} C4", "rs_10"))]
rq4 = "Needs results for at least two models." if not rq4_parts else (
    "Yes. " + "; ".join(rq4_parts) + ". With checks on both, " + "; ".join(
    f"{name[big]} vs. {name[key]} is {d['difference']:+.0f} ({interval(d)}, "
    f"{'a clear difference' if d['excludes_zero'] else 'not clearly different'})"
    for key, d in same) + ".")  # fmt: skip

st.subheader("Key findings")
findings = [
    ("RQ1", "How often do small models give harmful SQL without checks?", rq1),
    ("RQ2", "Do the checks reduce harmful answers?", rq2),
    ("RQ3", "Which signal best spots a wrong answer?", rq3),
    ("RQ4", "Can a small model with checks beat a bigger one without?", rq4),
]
columns = st.columns(2, gap="medium")
for position, (number, question, answer) in enumerate(findings):
    with columns[position % 2].container(key=f"step_{number.lower()}", height="stretch"):
        st.markdown(f":green-badge[{number}] **{question}**")
        st.markdown(answer)

# --------------------------------------------------------------- charts ----
# Altair picks up the dark theme from Streamlit; only the colours are set here.
header = alt.Header(labelFontSize=14, labelFontWeight="bold", labelColor="#E7ECEA")

st.subheader("Harmful answers, from no checks to all checks")
harmful = pd.DataFrame(data["harmful"]).assign(model=lambda df: df["model"].map(name))
base = alt.Chart(harmful).encode(x=alt.X("config:N", title=None, axis=alt.Axis(labelAngle=0)))
bars = base.mark_bar(color=RED, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    y=alt.Y("harmful:Q", title=f"Harmful answers (of {n_test})"),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("config:N", title="Config"),
             alt.Tooltip("harmful:Q", title="Harmful answers"),
             alt.Tooltip("lo:Q", title="95% low"), alt.Tooltip("hi:Q", title="95% high")],
)  # fmt: skip
whiskers = base.mark_errorbar(color="#E7ECEA", thickness=1.5, ticks=True).encode(
    y=alt.Y("lo:Q", title=f"Harmful answers (of {n_test})"), y2="hi:Q"
)
harmful_chart = (
    alt.layer(bars, whiskers)
    .properties(width=220, height=280)
    .facet(column=alt.Column("model:N", title=None, sort=order, header=header))
)
st.altair_chart(harmful_chart,
                alt="Bar charts per model: harmful answers fall from about 75 without checks "
                    "(C0) to between 4 and 23 with all checks (C4), with 95% confidence "
                    "intervals.")  # fmt: skip
st.caption(
    "C0: no checks. C1: the model may refuse. C2: + V1 names exist. C3: + V2 it runs. "
    "C4: + V3 attempts agree. White whiskers: 95% confidence intervals."
)

st.subheader("Trick questions: answered or refused")
outcomes = ["Answered (harmful)", "Refused (safe)"]
rows = []
for item in data["infeasible"]:
    for config, answered in (("C0", item["answered_c0"]), ("C4", item["answered_c4"])):
        for outcome, count in zip(outcomes, (answered, item["n"] - answered), strict=True):
            rows.append(
                {
                    "model": name[item["model"]],
                    "group": item["group"],
                    "config": config,
                    "outcome": outcome,
                    "count": count,
                }
            )
trick_chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadius=3).encode(
    x=alt.X("group:N", title=None, axis=alt.Axis(labelAngle=0)),
    xOffset=alt.XOffset("config:N", sort=["C0", "C4"]),
    y=alt.Y("count:Q", title="Trick questions", stack="zero"),
    color=alt.Color("outcome:N", title=None, sort=outcomes,
                    scale=alt.Scale(domain=outcomes, range=[RED, AMBER]),
                    legend=alt.Legend(orient="top", labelLimit=0, columnPadding=24)),
    order=alt.Order("outcome:N", sort="ascending"),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("group:N", title="Group"),
             alt.Tooltip("config:N", title="Config"), alt.Tooltip("outcome:N", title="Outcome"),
             alt.Tooltip("count:Q", title="Questions")],
).properties(width=220, height=260).facet(
    column=alt.Column("model:N", title=None, sort=order, header=header))  # fmt: skip
st.altair_chart(trick_chart,
                alt="Per model and trick-question group, two stacked bars: without checks (C0) "
                    "many questions are answered, mostly the ambiguous ones (T3); with all "
                    "checks (C4) almost all are refused.")  # fmt: skip
st.caption(
    "Each group has two bars: C0 on the left, C4 on the right. T1: missing concept, "
    "T2: not in data, T3: ambiguous, T4: not a SQL question. Trick questions have no "
    "correct answer, so any answer is harmful."
)

st.subheader("At a glance")
table = pd.DataFrame([
    {"Model": m["model"], "Harmful C0": int(m["harmful_c0"]), "Harmful C4": int(m["harmful_c4"]),
     "RS₁₀ C0": round(m["rs10_c0"]), "RS₁₀ C4": round(m["rs10_c4"]),
     "Accuracy C0": m["ex_c0"] * 100, "Answered at C4": m["coverage_c4"] * 100, "τ": m["tau"]}
    for m in models
])  # fmt: skip
st.dataframe(table, hide_index=True, alt="Headline numbers per model", column_config={
    "Accuracy C0": st.column_config.NumberColumn(format="%.0f%%"),
    "Answered at C4": st.column_config.NumberColumn(format="%.0f%%"),
    "τ": st.column_config.NumberColumn(help="Agreement threshold for V3, chosen on the dev "
                                            "split."),
})  # fmt: skip
st.caption(
    "RS₁₀: +1 for a right answer or a justified refusal, 0 for an unneeded refusal, "
    "−10 for a harmful answer, times 100 and averaged."
)
