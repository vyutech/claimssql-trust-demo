"""The report's figures, drawn from the CSV tables in results/tables/.

One function per figure; each returns a matplotlib Figure. The notebook
notebooks/analysis.ipynb calls them one by one and saves 300-dpi PNGs to
results/figures/.

Every figure must survive black-and-white printing, so nothing depends on
colour: bars differ by grey level AND hatching, lines by marker AND line style.
Models always appear in the order of the run (qwen, gemma, mid_oss). There are
no titles inside the images; the report's captions carry them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

DPI = 300
CONFIGS = ["C0", "C1", "C2", "C3", "C4"]
# (face grey, hatch) - distinguishable without colour.
FILLS = [("white", ""), ("0.78", "///"), ("0.45", "xx"), ("0.9", "..."), ("0.2", "")]
MARKERS = ["o", "s", "^", "D", "v"]
LINES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
GROUP_LABELS = {
    "easy": "easy", "medium": "medium", "hard": "hard",
    "T1": "T1\nmissing\nconcept", "T2": "T2\nnot in\ndata", "T3": "T3\nambiguous",
    "T4": "T4\nnot a SQL\nquestion",
}  # fmt: skip
TAXONOMY = [("infeasible_answered", "Infeasible question answered"),
            ("schema", "Unknown table or column"),
            ("execution_error", "SQL does not run"),
            ("logic", "Runs, wrong result")]  # fmt: skip
SIGNAL_LABELS = {"1 - agreement": "1 − agreement (V3)", "100 - confidence": "100 − confidence",
                 "sql length": "SQL length (baseline)"}  # fmt: skip

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "black", "legend.frameon": False, "hatch.linewidth": 0.6,
    "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight",
})  # fmt: skip


def read(tables: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(tables) / name, keep_default_na=True)


def model_order(tables: Path) -> list[str]:
    """Model keys in run order (the order every figure uses)."""
    return list(dict.fromkeys(read(tables, "e2_rs_by_config_penalty.csv")["model"]))


def model_labels(tables: Path) -> dict[str, str]:
    """Model key -> model name as run (e.g. the Ollama tag), from the cost table."""
    cost = read(tables, "e5_latency_cost.csv")
    return dict(zip(cost["model_key"], cost["model"], strict=True))


def save(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    return path


def _bar(ax, x, height, style: int, label: str | None = None, width: float = 0.8, **kwargs):
    face, hatch = FILLS[style % len(FILLS)]
    return ax.bar(x, height, width=width, color=face, hatch=hatch, edgecolor="black",
                  linewidth=0.8, label=label, **kwargs)  # fmt: skip


def _panels(n: int, width: float = 2.6, height: float = 2.9, sharey: bool = True):
    fig, axes = plt.subplots(1, n, figsize=(max(width * n, 3.2), height), sharey=sharey,
                             squeeze=False)  # fmt: skip
    return fig, list(axes[0])


def fig4_2_benchmark_composition(tables: Path) -> Figure:
    """Questions per group; answerable and infeasible groups drawn differently."""
    data = read(tables, "benchmark_composition.csv")
    data = data[data["group"] != "total"].reset_index(drop=True)
    answerable = data["group"].isin(["easy", "medium", "hard"])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(data))
    _bar(ax, [i for i in x if answerable[i]], data["count"][answerable], 0,
         "Answerable (has gold SQL)")
    _bar(ax, [i for i in x if not answerable[i]], data["count"][~answerable], 1,
         "Infeasible (must be refused)")
    for i, count in enumerate(data["count"]):
        ax.text(i, count + 0.6, str(count), ha="center", va="bottom")
    ax.set_xticks(list(x), [GROUP_LABELS.get(g, g) for g in data["group"]])
    ax.set_ylabel("Number of questions")
    ax.set_xlabel("Question group")
    ax.set_ylim(0, data["count"].max() * 1.18)
    ax.legend(loc="upper right")
    return fig


def fig7_1_ex_by_model_difficulty(tables: Path, split: str = "test") -> Figure:
    """Execution accuracy by difficulty, C0 next to C4, one panel per model."""
    data = read(tables, "e1_ex_by_model_difficulty.csv")
    data = data[data["split"] == split]
    models, labels = model_order(tables), model_labels(tables)
    difficulties = ["easy", "medium", "hard"]
    fig, axes = _panels(len(models))
    for ax, model in zip(axes, models, strict=True):
        rows = data[data["model"] == model].set_index("difficulty")
        for offset, (config, style) in enumerate((("C0", 0), ("C4", 2))):
            values = [100 * rows.loc[d, f"ex_{config}"] for d in difficulties]
            _bar(ax, [i + (offset - 0.5) * 0.38 for i in range(3)], values, style,
                 f"{config}" if ax is axes[0] else None, width=0.36)
        ax.set_xticks(range(3), difficulties)
        ax.set_xlabel(labels.get(model, model))
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Execution accuracy (% of answerable questions)")
    axes[0].legend(title="Configuration", loc="upper right")
    return fig


def fig7_2_harmful_by_config(tables: Path) -> Figure:
    """Harmful answers for C0..C4 per model, with 95% bootstrap intervals."""
    data = read(tables, "e3_harmful_ci.csv")
    models, labels = model_order(tables), model_labels(tables)
    n = int(data["n"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.8 / len(models)
    for m, model in enumerate(models):
        rows = data[data["model"] == model].set_index("config").loc[CONFIGS]
        x = [i + (m - (len(models) - 1) / 2) * width for i in range(len(CONFIGS))]
        errors = [rows["harmful"] - rows["harmful_lo"], rows["harmful_hi"] - rows["harmful"]]
        _bar(ax, x, rows["harmful"], m, labels.get(model, model), width=width * 0.94,
             yerr=errors, capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax.set_xticks(range(len(CONFIGS)), CONFIGS)
    ax.set_xlabel("Configuration (C0 = no verification, C4 = full verification)")
    ax.set_ylabel(f"Harmful answers (of {n} test questions)")
    ax.set_ylim(0, None)
    ax.legend(title="Model", loc="upper right")
    return fig


def fig7_3_rs_vs_penalty(tables: Path, split: str = "test") -> Figure:
    """Reliability Score against the penalty c; one line per config, one panel per model.

    RS_c = RS_0 - c x (harmful answers as % of questions), so it is a straight line
    in c. It is drawn as the line through RS_0 and RS_10 on a true numeric axis,
    with markers at the three penalties the report uses (c = 0, 1, 10).
    """
    data = read(tables, "e2_rs_by_config_penalty.csv")
    data = data[data["split"] == split]
    models, labels = model_order(tables), model_labels(tables)
    penalties = [0, 1, 10]
    fig, axes = _panels(len(models), height=3.1)
    for ax, model in zip(axes, models, strict=True):
        rows = data[data["model"] == model].set_index("config")
        for c, config in enumerate(CONFIGS):
            start, end = rows.loc[config, "rs_0"], rows.loc[config, "rs_10"]
            style = {"color": "black", "linewidth": 1.1}
            ax.plot([0, 10], [start, end], linestyle=LINES[c], **style)
            ax.plot(penalties, [rows.loc[config, f"rs_{p}"] for p in penalties],
                    linestyle="none", marker=MARKERS[c], markersize=5,
                    markerfacecolor="white" if c % 2 else "black", markeredgecolor="black")
            # Legend entry with both the line style and the marker of this config.
            ax.plot([], [], linestyle=LINES[c], marker=MARKERS[c], markersize=5,
                    markerfacecolor="white" if c % 2 else "black", label=config, **style)
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_xlim(-0.4, 10.4)
        ax.set_xticks([0, 1, 5, 10])
        ax.set_xlabel(f"Penalty c for a harmful answer\n{labels.get(model, model)}")
    axes[0].set_ylabel("Reliability Score RS$_c$")
    axes[-1].legend(title="Configuration", loc="lower left")
    return fig


def fig7_4_taxonomy_stacked(tables: Path, split: str = "test") -> Figure:
    """Harmful answers by type at C0 and C4, stacked, per model."""
    data = read(tables, "e3_taxonomy.csv")
    data = data[data["split"] == split]
    models, labels = model_order(tables), model_labels(tables)
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    positions, names = [], []
    for m, model in enumerate(models):
        for c, config in enumerate(("C0", "C4")):
            positions.append(m * 2.6 + c)
            names.append(f"{config}\n{labels.get(model, model) if c == 0 else ''}")
    for style, (column, label) in enumerate(TAXONOMY):
        bottoms, heights = [], []
        for model in models:
            for config in ("C0", "C4"):
                row = data[(data["model"] == model) & (data["config"] == config)].iloc[0]
                heights.append(row[column])
                bottoms.append(sum(row[c] for c, _ in TAXONOMY[:style]))
        _bar(ax, positions, heights, style, label, bottom=bottoms)
    ax.set_xticks(positions, names)
    ax.set_ylabel("Harmful answers (test questions)")
    ax.set_xlabel("Configuration and model")
    # Outside the plot area: the tallest bars reach the top on the right.
    ax.legend(title="Type of harmful answer", loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8)
    return fig


def fig7_5_roc_signals(tables: Path) -> Figure:
    """ROC curves of the three abstention signals, one panel per model."""
    points = read(tables, "e4_roc_points.csv")
    auroc = read(tables, "e4_signal_auroc.csv")
    auroc = auroc[auroc["split"] == "test"]
    models, labels = model_order(tables), model_labels(tables)
    fig, axes = _panels(len(models), width=2.7, height=3.9)
    for ax, model in zip(axes, models, strict=True):
        for s, signal in enumerate(SIGNAL_LABELS):
            curve = points[(points["model"] == model) & (points["signal"] == signal)]
            score = auroc[(auroc["model"] == model) & (auroc["signal"] == signal)]["auroc"]
            label = f"{SIGNAL_LABELS[signal]}, AUROC {score.iloc[0]:.2f}" if len(score) else None
            ax.plot(curve["false_positive_rate"], curve["true_positive_rate"], color="black",
                    linestyle=LINES[s], marker=MARKERS[s], markevery=max(len(curve) // 6, 1),
                    markersize=4, linewidth=1.1, markerfacecolor="white", label=label)
        ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(f"False positive rate\n{labels.get(model, model)}")
        # Below each panel, so the legend never covers a curve.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=6.5,
                  handlelength=2.6)
    axes[0].set_ylabel("True positive rate")
    return fig


def fig7_6_risk_coverage(tables: Path, split: str = "test") -> Figure:
    """Risk against coverage as tau varies; each model's chosen tau is marked."""
    data = read(tables, "e4_risk_coverage.csv")
    data = data[(data["split"] == split) & data["risk"].notna()]
    chosen = read(tables, "tau_selection.csv")
    chosen = chosen[chosen["chosen"] == "yes"].set_index("model")["tau"]
    models, labels = model_order(tables), model_labels(tables)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for m, model in enumerate(models):
        rows = data[data["model"] == model].sort_values("tau")
        ax.plot(100 * rows["coverage"], 100 * rows["risk"], color="black", linestyle=LINES[m],
                marker=MARKERS[m], markersize=5, markerfacecolor="white", linewidth=1.1,
                label=labels.get(model, model))
        pick = rows[rows["tau"] == chosen.get(model)]
        if len(pick):
            x, y = 100 * pick["coverage"].iloc[0], 100 * pick["risk"].iloc[0]
            ax.plot(x, y, marker=MARKERS[m], markersize=10, color="black", linestyle="none")
            ax.annotate(f"τ = {chosen[model]:g}", (x, y), textcoords="offset points",
                        xytext=(8, 6), fontsize=8)
    ax.set_xlabel("Coverage (% of test questions answered)")
    ax.set_ylabel("Risk (% of answers that are harmful)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    # Upper left is empty: risk rises with coverage, so the curves run to the upper right.
    ax.legend(title="Model (large filled marker = τ chosen on dev)", loc="upper left",
              fontsize=8, title_fontsize=8)
    return fig


def fig7_7_infeasible_answered(tables: Path, split: str = "test") -> Figure:
    """Infeasible questions answered at C0 and C4, by trap group, per model."""
    data = read(tables, "e7_infeasible_by_group.csv")
    data = data[data["split"] == split]
    models, labels = model_order(tables), model_labels(tables)
    groups = ["T1", "T2", "T3", "T4"]
    fig, axes = _panels(len(models))
    for ax, model in zip(axes, models, strict=True):
        rows = data[data["model"] == model].set_index("group").loc[groups]
        for offset, (config, style) in enumerate((("C0", 0), ("C4", 2))):
            _bar(ax, [i + (offset - 0.5) * 0.38 for i in range(4)], rows[f"answered_{config}"],
                 style, config if ax is axes[0] else None, width=0.36)
        ticks = [f"{g}\nn={int(rows.loc[g, 'n'])}" for g in groups]
        ax.set_xticks(range(4), ticks, fontsize=8)
        ax.set_xlabel(labels.get(model, model))
        ax.set_ylim(0, max(int(rows["n"].max()), 1) * 1.05)
    axes[0].set_ylabel("Infeasible questions answered")
    axes[0].legend(title="Configuration", loc="upper left")
    return fig


FIGURES = {
    "fig4_2_benchmark_composition": fig4_2_benchmark_composition,
    "fig7_1_ex_by_model_difficulty": fig7_1_ex_by_model_difficulty,
    "fig7_2_harmful_by_config": fig7_2_harmful_by_config,
    "fig7_3_rs_vs_penalty": fig7_3_rs_vs_penalty,
    "fig7_4_taxonomy_stacked": fig7_4_taxonomy_stacked,
    "fig7_5_roc_signals": fig7_5_roc_signals,
    "fig7_6_risk_coverage": fig7_6_risk_coverage,
    "fig7_7_infeasible_answered": fig7_7_infeasible_answered,
}
