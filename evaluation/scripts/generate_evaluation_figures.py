from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_utils import FIGURES_DIR, TABLES_DIR, ensure_output_dirs, read_csv_rows, to_float

MODEL_ORDER = [
    "minimax/minimax-m2.7",
    "qwen/qwen3.5-9b",
    "z-ai/glm-5",
]

MODEL_LABELS = {
    "minimax/minimax-m2.7": "MiniMax M2.7",
    "qwen/qwen3.5-9b": "Qwen 3.5 9B",
    "z-ai/glm-5": "GLM-5",
}

EXPERIMENT_ORDER = ["Single-pass", "Multi-pass"]
EXPERIMENT_COLORS = {
    "Single-pass": "#2a6f97",
    "Multi-pass": "#c25b36",
}

LABEL_ORDER = ["closed", "closed_duration", "left_open", "right_open"]


def main() -> None:
    ensure_output_dirs()
    plt.style.use("seaborn-v0_8-whitegrid")

    model_rows = read_csv_rows(TABLES_DIR / "model_metrics.csv")
    per_label_rows = read_csv_rows(TABLES_DIR / "per_label_metrics.csv")

    plot_metric_comparison(
        model_rows,
        "micro_precision",
        "Gold-restricted Micro Precision",
        "Precision",
        "figure_01_micro_precision",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "micro_recall",
        "Gold-restricted Micro Recall",
        "Recall",
        "figure_02_micro_recall",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "micro_f1",
        "Gold-restricted Micro F1",
        "F1",
        "figure_03_micro_f1",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "macro_precision",
        "Gold-restricted Macro Precision",
        "Precision",
        "figure_04_macro_precision",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "macro_recall",
        "Gold-restricted Macro Recall",
        "Recall",
        "figure_05_macro_recall",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "macro_f1",
        "Gold-restricted Macro F1",
        "F1",
        "figure_06_macro_f1",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "weighted_f1",
        "Gold-restricted Weighted F1",
        "F1",
        "figure_07_weighted_f1",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "accuracy",
        "Gold-restricted Accuracy",
        "Accuracy",
        "figure_08_accuracy",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "sentence_exact_rate",
        "Sentence-level Exact Rate on Gold",
        "Rate",
        "figure_09_sentence_exact_rate",
        ymin=0.75,
        ymax=1.0,
    )
    plot_metric_comparison(
        model_rows,
        "ignored_extra_predictions_total",
        "Ignored Extra Predictions",
        "Count",
        "figure_10_ignored_extra_predictions_total",
        ymin=0.0,
        ymax=None,
    )
    plot_per_label_metric(
        per_label_rows,
        "f1",
        "Per-label F1",
        "F1",
        "figure_11_per_label_f1",
    )
    plot_error_profile(model_rows)

    print(f"[NEW EVAL FIGURES DONE] figures={FIGURES_DIR}", flush=True)


def plot_metric_comparison(
    rows: list[dict[str, str]],
    metric_key: str,
    title: str,
    ylabel: str,
    basename: str,
    *,
    ymin: float | None,
    ymax: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    lookup = {(row["experiment_label"], row["model_name"]): row for row in rows}
    x_positions = list(range(len(MODEL_ORDER)))
    bar_width = 0.36

    for offset_index, experiment_label in enumerate(EXPERIMENT_ORDER):
        offset = -bar_width / 2 if offset_index == 0 else bar_width / 2
        values = [to_float(lookup[(experiment_label, model_name)][metric_key]) for model_name in MODEL_ORDER]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_width,
            label=experiment_label,
            color=EXPERIMENT_COLORS[experiment_label],
        )
        add_bar_labels(ax, bars, "{:.3f}" if ymax else "{:.0f}")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_LABELS[name] for name in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if ymin is not None:
        top = ymax if ymax is not None else max(
            to_float(row[metric_key]) for row in rows
        ) * 1.15
        ax.set_ylim(ymin, top)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, basename)


def plot_per_label_metric(
    rows: list[dict[str, str]],
    metric_key: str,
    title: str,
    ylabel: str,
    basename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    subset = [row for row in rows if row["label"] in LABEL_ORDER]

    combo_order = [(experiment_label, model_name) for experiment_label in EXPERIMENT_ORDER for model_name in MODEL_ORDER]
    combo_labels = [
        f"{'S' if experiment_label == 'Single-pass' else 'M'}-{MODEL_LABELS[model_name]}"
        for experiment_label, model_name in combo_order
    ]
    combo_colors = [
        EXPERIMENT_COLORS[experiment_label] if model_name != "z-ai/glm-5" else (
            "#5b8e7d" if experiment_label == "Single-pass" else "#9c6644"
        )
        for experiment_label, model_name in combo_order
    ]

    x_positions = list(range(len(LABEL_ORDER)))
    total_width = 0.84
    bar_width = total_width / len(combo_order)
    start = -total_width / 2 + bar_width / 2

    lookup = {
        (row["experiment_label"], row["model_name"], row["label"]): row
        for row in subset
    }

    for idx, (experiment_label, model_name) in enumerate(combo_order):
        offset = start + idx * bar_width
        values = [
            to_float(lookup[(experiment_label, model_name, label)][metric_key])
            for label in LABEL_ORDER
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_width,
            label=combo_labels[idx],
            color=combo_colors[idx],
        )
        add_bar_labels(ax, bars, "{:.3f}", fontsize=7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.75, 1.0)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, basename)


def plot_error_profile(rows: list[dict[str, str]]) -> None:
    ordered_rows = sorted(
        rows,
        key=lambda row: (EXPERIMENT_ORDER.index(row["experiment_label"]), MODEL_ORDER.index(row["model_name"])),
    )
    labels = [f"{MODEL_LABELS[row['model_name']]}\n{row['experiment_label']}" for row in ordered_rows]
    missing = [to_float(row["missing_prediction_count"]) for row in ordered_rows]
    wrong = [to_float(row["wrong_allowed_label_count"]) for row in ordered_rows]
    multiple = [to_float(row["multiple_prediction_count"]) for row in ordered_rows]

    x_positions = list(range(len(ordered_rows)))
    fig, ax = plt.subplots(figsize=(10, 5.3))
    bars_missing = ax.bar(x_positions, missing, label="Missing span", color="#8d99ae")
    bars_wrong = ax.bar(
        x_positions,
        wrong,
        bottom=missing,
        label="Wrong allowed label",
        color="#d1495b",
    )
    bars_multiple = ax.bar(
        x_positions,
        multiple,
        bottom=[a + b for a, b in zip(missing, wrong)],
        label="Multiple/ambiguous labels",
        color="#6d597a",
    )

    totals = [a + b + c for a, b, c in zip(missing, wrong, multiple)]
    for x_pos, total in zip(x_positions, totals):
        ax.text(x_pos, total + max(totals) * 0.02 if max(totals) else 1, f"{int(total)}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Error Profile on the 1000 Gold Instances", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure_12_error_profile_counts")


def add_bar_labels(ax: plt.Axes, bars, fmt: str, fontsize: int = 9) -> None:
    heights = [bar.get_height() for bar in bars]
    if not heights:
        return
    max_height = max(heights)
    offset = max_height * 0.02 if max_height > 0 else 0.01
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def save_figure(fig: plt.Figure, basename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{basename}.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
