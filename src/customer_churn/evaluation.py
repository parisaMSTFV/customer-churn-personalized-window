from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"navy": "#17324D", "blue": "#2F6690", "orange": "#F28E2B", "gray": "#9AA5B1"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    frame = comparison.sort_values("pr_auc")
    best_model = comparison.loc[comparison["pr_auc"].idxmax(), "model"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [COLORS["orange"] if name == best_model else COLORS["blue"] for name in frame["model"]]
    ax.barh(frame["model"], frame["pr_auc"], color=colors)
    ax.set_xlabel("Validation PR-AUC")
    ax.set_title("Predicting missed personalized purchase windows")
    ax.set_xlim(0, min(1.0, max(frame["pr_auc"]) + 0.12))
    for index, value in enumerate(frame["pr_auc"]):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    frame = importance.head(10).sort_values("mean_pr_auc_decrease")
    labels = frame["feature"].str.replace("_", " ").str.capitalize()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.barh(labels, frame["mean_pr_auc_decrease"], color=COLORS["blue"])
    ax.set_xlabel("Mean decrease in test PR-AUC")
    ax.set_title("Permutation importance on the time holdout")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], label="Perfect")
    ax.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_churn_rate"],
        marker="o",
        color=COLORS["orange"],
        linewidth=2,
        label="Calibrated model",
    )
    ax.set(xlabel="Predicted probability", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Probability calibration")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_cumulative_gain(y_true: pd.Series, probabilities: np.ndarray, path: Path) -> None:
    order = np.argsort(-probabilities)
    sorted_target = np.asarray(y_true)[order]
    customer_share = np.arange(1, len(sorted_target) + 1) / len(sorted_target)
    captured = np.cumsum(sorted_target) / max(sorted_target.sum(), 1)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(customer_share, captured, color=COLORS["orange"], linewidth=2.4, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], label="Random")
    ax.set(
        xlabel="Share of scored customers contacted",
        ylabel="Share of missed windows captured",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.set_title("Cumulative gain on the time holdout")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_personalized_windows(dataset: pd.DataFrame, path: Path) -> None:
    sample = dataset.sample(min(1_500, len(dataset)), random_state=42)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter = ax.scatter(
        sample["expected_gap_days"],
        sample["personalized_window_days"],
        c=sample["cadence_cv"],
        cmap="Blues",
        alpha=0.45,
        s=18,
    )
    ax.plot([0, 140], [0, 140], linestyle="--", color=COLORS["gray"])
    ax.set(
        xlabel="Expected purchase gap (days)",
        ylabel="Personalized deadline (days)",
        xlim=(0, 140),
        ylim=(0, 140),
    )
    ax.set_title("Different customers receive different purchase windows")
    fig.colorbar(scatter, ax=ax, label="Cadence variability")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def plot_priority_comparison(comparison: pd.DataFrame, path: Path) -> None:
    pivot = comparison.pivot(index="capacity", columns="strategy", values="captured_value_share")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    positions = np.arange(len(pivot))
    width = 0.34
    ax.bar(
        positions - width / 2,
        pivot["Risk only"],
        width,
        label="Risk only",
        color=COLORS["blue"],
    )
    ax.bar(
        positions + width / 2,
        pivot["Value at risk"],
        width,
        label="Value at risk",
        color=COLORS["orange"],
    )
    ax.set_xticks(positions, [f"{fraction:.0%}" for fraction in pivot.index])
    ax.set(xlabel="Campaign capacity", ylabel="Share of held-out margin proxy captured")
    ax.set_title("Historical prioritization against an outcome proxy")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)
