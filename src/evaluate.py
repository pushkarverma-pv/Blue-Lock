"""Metrics and calibration tools for three-way football outcome predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

OUTCOME_ORDER = ("H", "D", "A")
_ORDINAL_ORDER = ("A", "D", "H")


def _validated_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate and normalize a three-class probability matrix."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(OUTCOME_ORDER):
        raise ValueError("probabilities must have shape (n_samples, 3) in H, D, A order")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("each probability row must have a positive sum")
    return values / totals[:, None]


def multiclass_rps(
    y_true: Sequence[str], probabilities: np.ndarray
) -> float:
    """Compute Ranked Probability Score, with ordered outcomes away/draw/home."""
    probs = _validated_probabilities(probabilities)
    labels = np.asarray(y_true, dtype=str)
    if len(labels) != len(probs):
        raise ValueError("y_true and probabilities must have matching lengths")
    if not set(labels).issubset(OUTCOME_ORDER):
        raise ValueError("y_true labels must be H, D, or A")
    ordinal_indices = [OUTCOME_ORDER.index(label) for label in _ORDINAL_ORDER]
    ordinal_probs = probs[:, ordinal_indices]
    ordinal_actual = np.column_stack(
        [(labels == label).astype(float) for label in _ORDINAL_ORDER]
    )
    cumulative_error = np.cumsum(ordinal_probs, axis=1)[:, :-1] - np.cumsum(
        ordinal_actual, axis=1
    )[:, :-1]
    return float(np.mean(np.mean(cumulative_error**2, axis=1)))


def goals_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return mean absolute error for predicted goals."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ValueError("actual and predicted goals must have matching non-empty shapes")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("goal values must be finite")
    return float(np.mean(np.abs(actual_values - predicted_values)))


def evaluate_probabilities(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    *,
    actual_goals: np.ndarray | None = None,
    predicted_goals: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate accuracy, log loss, multiclass Brier score, and RPS.

    The multiclass Brier score is the per-match sum of squared errors across
    classes. Optional goal arrays must have home/away columns.
    """
    probs = _validated_probabilities(probabilities)
    labels = np.asarray(y_true, dtype=str)
    if len(labels) != len(probs) or len(labels) == 0:
        raise ValueError("y_true and probabilities must have matching non-empty lengths")
    if not set(labels).issubset(OUTCOME_ORDER):
        raise ValueError("y_true labels must be H, D, or A")

    actual = np.column_stack([(labels == label).astype(float) for label in OUTCOME_ORDER])
    clipped = np.clip(probs, 1e-15, 1.0)
    sklearn_order = tuple(sorted(OUTCOME_ORDER))
    sklearn_probabilities = clipped[
        :, [OUTCOME_ORDER.index(label) for label in sklearn_order]
    ]
    metrics = {
        "accuracy": float(accuracy_score(labels, np.asarray(OUTCOME_ORDER)[probs.argmax(axis=1)])),
        "log_loss": float(
            log_loss(labels, sklearn_probabilities, labels=list(sklearn_order))
        ),
        "brier_score": float(np.mean(np.sum((probs - actual) ** 2, axis=1))),
        "rps": multiclass_rps(labels, probs),
    }
    if (actual_goals is None) != (predicted_goals is None):
        raise ValueError("actual_goals and predicted_goals must be supplied together")
    if actual_goals is not None and predicted_goals is not None:
        actual_goal_values = np.asarray(actual_goals, dtype=float)
        predicted_goal_values = np.asarray(predicted_goals, dtype=float)
        if (
            actual_goal_values.ndim != 2
            or predicted_goal_values.shape != actual_goal_values.shape
            or actual_goal_values.shape[1] != 2
        ):
            raise ValueError("goal arrays must have matching shape (n_samples, 2)")
        metrics["home_goals_mae"] = goals_mae(
            actual_goal_values[:, 0], predicted_goal_values[:, 0]
        )
        metrics["away_goals_mae"] = goals_mae(
            actual_goal_values[:, 1], predicted_goal_values[:, 1]
        )
    return metrics


def calibration_table(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create one-vs-rest calibration bins for home, draw, and away outcomes."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    probs = _validated_probabilities(probabilities)
    labels = np.asarray(y_true, dtype=str)
    if len(labels) != len(probs) or len(labels) == 0:
        raise ValueError("y_true and probabilities must have matching non-empty lengths")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    records: list[dict[str, float | int | str]] = []
    for index, outcome in enumerate(OUTCOME_ORDER):
        prediction = probs[:, index]
        assigned_bins = np.minimum(np.digitize(prediction, edges[1:-1]), n_bins - 1)
        for bin_index in range(n_bins):
            selected = assigned_bins == bin_index
            if selected.any():
                records.append(
                    {
                        "outcome": outcome,
                        "bin": bin_index,
                        "count": int(selected.sum()),
                        "mean_predicted": float(prediction[selected].mean()),
                        "observed_frequency": float((labels[selected] == outcome).mean()),
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "outcome",
            "bin",
            "count",
            "mean_predicted",
            "observed_frequency",
        ],
    )


def save_calibration_plot(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Outcome calibration",
) -> None:
    """Save a reliability diagram for a calibration table."""
    required = {"outcome", "mean_predicted", "observed_frequency"}
    if not required.issubset(table.columns):
        raise ValueError(f"calibration table must contain: {', '.join(sorted(required))}")
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for outcome in OUTCOME_ORDER:
        rows = table.loc[table["outcome"] == outcome].sort_values("mean_predicted")
        if not rows.empty:
            axis.plot(
                rows["mean_predicted"],
                rows["observed_frequency"],
                marker="o",
                label={"H": "Home win", "D": "Draw", "A": "Away win"}[outcome],
            )
    axis.set(
        title=title,
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_calibration_comparison_plot(
    raw_table: pd.DataFrame,
    calibrated_table: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Raw and calibrated outcome reliability",
    calibrated_label: str = "calibrated",
) -> None:
    """Plot raw and calibrated one-vs-rest reliability curves by outcome."""
    required = {"outcome", "mean_predicted", "observed_frequency"}
    if not required.issubset(raw_table.columns) or not required.issubset(
        calibrated_table.columns
    ):
        raise ValueError("calibration tables are missing required columns")
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    colors = {"H": "tab:blue", "D": "tab:orange", "A": "tab:green"}
    labels = {"H": "Home win", "D": "Draw", "A": "Away win"}
    for outcome in OUTCOME_ORDER:
        for table, style, suffix in (
            (raw_table, "--", "raw"),
            (calibrated_table, "-", calibrated_label),
        ):
            rows = table.loc[table["outcome"] == outcome].sort_values(
                "mean_predicted"
            )
            if not rows.empty:
                axis.plot(
                    rows["mean_predicted"],
                    rows["observed_frequency"],
                    marker="o",
                    linestyle=style,
                    color=colors[outcome],
                    label=f"{labels[outcome]} ({suffix})",
                )
    axis.set(
        title=title,
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


__all__ = [
    "OUTCOME_ORDER",
    "calibration_table",
    "evaluate_probabilities",
    "goals_mae",
    "multiclass_rps",
    "save_calibration_comparison_plot",
    "save_calibration_plot",
]
