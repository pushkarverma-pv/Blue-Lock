from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calibration_table,
    evaluate_probabilities,
    goals_mae,
    multiclass_rps,
    save_calibration_plot,
)


def test_outcome_metrics_for_certain_correct_predictions() -> None:
    labels = np.array(["H", "D", "A"])
    probabilities = np.eye(3)

    scores = evaluate_probabilities(labels, probabilities)

    assert scores["accuracy"] == 1
    assert scores["log_loss"] < 1e-12
    assert scores["brier_score"] == 0
    assert scores["rps"] == 0


def test_rps_respects_ordered_outcomes() -> None:
    labels = ["H"]

    near = multiclass_rps(labels, np.array([[0.7, 0.2, 0.1]]))
    far = multiclass_rps(labels, np.array([[0.1, 0.2, 0.7]]))

    assert near < far


def test_goals_mae_and_optional_goal_metrics() -> None:
    actual = np.array([[1, 0], [2, 2]])
    predicted = np.array([[2, 0], [1, 3]])

    assert goals_mae(actual[:, 0], predicted[:, 0]) == 1
    scores = evaluate_probabilities(
        ["H", "D"],
        np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
        actual_goals=actual,
        predicted_goals=predicted,
    )
    assert scores["home_goals_mae"] == 1
    assert scores["away_goals_mae"] == 0.5


def test_calibration_table_and_plot(tmp_path: Path) -> None:
    table = calibration_table(
        ["H", "D", "A", "H"],
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.4, 0.3, 0.3],
            ]
        ),
        n_bins=5,
    )
    output = tmp_path / "calibration.png"

    assert set(table["outcome"]) == {"H", "D", "A"}
    assert table["count"].sum() == 12
    save_calibration_plot(table, output)
    assert output.is_file()


def test_invalid_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_probabilities(["H"], np.array([[1.1, -0.1, 0]]))
