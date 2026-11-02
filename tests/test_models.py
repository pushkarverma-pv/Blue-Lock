import numpy as np
import pandas as pd
import pytest

from src.models import (
    _poisson_estimator,
    outcome_probabilities_from_grid,
    scoreline_probability_grid,
    scoreline_probability_grids,
)


def test_scoreline_grid_is_normalized_and_outcomes_match_grid() -> None:
    grid = scoreline_probability_grid(1.4, 0.9, max_goals=10)

    outcomes = outcome_probabilities_from_grid(grid)

    assert grid.shape == (11, 11)
    assert grid.sum() == pytest.approx(1)
    assert outcomes.sum() == pytest.approx(1)
    assert outcomes[0] > outcomes[2]


def test_scoreline_grid_batch_matches_single_match_grids() -> None:
    batch = scoreline_probability_grids([1.0, 2.0], [1.5, 0.5], max_goals=6)

    assert batch.shape == (2, 7, 7)
    assert np.allclose(batch.sum(axis=(1, 2)), 1)
    assert np.allclose(batch[0], scoreline_probability_grid(1.0, 1.5, max_goals=6))


@pytest.mark.parametrize(
    ("home", "away", "max_goals"),
    [(-0.1, 1, 10), (1, float("nan"), 10), (1, 1, -1)],
)
def test_scoreline_grid_rejects_invalid_values(
    home: float, away: float, max_goals: int
) -> None:
    with pytest.raises(ValueError):
        scoreline_probability_grid(home, away, max_goals=max_goals)


def test_scaled_poisson_regression_fits_goal_counts() -> None:
    features = pd.DataFrame({"form": [0.5, 1.0, 1.5, 2.0, 2.5]})
    goals = np.array([0, 1, 1, 2, 3])
    model = _poisson_estimator(alpha=0.1)

    model.fit(features, goals)
    prediction = model.predict(features)

    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
