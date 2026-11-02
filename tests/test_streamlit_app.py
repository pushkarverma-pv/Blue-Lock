import numpy as np
import pytest

from app.streamlit_app import _scoreline_probabilities


def test_scoreline_probabilities_are_consistent_with_grid() -> None:
    grid = np.full((4, 4), 1 / 16)

    probabilities = _scoreline_probabilities(grid, "Man United", "Aston Villa")

    assert probabilities["Over 2.5 goals"] == pytest.approx(10 / 16)
    assert probabilities["Under 2.5 goals"] == pytest.approx(6 / 16)
    assert (
        probabilities["Over 2.5 goals"] + probabilities["Under 2.5 goals"]
        == pytest.approx(1.0)
    )
    assert probabilities["Both teams to score"] == pytest.approx(9 / 16)
    assert probabilities["Man United clean sheet"] == pytest.approx(4 / 16)
    assert probabilities["Aston Villa clean sheet"] == pytest.approx(4 / 16)


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[0.5, 0.25], [0.25, 0.25]]),
        np.array([[0.5, -0.5], [0.5, 0.5]]),
        np.array([[0.5, np.nan], [0.25, 0.25]]),
        np.ones((2, 3)) / 6,
    ],
)
def test_scoreline_probabilities_reject_invalid_grids(grid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="normalized square probability matrix"):
        _scoreline_probabilities(grid, "Home", "Away")
