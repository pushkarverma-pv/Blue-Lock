import numpy as np
import pandas as pd

from src.baselines import (
    bookmaker_probabilities,
    fit_baselines,
    outcome_labels,
    predict_baselines,
)


def _training_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FTHG": [2, 0, 1, 0, 3, 1],
            "FTAG": [0, 0, 1, 2, 1, 2],
            "elo_diff": [200, 0, 10, -150, 100, -40],
        }
    )


def test_outcome_labels_derive_from_goals() -> None:
    assert outcome_labels(_training_matches()).tolist() == ["H", "D", "D", "A", "H", "A"]


def test_fit_and_predict_baselines_return_normalized_probabilities() -> None:
    train = _training_matches()
    fitted = fit_baselines(train)
    predictions = predict_baselines(train, fitted)

    assert predictions["Always home win"].shape == (6, 3)
    assert predictions["Class frequency"].shape == (6, 3)
    assert predictions["Elo logistic"].shape == (6, 3)
    assert np.allclose(predictions["Class frequency"].sum(axis=1), 1)
    assert np.allclose(predictions["Elo logistic"].sum(axis=1), 1)
    assert np.isnan(predictions["Bookmaker"]).all()


def test_bookmaker_probabilities_remove_overround_and_flag_missing_rows() -> None:
    odds = pd.DataFrame(
        {
            "AvgH": [2.0, 0.0],
            "AvgD": [3.0, 3.0],
            "AvgA": [4.0, 4.0],
            "B365H": [2.2, 2.0],
            "B365D": [3.3, 4.0],
            "B365A": [4.4, 4.0],
        }
    )

    probabilities, valid = bookmaker_probabilities(odds)

    assert valid.tolist() == [True, True]
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.allclose(probabilities[0], np.array([1 / 2, 1 / 3, 1 / 4]) / (1 / 2 + 1 / 3 + 1 / 4))
    assert not np.allclose(probabilities[1], probabilities[0])


def test_bookmaker_probabilities_return_invalid_mask_when_no_odds() -> None:
    probabilities, valid = bookmaker_probabilities(pd.DataFrame({"FTHG": [1]}))

    assert not valid.any()
    assert np.isnan(probabilities).all()
