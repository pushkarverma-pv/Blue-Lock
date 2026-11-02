import numpy as np
import pandas as pd
import pytest

from src.prediction_log import append_prediction, settle_prediction_log


def _prediction() -> dict:
    return {
        "date": "2026-08-21",
        "home_team": "Arsenal",
        "away_team": "Liverpool",
        "expected_goals": {"home": 1.4, "away": 0.9},
        "outcome_probabilities": {
            "home_win": 0.6,
            "draw": 0.25,
            "away_win": 0.15,
        },
    }


def test_prediction_log_appends_predictions_and_settles_played_fixtures(
    tmp_path,
) -> None:
    log_path = tmp_path / "reports" / "prediction_log.csv"
    matches_path = tmp_path / "cleaned_matches.csv"
    metrics_path = tmp_path / "reports" / "metrics.md"
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Date": "2026-08-21",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Liverpool",
                "FTHG": 2,
                "FTAG": 1,
            }
        ]
    ).to_csv(matches_path, index=False)

    append_prediction(_prediction(), log_path)
    append_prediction(_prediction(), log_path)
    settled, metrics = settle_prediction_log(
        log_path, matches_path, metrics_path
    )

    assert len(settled) == 2
    assert settled["actual_result"].tolist() == ["H", "H"]
    assert settled["actual_home_goals"].tolist() == [2, 2]
    assert metrics is not None
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["log_loss"] == pytest.approx(-np.log(0.6))
    assert metrics["brier_score"] == pytest.approx(0.245)
    assert metrics_path.is_file()
    assert "Settled predictions: 2" in metrics_path.read_text(encoding="utf-8")

    second_settle, repeated_metrics = settle_prediction_log(
        log_path, matches_path, metrics_path
    )
    assert second_settle["actual_result"].tolist() == ["H", "H"]
    assert repeated_metrics == metrics


def test_prediction_log_rejects_invalid_probabilities(tmp_path) -> None:
    prediction = _prediction()
    prediction["outcome_probabilities"]["away_win"] = 0.2

    with pytest.raises(ValueError, match="sum to one"):
        append_prediction(prediction, tmp_path / "prediction_log.csv")
