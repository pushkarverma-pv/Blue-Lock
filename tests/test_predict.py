import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import build_features, feature_columns
from src.models import _poisson_estimator
from src.predict import (
    history_is_stale,
    latest_match_date,
    latest_season_teams,
    predict,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-08",
                    "2023-08-08",
                    "2023-08-15",
                    "2023-08-22",
                ]
            ),
            "HomeTeam": ["Arsenal", "Liverpool", "Man City", "Arsenal", "Liverpool"],
            "AwayTeam": ["Liverpool", "Arsenal", "Arsenal", "Man City", "Man City"],
            "FTHG": [2, 1, 0, 3, 1],
            "FTAG": [0, 1, 1, 0, 2],
            "HS": [12, 10, 8, 14, 11],
            "AS": [4, 9, 8, 5, 10],
            "HST": [5, 4, 2, 7, 4],
            "AST": [2, 3, 3, 2, 4],
            "season": ["2324"] * 5,
            "league": ["E0"] * 5,
        }
    )


def _write_artifacts(tmp_path, monkeypatch) -> pd.DataFrame:
    history = _history()
    built = build_features(history)
    columns = feature_columns()
    X = built.loc[:, columns]
    model_home = _poisson_estimator(1.0).fit(X, built["FTHG"])
    model_away = _poisson_estimator(1.0).fit(X, built["FTAG"])
    model_path = tmp_path / "poisson_model.joblib"
    history_path = tmp_path / "cleaned_matches.csv"
    joblib.dump(
        {
            "model_type": "Poisson regression",
            "home_goals_model": model_home,
            "away_goals_model": model_away,
            "feature_names": columns,
            "league": "E0",
            "max_goals": 10,
        },
        model_path,
    )
    history.to_csv(history_path, index=False)
    monkeypatch.setattr("src.predict.MODEL_PATH", model_path)
    monkeypatch.setattr("src.predict.HISTORY_PATH", history_path)
    return history


def test_predict_returns_expected_goals_outcomes_scorelines_and_grid(
    tmp_path, monkeypatch
) -> None:
    _write_artifacts(tmp_path, monkeypatch)

    result = predict("Arsenal", "Liverpool", "2023-08-29")

    assert result["date"] == "2023-08-29"
    assert result["history_last_match_date"] == "2023-08-22"
    assert result["history_age_days"] == 7
    assert result["history_stale"] is False
    assert set(result["expected_goals"]) == {"home", "away"}
    assert set(result["outcome_probabilities"]) == {"home_win", "draw", "away_win"}
    assert sum(result["outcome_probabilities"].values()) == pytest.approx(1)
    assert len(result["top_scorelines"]) == 5
    assert result["most_likely_score"] == result["top_scorelines"][0]["score"]
    assert np.asarray(result["scoreline_grid"]).sum() == pytest.approx(1)
    assert "not guarantees" in result["note"]


def test_predict_default_date_follows_history_and_excludes_same_day(
    tmp_path, monkeypatch
) -> None:
    history = _write_artifacts(tmp_path, monkeypatch)

    result = predict("Arsenal", "Liverpool")

    assert result["date"] == "2023-08-23"
    assert result["expected_goals"]["home"] >= 0
    assert history["Date"].max() < pd.Timestamp(result["date"])


def test_predict_normalizes_alias_and_rejects_unknown_team(
    tmp_path, monkeypatch
) -> None:
    history = _history()
    history.loc[2, "HomeTeam"] = "Man City"
    built = build_features(history)
    columns = feature_columns()
    home_model = _poisson_estimator(1.0).fit(
        built.loc[:, columns], built["FTHG"]
    )
    away_model = _poisson_estimator(1.0).fit(
        built.loc[:, columns], built["FTAG"]
    )
    model_path = tmp_path / "poisson_model.joblib"
    history_path = tmp_path / "cleaned_matches.csv"
    joblib.dump(
        {
            "model_type": "Poisson regression",
            "home_goals_model": home_model,
            "away_goals_model": away_model,
            "feature_names": columns,
            "league": "E0",
            "max_goals": 10,
        },
        model_path,
    )
    history.to_csv(history_path, index=False)
    monkeypatch.setattr("src.predict.MODEL_PATH", model_path)
    monkeypatch.setattr("src.predict.HISTORY_PATH", history_path)

    result = predict("Manchester City", "Arsenal", "2023-08-29")

    assert result["home_team"] == "Man City"
    with pytest.raises(ValueError, match="Unknown home team"):
        predict("Not a team", "Arsenal", "2023-08-29")


def test_predict_rejects_same_teams_and_invalid_dates(tmp_path, monkeypatch) -> None:
    _write_artifacts(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="different"):
        predict("Arsenal", "Arsenal", "2023-08-29")
    with pytest.raises(ValueError, match="Invalid prediction date"):
        predict("Arsenal", "Liverpool", "not-a-date")


def test_latest_season_teams_excludes_historical_teams() -> None:
    history = _history()
    history["season"] = ["2324"] * 4 + ["2425"]
    history.loc[4, "HomeTeam"] = "Newly Promoted"

    teams = latest_season_teams(history)

    assert "Newly Promoted" in teams
    assert "Arsenal" not in teams
    assert set(teams) == {"Man City", "Newly Promoted"}


def test_latest_match_date_and_stale_history_threshold() -> None:
    latest = latest_match_date(_history())

    assert latest == pd.Timestamp("2023-08-22")
    assert not history_is_stale("2023-09-05", latest)
    assert history_is_stale("2023-09-06", latest)
