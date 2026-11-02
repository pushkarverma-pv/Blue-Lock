"""Prediction API for expected goals, scorelines, and match outcomes."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.cleaning import normalize_team_name
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.features import build_features
from src.models import outcome_probabilities_from_grid, scoreline_probability_grid

MODEL_PATH = MODELS_DIR / "poisson_model.joblib"
HISTORY_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"
MAX_GOALS = 10
STALE_HISTORY_DAYS = 14
_HISTORY_COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "season",
)


def latest_season_teams(history: pd.DataFrame) -> list[str]:
    """Return teams present in the newest numeric season code in history."""
    required = {"season", "HomeTeam", "AwayTeam"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"History is missing season/team columns: {', '.join(missing)}")
    season_codes = pd.to_numeric(history["season"], errors="coerce")
    if season_codes.isna().all():
        raise ValueError("History contains no valid numeric season codes")
    latest_mask = season_codes == season_codes.max()
    latest = history.loc[latest_mask]
    teams = set(latest["HomeTeam"].dropna().map(normalize_team_name).astype(str))
    teams.update(latest["AwayTeam"].dropna().map(normalize_team_name).astype(str))
    if not teams:
        raise ValueError("No teams were found in the most recent season")
    return sorted(teams)


def latest_match_date(history: pd.DataFrame) -> pd.Timestamp:
    """Return the most recent valid match date from historical match rows."""
    if "Date" not in history:
        raise ValueError("History is missing the Date column")
    dates = pd.to_datetime(history["Date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("History contains no valid match dates")
    return pd.Timestamp(dates.max()).normalize()


def history_is_stale(
    match_date: str | pd.Timestamp,
    last_match_date: str | pd.Timestamp,
    *,
    stale_after_days: int = STALE_HISTORY_DAYS,
) -> bool:
    """Whether a prediction is more than the allowed days after the latest match."""
    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative")
    prediction_date = pd.Timestamp(match_date).normalize()
    latest_date = pd.Timestamp(last_match_date).normalize()
    return (prediction_date - latest_date).days > stale_after_days


def _season_code(match_date: pd.Timestamp) -> str:
    """Return the Premier League season code containing a calendar date."""
    start_year = match_date.year if match_date.month >= 7 else match_date.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _prediction_date(value: str | None, history: pd.DataFrame) -> pd.Timestamp:
    """Parse a prediction date, defaulting to the day after the last result."""
    if value is None:
        if history.empty:
            raise ValueError("Cannot infer a prediction date without historical matches")
        return pd.to_datetime(history["Date"]).max().normalize() + pd.Timedelta(days=1)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid prediction date: {value!r}; use YYYY-MM-DD")
    return pd.Timestamp(parsed).normalize()


def _validate_artifact(artifact: Any) -> dict[str, Any]:
    """Validate model artifact structure before inference."""
    if not isinstance(artifact, dict):
        raise ValueError("Saved model artifact has an invalid format")
    required = {"model_type", "home_goals_model", "away_goals_model", "feature_names"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(
            "Saved model artifact is missing: " + ", ".join(sorted(missing))
        )
    if artifact["model_type"] != "Poisson regression":
        raise ValueError(f"Unsupported production model: {artifact['model_type']!r}")
    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Saved model artifact has an invalid feature list")
    return artifact


def _fixture_feature_row(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a fixture's feature row from strict prior history only."""
    prior = history.loc[pd.to_datetime(history["Date"]) < match_date].copy()
    if prior.empty:
        prior = history.iloc[0:0].copy()
    fixture = {column: np.nan for column in history.columns}
    fixture.update(
        {
            "Date": match_date,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            # The fixture outcome placeholders are never used by its own features.
            "FTHG": 0,
            "FTAG": 0,
            "season": _season_code(match_date),
        }
    )
    combined = pd.concat([prior, pd.DataFrame([fixture])], ignore_index=True)
    enriched = build_features(combined, feature_positions=[len(combined) - 1])
    if enriched.empty:
        raise RuntimeError("Feature builder did not return the requested fixture")
    return enriched


def _top_scorelines(grid: np.ndarray, count: int = 5) -> list[dict[str, float | int]]:
    """Return the highest-probability scorelines in deterministic order."""
    scores = [
        (float(grid[home, away]), home, away)
        for home in range(grid.shape[0])
        for away in range(grid.shape[1])
    ]
    scores.sort(key=lambda result: (-result[0], result[1], result[2]))
    return [
        {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "score": f"{home_goals}-{away_goals}",
            "probability": probability,
        }
        for probability, home_goals, away_goals in scores[:count]
    ]


def predict(
    home_team: str, away_team: str, date: str | None = None
) -> dict[str, object]:
    """Predict a match from the fitted Poisson model and prior match history.

    The returned probabilities describe uncertainty; the most-likely score is
    only the highest-probability scoreline, not a guarantee.
    """
    if not home_team or not home_team.strip():
        raise ValueError("home_team must be a non-empty team name")
    if not away_team or not away_team.strip():
        raise ValueError("away_team must be a non-empty team name")
    home = str(normalize_team_name(home_team))
    away = str(normalize_team_name(away_team))
    if home.casefold() == away.casefold():
        raise ValueError("Home and away teams must be different")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Production model not found at {MODEL_PATH}; run `python -m src.train` first."
        )
    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Match history not found at {HISTORY_PATH}; run `python -m src.eda` first."
        )

    artifact = _validate_artifact(joblib.load(MODEL_PATH))
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    missing_history = [column for column in _HISTORY_COLUMNS if column not in history]
    if missing_history:
        raise ValueError(
            "Match history is missing required columns: "
            + ", ".join(missing_history)
        )
    available_teams = sorted(
        set(history["HomeTeam"].dropna().astype(str))
        | set(history["AwayTeam"].dropna().astype(str))
    )
    normalized_teams = {str(normalize_team_name(team)) for team in available_teams}
    if home not in normalized_teams:
        raise ValueError(
            f"Unknown home team {home_team!r}. Available teams: "
            + ", ".join(sorted(normalized_teams))
        )
    if away not in normalized_teams:
        raise ValueError(
            f"Unknown away team {away_team!r}. Available teams: "
            + ", ".join(sorted(normalized_teams))
        )

    kickoff = _prediction_date(date, history)
    latest_date = latest_match_date(history)
    fixture_features = _fixture_feature_row(history, home, away, kickoff)
    feature_names = artifact["feature_names"]
    missing_features = [
        name for name in feature_names if name not in fixture_features.columns
    ]
    if missing_features:
        raise ValueError(
            "Feature generation did not produce: " + ", ".join(missing_features)
        )
    X = fixture_features.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Prediction features contain missing or non-finite values")

    home_expected = float(
        np.clip(artifact["home_goals_model"].predict(X)[0], 0.0, None)
    )
    away_expected = float(
        np.clip(artifact["away_goals_model"].predict(X)[0], 0.0, None)
    )
    grid = scoreline_probability_grid(
        home_expected, away_expected, max_goals=int(artifact.get("max_goals", MAX_GOALS))
    )
    outcome_probabilities = outcome_probabilities_from_grid(grid)
    top_scores = _top_scorelines(grid)
    outcomes = {
        "home_win": float(outcome_probabilities[0]),
        "draw": float(outcome_probabilities[1]),
        "away_win": float(outcome_probabilities[2]),
    }
    return {
        "home_team": home,
        "away_team": away,
        "date": kickoff.date().isoformat(),
        "history_last_match_date": latest_date.date().isoformat(),
        "history_age_days": int((kickoff - latest_date).days),
        "history_stale": history_is_stale(kickoff, latest_date),
        "expected_goals": {
            "home": home_expected,
            "away": away_expected,
        },
        "outcome_probabilities": outcomes,
        "top_scorelines": top_scores,
        "most_likely_score": top_scores[0]["score"],
        "scoreline_grid": grid.tolist(),
        "goals_range": list(range(grid.shape[0])),
        "note": "These are model probabilities, not guarantees; the most-likely score is only the single highest-probability scoreline.",
    }


__all__ = ["predict"]
