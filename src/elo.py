"""Leakage-safe Elo ratings for football match histories."""

from __future__ import annotations

import math

import pandas as pd


def _result_score(goals_for: int, goals_against: int) -> float:
    """Return the standard Elo score for a match: win, draw, or loss."""
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def _margin_factor(goal_difference: int, rating_difference: float, scale: float) -> float:
    """Return a smooth goal-margin multiplier, with scale zero disabling it."""
    if scale < 0:
        raise ValueError("goal_difference_scale must be non-negative")
    if goal_difference <= 1 or scale == 0:
        standard_factor = 1.0
    else:
        standard_factor = math.log(goal_difference + 1) * (
            2.2 / (2.2 + 0.001 * abs(rating_difference))
        )
    return 1.0 + scale * (standard_factor - 1.0)


def pre_match_ratings(
    matches: pd.DataFrame,
    *,
    k_factor: float = 20.0,
    home_advantage: float = 60.0,
    goal_difference_scale: float = 1.0,
    season_regression: float = 0.25,
    initial_rating: float = 1500.0,
    date_column: str = "Date",
    home_team_column: str = "HomeTeam",
    away_team_column: str = "AwayTeam",
    home_goals_column: str = "FTHG",
    away_goals_column: str = "FTAG",
    season_column: str = "season",
) -> pd.DataFrame:
    """Return each match's home and away Elo immediately before kickoff.

    All matches on a calendar date receive ratings from before any result on
    that date is applied. Ratings regress toward the initial rating when the
    season changes. Input row order and index are preserved.
    """
    required = (
        date_column,
        home_team_column,
        away_team_column,
        home_goals_column,
        away_goals_column,
    )
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing columns required for Elo: {', '.join(missing)}")
    if matches.empty:
        return pd.DataFrame(
            {"home_elo": pd.Series(dtype=float), "away_elo": pd.Series(dtype=float),
             "elo_diff": pd.Series(dtype=float)},
            index=matches.index,
        )
    if k_factor <= 0:
        raise ValueError("k_factor must be positive")
    if not 0 <= season_regression <= 1:
        raise ValueError("season_regression must be between 0 and 1")
    if goal_difference_scale < 0:
        raise ValueError("goal_difference_scale must be non-negative")

    frame = matches.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError("Elo input contains invalid dates")
    if frame[[home_team_column, away_team_column]].isna().any().any():
        raise ValueError("Elo input contains missing team names")
    for column in (home_goals_column, away_goals_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[[home_goals_column, away_goals_column]].isna().any().any():
        raise ValueError("Elo input contains missing full-time goals")
    if (frame[[home_goals_column, away_goals_column]] < 0).any().any():
        raise ValueError("Elo goals must be non-negative")

    frame["_original_position"] = range(len(frame))
    sort_columns = [date_column]
    if season_column in frame.columns:
        sort_columns.append(season_column)
    frame = frame.sort_values(sort_columns + ["_original_position"], kind="stable")

    ratings: dict[str, float] = {}
    prior_season: object | None = None
    results: dict[int, tuple[float, float]] = {}
    for _, date_matches in frame.groupby(date_column, sort=True):
        current_seasons = (
            date_matches[season_column].drop_duplicates().tolist()
            if season_column in date_matches
            else []
        )
        current_season = current_seasons[0] if current_seasons else None
        if (
            current_season is not None
            and prior_season is not None
            and current_season != prior_season
        ):
            ratings = {
                team: initial_rating
                + (rating - initial_rating) * (1.0 - season_regression)
                for team, rating in ratings.items()
            }
        if current_season is not None:
            prior_season = current_season

        updates: list[tuple[str, str, float, float]] = []
        for row in date_matches.itertuples(index=False, name=None):
            values = dict(zip(date_matches.columns, row))
            position = int(values["_original_position"])
            home_team = str(values[home_team_column])
            away_team = str(values[away_team_column])
            home_rating = ratings.get(home_team, initial_rating)
            away_rating = ratings.get(away_team, initial_rating)
            results[position] = (home_rating, away_rating)

            effective_home = home_rating + home_advantage
            expected_home = 1.0 / (
                1.0 + 10.0 ** ((away_rating - effective_home) / 400.0)
            )
            actual_home = _result_score(
                int(values[home_goals_column]), int(values[away_goals_column])
            )
            margin = abs(int(values[home_goals_column]) - int(values[away_goals_column]))
            factor = _margin_factor(
                margin, effective_home - away_rating, goal_difference_scale
            )
            change = k_factor * factor * (actual_home - expected_home)
            updates.append((home_team, away_team, change, -change))

        for home_team, away_team, home_change, away_change in updates:
            ratings[home_team] = ratings.get(home_team, initial_rating) + home_change
            ratings[away_team] = ratings.get(away_team, initial_rating) + away_change

    ordered = [results[position] for position in range(len(matches))]
    rating_frame = pd.DataFrame(
        ordered,
        columns=["home_elo", "away_elo"],
        index=matches.index,
    )
    rating_frame["elo_diff"] = rating_frame["home_elo"] - rating_frame["away_elo"]
    return rating_frame


def add_pre_match_ratings(
    matches: pd.DataFrame,
    **elo_options: float,
) -> pd.DataFrame:
    """Return a copy of matches with pre-match Elo rating columns appended."""
    return matches.join(pre_match_ratings(matches, **elo_options))


__all__ = ["add_pre_match_ratings", "pre_match_ratings"]
