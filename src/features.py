"""Leakage-safe historical match feature engineering."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.elo import pre_match_ratings

INPUT_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "features.parquet"
SUMMARY_PATH = REPORTS_DIR / "feature_summary.md"
CONTEXTS = ("overall", "home", "away")
WINDOWS = (5, 10)
STATISTICS = (
    "goals_scored",
    "goals_conceded",
    "shots",
    "shots_on_target",
    "points",
)

_DEFAULT_LEAGUE_VALUES = {
    "goals_scored": 1.35,
    "goals_conceded": 1.35,
    "shots": 12.0,
    "shots_on_target": 4.0,
    "points": 1.35,
}
_STAT_COLUMNS = {
    "shots": {"home": "HS", "away": "AS"},
    "shots_on_target": {"home": "HST", "away": "AST"},
}
_REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "season")


def _numeric_stat(row: dict[str, object], column: str | None) -> float:
    """Get a finite numeric stat from a row, or NaN when unavailable."""
    if column is None:
        return float("nan")
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _team_match_stats(
    row: dict[str, object], side: str
) -> dict[str, float]:
    """Build one team's observed stats from a completed match."""
    goals_for = _numeric_stat(row, "FTHG" if side == "home" else "FTAG")
    goals_against = _numeric_stat(row, "FTAG" if side == "home" else "FTHG")
    if goals_for > goals_against:
        points = 3.0
    elif goals_for == goals_against:
        points = 1.0
    else:
        points = 0.0
    return {
        "goals_scored": goals_for,
        "goals_conceded": goals_against,
        "shots": _numeric_stat(row, _STAT_COLUMNS["shots"][side]),
        "shots_on_target": _numeric_stat(row, _STAT_COLUMNS["shots_on_target"][side]),
        "points": points,
    }


def _fallback_value(
    statistic: str,
    context: str,
    league_totals: dict[str, dict[str, tuple[float, int]]],
) -> float:
    """Use only previously observed league stats, with a neutral cold-start prior."""
    total, count = league_totals[context][statistic]
    if count == 0 and context != "overall":
        total, count = league_totals["overall"][statistic]
    if count:
        return total / count
    return _DEFAULT_LEAGUE_VALUES[statistic]


def _rolling_mean(
    history: list[dict[str, float]],
    statistic: str,
    window: int,
    fallback: float,
) -> float:
    """Compute a past-only rolling mean, falling back when no observations exist."""
    observed = [
        event[statistic]
        for event in history[-window:]
        if np.isfinite(event[statistic])
    ]
    return float(np.mean(observed)) if observed else fallback


def feature_columns() -> list[str]:
    """Return generated form, Elo, rest, season-count, and history-flag columns."""
    columns = [
        f"{role}_{context}_{statistic}_last_{window}"
        for role in ("home", "away")
        for context in CONTEXTS
        for window in WINDOWS
        for statistic in STATISTICS
    ]
    columns.extend(
        [
            "home_elo",
            "away_elo",
            "elo_diff",
            "home_days_rest",
            "away_days_rest",
            "home_matches_played_season",
            "away_matches_played_season",
            "home_first_appearance",
            "away_first_appearance",
        ]
    )
    return columns


def build_features(
    matches: pd.DataFrame,
    *,
    feature_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Build pre-match form, Elo, rest, and season-count features for each row.

    All rolling history is updated only after every fixture on a date has
    received its features. ``feature_positions`` can restrict feature output
    while still replaying the complete history, which is useful for one-off
    inference. Missing historical shots are ignored in rolling means; first
    appearances fall back to prior league averages or neutral cold-start
    priors.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in matches]
    if missing:
        raise ValueError(f"Missing columns required for features: {', '.join(missing)}")
    if matches.empty:
        return matches.copy()

    frame = matches.copy().reset_index(drop=True)
    if feature_positions is None:
        selected_positions = set(range(len(frame)))
    else:
        selected_positions = set(feature_positions)
        invalid_positions = [
            position
            for position in selected_positions
            if not isinstance(position, int) or position < 0 or position >= len(frame)
        ]
        if invalid_positions:
            raise ValueError(
                f"Feature positions must be integers in [0, {len(frame) - 1}]"
            )
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("Feature input contains invalid dates")
    for column in ("HomeTeam", "AwayTeam"):
        frame[column] = frame[column].astype("string").str.strip()
    if frame[["HomeTeam", "AwayTeam"]].isna().any().any():
        raise ValueError("Feature input contains missing team names")
    for column in ("FTHG", "FTAG"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["FTHG", "FTAG"]].isna().any().any():
        raise ValueError("Feature input must contain completed matches with full-time goals")
    frame["season"] = frame["season"].astype("string")

    ratings = pre_match_ratings(frame)
    frame = frame.join(ratings)
    frame["_feature_position"] = np.arange(len(frame))
    processing_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "season",
        "HS",
        "AS",
        "HST",
        "AST",
        "home_elo",
        "away_elo",
        "elo_diff",
        "_feature_position",
    ]
    ordered = frame.loc[
        :, [column for column in processing_columns if column in frame.columns]
    ].sort_values(["Date", "season", "_feature_position"], kind="stable")

    team_history: defaultdict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {context: [] for context in CONTEXTS}
    )
    league_totals: dict[str, dict[str, tuple[float, int]]] = {
        context: {statistic: (0.0, 0) for statistic in STATISTICS}
        for context in CONTEXTS
    }
    played_by_season: defaultdict[tuple[str, str], int] = defaultdict(int)
    last_played_date: dict[str, pd.Timestamp] = {}
    prior_rest_days: list[float] = []
    records: dict[int, dict[str, float | int | bool]] = {}

    for match_date, date_matches in ordered.groupby("Date", sort=True):
        date_events: list[
            tuple[str, str, dict[str, float], dict[str, float], str]
        ] = []
        prior_rest_mean = (
            float(np.mean(prior_rest_days)) if prior_rest_days else 7.0
        )
        fallback_values = {
            (context, statistic): _fallback_value(
                statistic, context, league_totals
            )
            for context in CONTEXTS
            for statistic in STATISTICS
        }
        for row_values in date_matches.itertuples(index=False, name=None):
            row = dict(zip(date_matches.columns, row_values))
            position = int(row["_feature_position"])
            home_team = str(row["HomeTeam"])
            away_team = str(row["AwayTeam"])
            season = str(row["season"])
            if position in selected_positions:
                home_history = team_history[home_team]
                away_history = team_history[away_team]
                values: dict[str, float | int | bool] = {}

                for role, histories in (("home", home_history), ("away", away_history)):
                    for context in CONTEXTS:
                        history = histories[context]
                        for window in WINDOWS:
                            for statistic in STATISTICS:
                                values[
                                    f"{role}_{context}_{statistic}_last_{window}"
                                ] = _rolling_mean(
                                    history,
                                    statistic,
                                    window,
                                    fallback_values[(context, statistic)],
                                )

                values["home_days_rest"] = float(
                    (match_date - last_played_date[home_team]).total_seconds() / 86400
                ) if home_team in last_played_date else prior_rest_mean
                values["away_days_rest"] = float(
                    (match_date - last_played_date[away_team]).total_seconds() / 86400
                ) if away_team in last_played_date else prior_rest_mean
                values["home_matches_played_season"] = played_by_season[
                    (season, home_team)
                ]
                values["away_matches_played_season"] = played_by_season[
                    (season, away_team)
                ]
                values["home_first_appearance"] = not bool(home_history["overall"])
                values["away_first_appearance"] = not bool(away_history["overall"])
                records[position] = values

            home_stats = _team_match_stats(row, "home")
            away_stats = _team_match_stats(row, "away")
            date_events.append(
                (home_team, away_team, home_stats, away_stats, season)
            )

        # Apply match results after the date group so same-day fixtures cannot
        # observe one another's outcomes.
        teams_on_date: set[str] = set()
        for home_team, away_team, home_stats, away_stats, season in date_events:
            for team, stats, context in (
                (home_team, home_stats, "home"),
                (away_team, away_stats, "away"),
            ):
                team_history[team]["overall"].append(stats)
                team_history[team][context].append(stats)
                for statistic in STATISTICS:
                    value = stats[statistic]
                    if np.isfinite(value):
                        total, count = league_totals["overall"][statistic]
                        league_totals["overall"][statistic] = (total + value, count + 1)
                        total, count = league_totals[context][statistic]
                        league_totals[context][statistic] = (total + value, count + 1)
                played_by_season[(season, team)] += 1
                teams_on_date.add(team)

        for team in teams_on_date:
            previous_date = last_played_date.get(team)
            if previous_date is not None:
                prior_rest_days.append(
                    float((match_date - previous_date).total_seconds() / 86400)
                )
            last_played_date[team] = match_date

    output_positions = sorted(records)
    generated = pd.DataFrame([records[position] for position in output_positions])
    feature_frame = pd.concat(
        [
            frame.iloc[output_positions].drop(columns=["_feature_position"]).reset_index(
                drop=True
            ),
            generated,
        ],
        axis=1,
    )
    return feature_frame.sort_values("Date", kind="stable").reset_index(drop=True)


def feature_nan_counts(features: pd.DataFrame) -> pd.Series:
    """Count missing values in generated features only."""
    columns = [column for column in feature_columns() if column in features.columns]
    return features[columns].isna().sum()


def save_feature_summary(features: pd.DataFrame, output_path: Path = SUMMARY_PATH) -> str:
    """Write generated-feature missing counts and fallback notes to Markdown."""
    counts = feature_nan_counts(features)
    table = "\n".join(f"| {column} | {int(count)} |" for column, count in counts.items())
    report = f"""# Feature engineering summary

- Matches: {len(features)}
- Date range: {features["Date"].min().date()} to {features["Date"].max().date()}
- Generated features: {len(counts)}
- All match features are calculated from matches strictly before the match date.
- Same-day fixtures share a pre-date snapshot; their results are applied only after every match on that date receives features.
- Form windows use available prior matches up to their window size. First appearances use an expanding prior league/context average, or a neutral cold-start prior before any league results exist.
- Elo is also recorded before the current match result is applied.

## Missing values by generated feature

| Feature | Missing values |
|---|---:|
{table}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved feature summary to {output_path}")
    return report


def run_feature_engineering(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    """Load cleaned match data, create the feature table, and save artifacts."""
    matches = pd.read_csv(input_path, low_memory=False)
    features = build_features(matches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    save_feature_summary(features, summary_path)
    print(f"Saved {len(features)} rows and {len(feature_columns())} generated features to {output_path}")
    return features


def main(argv: Sequence[str] | None = None) -> int:
    """Run feature engineering from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    run_feature_engineering(args.input, args.output, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
