"""Generate exploratory summaries and plots from cleaned match data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.cleaning import DEFAULT_INPUT_PATH, clean_matches
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"
SUMMARY_PATH = REPORTS_DIR / "eda_summary.md"
GOAL_DISTRIBUTION_PLOT = REPORTS_DIR / "goal_distribution.png"
HOME_ADVANTAGE_PLOT = REPORTS_DIR / "home_advantage.png"


def _outcome_percentages(matches: pd.DataFrame) -> pd.Series:
    """Return full-time home/draw/away outcome percentages."""
    outcomes = matches["FTR"].astype("string").str.upper()
    counts = outcomes.value_counts().reindex(["H", "D", "A"], fill_value=0)
    if counts.sum() == 0:
        raise ValueError("No valid full-time result values are available for EDA")
    return counts / counts.sum() * 100


def _team_goal_averages(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate scored and conceded goals per appearance for each team."""
    home = matches[["HomeTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "team", "FTHG": "scored", "FTAG": "conceded"}
    )
    away = matches[["AwayTeam", "FTAG", "FTHG"]].rename(
        columns={"AwayTeam": "team", "FTAG": "scored", "FTHG": "conceded"}
    )
    appearances = pd.concat([home, away], ignore_index=True)
    return (
        appearances.groupby("team", as_index=False)[["scored", "conceded"]]
        .mean()
        .rename(
            columns={
                "scored": "avg_goals_scored_per_match",
                "conceded": "avg_goals_conceded_per_match",
            }
        )
        .sort_values("team")
        .reset_index(drop=True)
    )


def _goal_distribution(matches: pd.DataFrame) -> tuple[pd.Series, float]:
    """Return observed total-goals counts and their fitted Poisson mean."""
    total_goals = matches["FTHG"] + matches["FTAG"]
    mean_goals = float(total_goals.mean())
    return total_goals.value_counts().sort_index(), mean_goals


def _home_advantage_by_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate home-minus-away average goals by season."""
    if "season" not in matches:
        raise ValueError("A season column is required for the home-advantage summary")
    return (
        matches.groupby("season", as_index=False)
        .agg(home_goals=("FTHG", "mean"), away_goals=("FTAG", "mean"))
        .assign(home_advantage=lambda data: data["home_goals"] - data["away_goals"])
        .sort_values("season")
        .reset_index(drop=True)
    )


def _write_goal_distribution_plot(
    matches: pd.DataFrame, output_path: Path
) -> None:
    """Plot observed total-goal frequencies against a fitted Poisson curve."""
    counts, mean_goals = _goal_distribution(matches)
    total_goals = matches["FTHG"] + matches["FTAG"]
    maximum = max(int(total_goals.max()), 1)
    values = np.arange(0, maximum + 1)
    probabilities = poisson.pmf(values, mean_goals)
    observed_probabilities = counts.reindex(values, fill_value=0) / len(matches)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(values, observed_probabilities, label="Observed")
    axis.plot(values, probabilities, "o-", label=f"Poisson fit (mean={mean_goals:.2f})")
    axis.set(title="Match total goals: observed vs fitted Poisson", xlabel="Total goals", ylabel="Probability")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_home_advantage_plot(matches: pd.DataFrame, output_path: Path) -> None:
    """Plot the seasonal difference in average home and away goals."""
    advantage = _home_advantage_by_season(matches)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(advantage["season"].astype(str), advantage["home_advantage"], marker="o")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        title="Home-goal advantage by season",
        xlabel="Season code",
        ylabel="Average home goals minus away goals",
    )
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def generate_eda_report(
    matches: pd.DataFrame,
    report_path: Path = SUMMARY_PATH,
    goal_plot_path: Path = GOAL_DISTRIBUTION_PLOT,
    home_advantage_plot_path: Path = HOME_ADVANTAGE_PLOT,
) -> str:
    """Create EDA plots and a markdown summary from cleaned match rows."""
    required = {"FTR", "FTHG", "FTAG", "HomeTeam", "AwayTeam", "season"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Missing columns required for EDA: {', '.join(missing)}")
    if matches.empty:
        raise ValueError("Cannot generate EDA for an empty match dataset")

    outcomes = _outcome_percentages(matches)
    team_averages = _team_goal_averages(matches)
    goal_counts, mean_goals = _goal_distribution(matches)
    advantage = _home_advantage_by_season(matches)
    _write_goal_distribution_plot(matches, goal_plot_path)
    _write_home_advantage_plot(matches, home_advantage_plot_path)

    outcome_rows = "\n".join(
        f"| {label} | {outcomes[code]:.2f}% |"
        for code, label in (("H", "Home win"), ("D", "Draw"), ("A", "Away win"))
    )
    team_rows = "\n".join(
        f"| {row.team} | {row.avg_goals_scored_per_match:.3f} | "
        f"{row.avg_goals_conceded_per_match:.3f} |"
        for row in team_averages.itertuples(index=False)
    )
    season_rows = "\n".join(
        f"| {row.season} | {row.home_goals:.3f} | {row.away_goals:.3f} | "
        f"{row.home_advantage:.3f} |"
        for row in advantage.itertuples(index=False)
    )
    max_goal_count = max(10, int(goal_counts.index.max()))
    distribution_rows = "\n".join(
        f"| {goals} | {goal_counts.get(goals, 0) / len(matches):.4f} | "
        f"{poisson.pmf(goals, mean_goals):.4f} |"
        for goals in range(max_goal_count + 1)
    )
    summary = f"""# Exploratory data analysis

## Dataset

- Matches: {len(matches)}
- Date range: {matches["Date"].min().date()} to {matches["Date"].max().date()}
- Seasons: {matches["season"].nunique()}
- Total goals per match: mean {mean_goals:.3f}

## Full-time outcomes

| Outcome | Percentage |
|---|---:|
{outcome_rows}

## Average goals per team appearance

| Team | Scored | Conceded |
|---|---:|---:|
{team_rows}

## Match total-goals distribution

The Poisson comparison is fitted using the observed mean total goals ({mean_goals:.3f}).

| Total goals | Observed probability | Poisson probability |
|---:|---:|---:|
{distribution_rows}

![Observed total goals compared with a fitted Poisson distribution](goal_distribution.png)

## Home advantage by season

Home advantage is average home goals minus average away goals; positive values mean more home goals.

| Season | Home goals | Away goals | Home advantage |
|---|---:|---:|---:|
{season_rows}

![Home-goal advantage by season](home_advantage.png)
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved EDA summary to {report_path}")
    print(f"Saved goal distribution plot to {goal_plot_path}")
    print(f"Saved home advantage plot to {home_advantage_plot_path}")
    return summary


def run_eda(
    input_path: Path = DEFAULT_INPUT_PATH,
    cleaned_path: Path = CLEANED_DATA_PATH,
) -> tuple[pd.DataFrame, str]:
    """Clean the collected matches, save cleaned data, and generate EDA outputs."""
    source = pd.read_csv(input_path, low_memory=False)
    cleaned, cleaning_summary = clean_matches(source)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False, date_format="%Y-%m-%d")
    print(f"Input rows: {cleaning_summary.input_rows}")
    print(f"Rows with missing goals dropped: {cleaning_summary.missing_goal_rows_dropped}")
    print(f"Team-name rows normalized: {cleaning_summary.team_names_normalized}")
    print(f"Clean rows: {cleaning_summary.output_rows}")
    print(f"Saved cleaned data to {cleaned_path}")
    return cleaned, generate_eda_report(cleaned)


def main(argv: Sequence[str] | None = None) -> int:
    """Run cleaning and EDA from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--cleaned-output", type=Path, default=CLEANED_DATA_PATH)
    args = parser.parse_args(argv)
    run_eda(args.input, args.cleaned_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
