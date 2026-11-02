"""Append match predictions and settle them against played results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cleaning import normalize_team_name
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.evaluate import evaluate_probabilities

DEFAULT_LOG_PATH = REPORTS_DIR / "prediction_log.csv"
DEFAULT_MATCHES_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"
DEFAULT_METRICS_PATH = REPORTS_DIR / "prediction_log_metrics.md"
LOG_COLUMNS = (
    "prediction_made_at_utc",
    "match_date",
    "home_team",
    "away_team",
    "expected_home_goals",
    "expected_away_goals",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "actual_home_goals",
    "actual_away_goals",
    "actual_result",
    "settled_at_utc",
)


def append_prediction(
    prediction: dict[str, Any],
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one prediction API result to the persistent CSV log."""
    try:
        expected = prediction["expected_goals"]
        probabilities = prediction["outcome_probabilities"]
        row = {
            "prediction_made_at_utc": datetime.now(timezone.utc).isoformat(),
            "match_date": pd.Timestamp(prediction["date"]).date().isoformat(),
            "home_team": str(normalize_team_name(prediction["home_team"])),
            "away_team": str(normalize_team_name(prediction["away_team"])),
            "expected_home_goals": float(expected["home"]),
            "expected_away_goals": float(expected["away"]),
            "prob_home_win": float(probabilities["home_win"]),
            "prob_draw": float(probabilities["draw"]),
            "prob_away_win": float(probabilities["away_win"]),
            "actual_home_goals": np.nan,
            "actual_away_goals": np.nan,
            "actual_result": "",
            "settled_at_utc": "",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Prediction result has an invalid shape: {exc}") from exc

    numeric = [
        row["expected_home_goals"],
        row["expected_away_goals"],
        row["prob_home_win"],
        row["prob_draw"],
        row["prob_away_win"],
    ]
    if not np.isfinite(numeric).all() or min(numeric[:2]) < 0:
        raise ValueError("Prediction goals and probabilities must be finite and valid")
    probs = np.asarray(numeric[2:], dtype=float)
    if (probs < 0).any() or not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError("Outcome probabilities must be non-negative and sum to one")
    if not row["home_team"] or not row["away_team"] or row["home_team"] == row["away_team"]:
        raise ValueError("Prediction must contain two different non-empty teams")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_file():
        existing = pd.read_csv(log_path, low_memory=False)
        missing = [column for column in LOG_COLUMNS if column not in existing]
        if missing:
            raise ValueError(
                f"Prediction log is missing required columns: {', '.join(missing)}"
            )
    else:
        existing = pd.DataFrame(columns=LOG_COLUMNS)
    updated = pd.concat(
        [existing.loc[:, LOG_COLUMNS], pd.DataFrame([row], columns=LOG_COLUMNS)],
        ignore_index=True,
    )
    updated.to_csv(log_path, index=False)


def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


def _settled_metrics(log: pd.DataFrame) -> dict[str, float] | None:
    settled = log.loc[log["actual_result"].isin(["H", "D", "A"])]
    if settled.empty:
        return None
    probabilities = settled[
        ["prob_home_win", "prob_draw", "prob_away_win"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return evaluate_probabilities(
        settled["actual_result"].astype(str).to_numpy(),
        probabilities,
    )


def settle_prediction_log(
    log_path: Path = DEFAULT_LOG_PATH,
    matches_path: Path = DEFAULT_MATCHES_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """Fill played results once and write cumulative metrics for settled rows."""
    if not log_path.is_file():
        raise FileNotFoundError(f"Prediction log not found: {log_path}")
    if not matches_path.is_file():
        raise FileNotFoundError(f"Cleaned match history not found: {matches_path}")
    log = pd.read_csv(log_path, low_memory=False)
    missing_log = [column for column in LOG_COLUMNS if column not in log]
    if missing_log:
        raise ValueError(
            f"Prediction log is missing required columns: {', '.join(missing_log)}"
        )
    matches = pd.read_csv(matches_path, low_memory=False)
    required_matches = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing_matches = sorted(required_matches.difference(matches.columns))
    if missing_matches:
        raise ValueError(
            "Match history is missing required columns: "
            + ", ".join(missing_matches)
        )

    matches = matches.copy()
    matches["match_date_key"] = pd.to_datetime(matches["Date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    matches["home_team_key"] = matches["HomeTeam"].map(normalize_team_name).astype("string")
    matches["away_team_key"] = matches["AwayTeam"].map(normalize_team_name).astype("string")
    matches["home_goals_value"] = pd.to_numeric(matches["FTHG"], errors="coerce")
    matches["away_goals_value"] = pd.to_numeric(matches["FTAG"], errors="coerce")
    result_by_fixture: dict[tuple[str, str, str], tuple[int, int]] = {}
    for row in matches.dropna(
        subset=[
            "match_date_key",
            "home_team_key",
            "away_team_key",
            "home_goals_value",
            "away_goals_value",
        ]
    ).itertuples(index=False):
        key = (row.match_date_key, str(row.home_team_key), str(row.away_team_key))
        value = (int(row.home_goals_value), int(row.away_goals_value))
        if key in result_by_fixture and result_by_fixture[key] != value:
            raise ValueError(f"Conflicting actual results found for fixture {key}")
        result_by_fixture[key] = value

    log["actual_result"] = log["actual_result"].astype("string").fillna("")
    log["settled_at_utc"] = log["settled_at_utc"].astype("string").fillna("")
    unsettled = log["actual_result"].astype(str).eq("")
    settled_at = datetime.now(timezone.utc).isoformat()
    for index in log.index[unsettled]:
        row = log.loc[index]
        match_date = pd.to_datetime(row["match_date"], errors="coerce")
        if pd.isna(match_date):
            continue
        key = (
            pd.Timestamp(match_date).date().isoformat(),
            str(normalize_team_name(row["home_team"])),
            str(normalize_team_name(row["away_team"])),
        )
        actual = result_by_fixture.get(key)
        if actual is None:
            continue
        home_goals, away_goals = actual
        log.at[index, "actual_home_goals"] = home_goals
        log.at[index, "actual_away_goals"] = away_goals
        log.at[index, "actual_result"] = _result_label(home_goals, away_goals)
        log.at[index, "settled_at_utc"] = settled_at

    log.to_csv(log_path, index=False)
    metrics = _settled_metrics(log)
    if metrics is None:
        report = "# Prediction log metrics\n\nNo logged predictions have a matching played result yet.\n"
    else:
        settled_count = int(log["actual_result"].isin(["H", "D", "A"]).sum())
        report = (
            "# Prediction log metrics\n\n"
            f"Settled predictions: {settled_count}\n\n"
            "| Metric | Value |\n|---|---:|\n"
            f"| Log loss | {metrics['log_loss']:.6f} |\n"
            f"| Brier score | {metrics['brier_score']:.6f} |\n"
            f"| Accuracy | {metrics['accuracy']:.6f} |\n"
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Updated prediction log: {log_path}")
    print(f"Saved running metrics: {metrics_path}")
    return log, metrics


def main(argv: list[str] | None = None) -> int:
    """Settle predictions and report cumulative scores from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settle",
        action="store_true",
        help="Fill played results and report running log loss, Brier score, and accuracy.",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--metrics-report", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args(argv)
    if not args.settle:
        parser.error("specify --settle to update the prediction log")
    settle_prediction_log(args.log, args.matches, args.metrics_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
