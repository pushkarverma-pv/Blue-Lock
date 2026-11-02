"""Time-aware football outcome baselines and bookmaker benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR, SEASONS
from src.elo import pre_match_ratings
from src.evaluate import (
    OUTCOME_ORDER,
    calibration_table,
    evaluate_probabilities,
    save_calibration_plot,
)

CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"
REPORT_PATH = REPORTS_DIR / "baselines.md"
TRAIN_SEASONS = tuple(SEASONS[:7])
VALIDATION_SEASONS = ("2223",)
TEST_SEASONS = ("2324", "2425", "2526")
BOOKMAKER_COLUMNS = (
    ("AvgH", "AvgD", "AvgA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
)
BASELINE_NAMES = (
    "Always home win",
    "Class frequency",
    "Elo logistic",
    "Bookmaker",
)


@dataclass(frozen=True)
class FittedBaselines:
    """Baseline parameters estimated using training-season outcomes only."""

    class_frequencies: np.ndarray
    elo_classifier: LogisticRegression


def outcome_labels(matches: pd.DataFrame) -> np.ndarray:
    """Derive full-time home/draw/away labels from goals."""
    required = {"FTHG", "FTAG"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Missing goal columns: {', '.join(missing)}")
    home = pd.to_numeric(matches["FTHG"], errors="coerce")
    away = pd.to_numeric(matches["FTAG"], errors="coerce")
    if home.isna().any() or away.isna().any():
        raise ValueError("Cannot derive outcomes from missing full-time goals")
    return np.select([home > away, home == away], ["H", "D"], default="A")


def bookmaker_probabilities(matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized no-vig bookmaker probabilities in H/D/A order.

    The function chooses the first available complete odds market per row in
    this order: average odds, Pinnacle odds, then Bet365. Rows without valid
    positive odds are marked false in the returned validity mask.
    """
    probabilities = np.full((len(matches), len(OUTCOME_ORDER)), np.nan, dtype=float)
    valid = np.zeros(len(matches), dtype=bool)
    for columns in BOOKMAKER_COLUMNS:
        if not set(columns).issubset(matches.columns):
            continue
        odds = matches.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
        market_valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
        use_rows = market_valid & ~valid
        if not use_rows.any():
            continue
        inverse = 1.0 / odds[use_rows]
        probabilities[use_rows] = inverse / inverse.sum(axis=1, keepdims=True)
        valid[use_rows] = True
    return probabilities, valid


def fit_baselines(
    train: pd.DataFrame,
    *,
    elo_diff_column: str = "elo_diff",
) -> FittedBaselines:
    """Fit class-frequency and Elo-to-outcome mappings on training matches."""
    if train.empty:
        raise ValueError("Training data cannot be empty")
    if elo_diff_column not in train.columns:
        raise ValueError(f"Missing Elo feature column: {elo_diff_column}")
    labels = outcome_labels(train)
    frequencies = np.array(
        [(labels == outcome).mean() for outcome in OUTCOME_ORDER], dtype=float
    )
    elo_diff = pd.to_numeric(train[elo_diff_column], errors="coerce").to_numpy()
    if not np.isfinite(elo_diff).all():
        raise ValueError("Training Elo differences must be finite")
    classifier = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs",
    )
    classifier.fit((elo_diff / 400.0).reshape(-1, 1), labels)
    return FittedBaselines(frequencies, classifier)


def _align_classifier_probabilities(
    classifier: LogisticRegression, features: np.ndarray
) -> np.ndarray:
    """Align sklearn class probabilities to the public H/D/A order."""
    raw = classifier.predict_proba(features)
    aligned = np.zeros((len(features), len(OUTCOME_ORDER)), dtype=float)
    for source_index, label in enumerate(classifier.classes_):
        aligned[:, OUTCOME_ORDER.index(str(label))] = raw[:, source_index]
    return aligned


def predict_baselines(
    matches: pd.DataFrame,
    fitted: FittedBaselines,
    *,
    elo_diff_column: str = "elo_diff",
) -> dict[str, np.ndarray]:
    """Produce baseline probabilities for a set of matches."""
    if elo_diff_column not in matches.columns:
        raise ValueError(f"Missing Elo feature column: {elo_diff_column}")
    elo_diff = pd.to_numeric(matches[elo_diff_column], errors="coerce").to_numpy()
    if not np.isfinite(elo_diff).all():
        raise ValueError("Elo differences must be finite")
    home_win = np.zeros((len(matches), len(OUTCOME_ORDER)), dtype=float)
    home_win[:, OUTCOME_ORDER.index("H")] = 1.0
    predictions: dict[str, np.ndarray] = {
        "Always home win": home_win,
        "Class frequency": np.tile(fitted.class_frequencies, (len(matches), 1)),
        "Elo logistic": _align_classifier_probabilities(
            fitted.elo_classifier, (elo_diff / 400.0).reshape(-1, 1)
        ),
    }
    bookmaker, _ = bookmaker_probabilities(matches)
    predictions["Bookmaker"] = bookmaker
    return predictions


def _metric_row(
    name: str,
    split: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, str | int | float]:
    """Convert a baseline prediction into a table row."""
    metrics = evaluate_probabilities(labels, probabilities)
    return {
        "Baseline": name,
        "Split": split,
        "N": len(labels),
        **metrics,
    }


def run_baseline_evaluation(
    matches: pd.DataFrame,
    *,
    report_path: Path = REPORT_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> pd.DataFrame:
    """Fit on 1516-2122 and evaluate baselines on validation and test seasons."""
    required = {"season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Missing columns required for baseline evaluation: {', '.join(missing)}")

    frame = matches.copy()
    frame["season"] = frame["season"].astype("string")
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    frame = frame.join(pre_match_ratings(frame))
    train = frame.loc[frame["season"].isin(TRAIN_SEASONS)].copy()
    if train.empty:
        raise ValueError(f"No training rows found for seasons {TRAIN_SEASONS}")
    fitted = fit_baselines(train)
    split_rows: list[dict[str, str | int | float]] = []
    split_calibrations: dict[str, list[pd.DataFrame]] = {}
    for split_name, split_seasons in (
        ("Validation", VALIDATION_SEASONS),
        ("Test", TEST_SEASONS),
    ):
        split = frame.loc[frame["season"].isin(split_seasons)].copy()
        if split.empty:
            raise ValueError(f"No rows found for {split_name.lower()} seasons {split_seasons}")
        labels = outcome_labels(split)
        predictions = predict_baselines(split, fitted)
        for baseline in BASELINE_NAMES:
            probabilities = predictions[baseline]
            if baseline == "Bookmaker":
                valid = np.isfinite(probabilities).all(axis=1)
                if not valid.any():
                    split_rows.append(
                        {
                            "Baseline": baseline,
                            "Split": split_name,
                            "N": 0,
                            "accuracy": np.nan,
                            "log_loss": np.nan,
                            "brier_score": np.nan,
                            "rps": np.nan,
                        }
                    )
                    continue
                split_rows.append(
                    _metric_row(baseline, split_name, labels[valid], probabilities[valid])
                )
            else:
                split_rows.append(_metric_row(baseline, split_name, labels, probabilities))

        for model_name in ("Elo logistic", "Bookmaker"):
            probabilities = predictions[model_name]
            valid = np.isfinite(probabilities).all(axis=1)
            if valid.any():
                table = calibration_table(labels[valid], probabilities[valid])
                table.insert(0, "model", model_name)
                split_calibrations.setdefault(split_name, []).append(table)

    results = pd.DataFrame.from_records(split_rows)
    results = results[
        ["Baseline", "Split", "N", "accuracy", "log_loss", "brier_score", "rps"]
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline evaluation",
        "",
        f"- Training seasons: {', '.join(TRAIN_SEASONS)}",
        f"- Validation season: {', '.join(VALIDATION_SEASONS)}",
        f"- Test seasons: {', '.join(TEST_SEASONS)}",
        "- All predictions and probability mappings were fit using training seasons only.",
        "- Elo ratings are generated pre-match using K=20, a 60-point home advantage, a 1.0 goal-margin scale, and 25% season regression; past results update subsequent ratings.",
        "- Bookmaker probabilities use inverse odds normalized to remove the overround; bookmaker rows without valid odds are excluded.",
        "- RPS uses the ordered outcomes away win, draw, home win. Brier score sums squared errors across the three classes.",
        "",
        "## Results",
        "",
        "| Baseline | Split | N | Accuracy | Log loss | Brier score | RPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        metric_values = []
        for metric in ("accuracy", "log_loss", "brier_score", "rps"):
            value = getattr(row, metric)
            metric_values.append("—" if pd.isna(value) else f"{value:.4f}")
        lines.append(
            f"| {row.Baseline} | {row.Split} | {row.N} | "
            + " | ".join(metric_values)
            + " |"
        )

    for split_name, calibration_tables in split_calibrations.items():
        calibration = pd.concat(calibration_tables, ignore_index=True)
        csv_path = reports_dir / f"calibration_{split_name.lower()}.csv"
        plot_path = reports_dir / f"calibration_{split_name.lower()}.png"
        calibration.to_csv(csv_path, index=False)
        elo_calibration = calibration.loc[calibration["model"] == "Elo logistic"]
        save_calibration_plot(
            elo_calibration,
            plot_path,
            title=f"Elo logistic calibration: {split_name.lower()}",
        )
        lines.extend(
            [
                "",
                f"## {split_name} Elo calibration",
                "",
                f"Reliability plot: [{plot_path.name}]({plot_path.name})",
                f"Calibration data: [{csv_path.name}]({csv_path.name})",
                "",
            ]
        )

    report = "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved baseline report to {report_path}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Load cleaned match data and generate the baseline report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEANED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    matches = pd.read_csv(args.input, low_memory=False)
    run_baseline_evaluation(matches, report_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
