"""Calibration, subgroup error analysis, and model interpretation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.baselines import (
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    fit_baselines,
    outcome_labels,
    predict_baselines,
)
from src.calibration import MulticlassPlattCalibrator, expected_calibration_error
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.evaluate import (
    OUTCOME_ORDER,
    calibration_table,
    evaluate_probabilities,
    save_calibration_comparison_plot,
)
from src.models import fit_model_suite, predict_model_suite

INPUT_PATH = PROCESSED_DATA_DIR / "features.parquet"
REPORT_PATH = REPORTS_DIR / "analysis.md"
MODEL_NAMES = (
    "Elo logistic",
    "Poisson regression",
    "XGBoost Poisson goals",
    "XGBoost direct W/D/L",
)
BIG_SIX = {
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Man City",
    "Man United",
    "Tottenham",
}


def _frame_probabilities(
    model_suite: object,
    baseline_models: object,
    frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Combine fitted baseline and model-suite predictions for one split."""
    outputs = predict_model_suite(model_suite, frame)
    baselines = predict_baselines(frame, baseline_models)
    return {
        **baselines,
        **{
            name: prediction["probabilities"]
            for name, prediction in outputs.items()
        },
    }


def _metric_triplet(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Return metrics used for calibration selection and subgroup tables."""
    metrics = evaluate_probabilities(labels, probabilities)
    return {
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "brier_score": metrics["brier_score"],
        "rps": metrics["rps"],
    }


def _calibration_table_with_tags(
    model_name: str,
    version: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Create reliability bins tagged with their model and calibration variant."""
    table = calibration_table(labels, probabilities)
    table.insert(0, "variant", version)
    table.insert(0, "model", model_name)
    return table


def _test_subgroups(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Build deterministic test subsets for season, phase, promotion, and big six."""
    groups: list[tuple[str, np.ndarray]] = []
    for season in sorted(frame["season"].astype(str).unique()):
        groups.append((f"Season {season}", frame["season"].astype(str).eq(season).to_numpy()))

    appearances = np.maximum(
        pd.to_numeric(frame["home_matches_played_season"]).to_numpy(),
        pd.to_numeric(frame["away_matches_played_season"]).to_numpy(),
    )
    groups.extend(
        [
            ("Early season (<5 prior matches)", appearances < 5),
            ("Later season", appearances >= 5),
        ]
    )

    season_order = (
        frame[["season", "Date"]]
        .drop_duplicates("season")
        .sort_values("Date")["season"]
        .astype(str)
        .tolist()
    )
    teams_by_season = {
        season: set(
            frame.loc[frame["season"].astype(str) == season, "HomeTeam"].astype(str)
        )
        | set(
            frame.loc[frame["season"].astype(str) == season, "AwayTeam"].astype(str)
        )
        for season in season_order
    }
    promoted_by_season: dict[str, set[str]] = {}
    previous_teams: set[str] = set()
    for season in season_order:
        promoted_by_season[season] = (
            teams_by_season[season] - previous_teams if previous_teams else set()
        )
        previous_teams = teams_by_season[season]
    promoted_involved = np.array(
        [
            str(home) in promoted_by_season.get(str(season), set())
            or str(away) in promoted_by_season.get(str(season), set())
            for season, home, away in zip(
                frame["season"], frame["HomeTeam"], frame["AwayTeam"]
            )
        ],
        dtype=bool,
    )
    big_six_match = frame["HomeTeam"].astype(str).isin(BIG_SIX).to_numpy() | frame[
        "AwayTeam"
    ].astype(str).isin(BIG_SIX).to_numpy()
    groups.extend(
        [
            ("Promoted team involved", promoted_involved),
            ("No promoted team involved", ~promoted_involved),
            ("Big-six match", big_six_match),
            ("Other match", ~big_six_match),
        ]
    )
    return groups


def _subgroup_metric_rows(
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, str | int | float]]:
    """Compute selected-model scores for each test error-analysis group."""
    records: list[dict[str, str | int | float]] = []
    for group_name, mask in _test_subgroups(frame):
        if not mask.any():
            continue
        metrics = _metric_triplet(labels[mask], probabilities[mask])
        records.append({"Group": group_name, "N": int(mask.sum()), **metrics})
    return records


def _feature_interpretation(model_suite: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize averaged XGBoost importances and standardized Poisson coefficients."""
    models = [
        model_suite.xgb_home,
        model_suite.xgb_away,
        model_suite.xgb_classifier,
    ]
    importances = np.mean(
        np.vstack([model.feature_importances_ for model in models]), axis=0
    )
    importance = pd.DataFrame(
        {"feature": model_suite.feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    home_coefficients = model_suite.poisson_home.named_steps[
        "poissonregressor"
    ].coef_
    away_coefficients = model_suite.poisson_away.named_steps[
        "poissonregressor"
    ].coef_
    coefficients = pd.DataFrame(
        {
            "feature": model_suite.feature_names,
            "home_goal_coefficient": home_coefficients,
            "away_goal_coefficient": away_coefficients,
            "mean_absolute_coefficient": (
                np.abs(home_coefficients) + np.abs(away_coefficients)
            )
            / 2,
        }
    ).sort_values("mean_absolute_coefficient", ascending=False)
    return importance, coefficients


def run_model_analysis(
    matches: pd.DataFrame,
    *,
    report_path: Path = REPORT_PATH,
    reports_dir: Path = REPORTS_DIR,
    cv_splits: int = 3,
) -> str:
    """Fit on training seasons, calibrate on validation, and analyze held-out test."""
    required = {
        "season",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "home_matches_played_season",
        "away_matches_played_season",
    }
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Missing columns required for analysis: {', '.join(missing)}")
    frame = matches.copy()
    frame["season"] = frame["season"].astype("string")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("Analysis data contains invalid dates")
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    labels = outcome_labels(frame)

    train_mask = frame["season"].isin(TRAIN_SEASONS).to_numpy()
    validation_mask = frame["season"].isin(VALIDATION_SEASONS).to_numpy()
    test_mask = frame["season"].isin(TEST_SEASONS).to_numpy()
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Training, validation, and test seasons must all be present")
    train = frame.loc[train_mask]
    train_labels = labels[train_mask]
    model_suite = fit_model_suite(
        train,
        pd.to_numeric(train["FTHG"]).to_numpy(dtype=float),
        pd.to_numeric(train["FTAG"]).to_numpy(dtype=float),
        train_labels,
        cv_splits=cv_splits,
    )
    baseline_models = fit_baselines(train)
    validation = frame.loc[validation_mask].reset_index(drop=True)
    validation_labels = labels[validation_mask]
    raw_validation = _frame_probabilities(
        model_suite, baseline_models, validation
    )

    calibrators: dict[str, MulticlassPlattCalibrator] = {}
    use_calibrated: dict[str, bool] = {}
    calibration_rows: list[dict[str, str | float]] = []
    validation_calibration_tables: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        raw_probabilities = raw_validation[model_name]
        raw_metrics = _metric_triplet(validation_labels, raw_probabilities)
        calibrator = MulticlassPlattCalibrator().fit(
            raw_probabilities, validation_labels
        )
        platt_probabilities = calibrator.predict_proba(raw_probabilities)
        platt_metrics = _metric_triplet(validation_labels, platt_probabilities)
        applied = (
            platt_metrics["log_loss"] < raw_metrics["log_loss"]
            and platt_metrics["brier_score"] < raw_metrics["brier_score"]
        )
        calibrators[model_name] = calibrator
        use_calibrated[model_name] = applied
        calibration_rows.append(
            {
                "model": model_name,
                "applied": "yes" if applied else "no",
                "validation_raw_log_loss": raw_metrics["log_loss"],
                "validation_platt_log_loss": platt_metrics["log_loss"],
                "validation_raw_brier": raw_metrics["brier_score"],
                "validation_platt_brier": platt_metrics["brier_score"],
                "validation_raw_ece": expected_calibration_error(
                    validation_labels, raw_probabilities
                ),
                "validation_platt_ece": expected_calibration_error(
                    validation_labels, platt_probabilities
                ),
            }
        )
        validation_calibration_tables.extend(
            [
                _calibration_table_with_tags(
                    model_name, "raw", validation_labels, raw_probabilities
                ),
                _calibration_table_with_tags(
                    model_name,
                    "platt",
                    validation_labels,
                    platt_probabilities,
                ),
            ]
        )

    selected_model = min(
        MODEL_NAMES,
        key=lambda name: (
            _metric_triplet(
                validation_labels,
                calibrators[name].predict_proba(raw_validation[name])
                if use_calibrated[name]
                else raw_validation[name],
            )["log_loss"]
        ),
    )
    test = frame.loc[test_mask].reset_index(drop=True)
    test_labels = labels[test_mask]
    raw_test = _frame_probabilities(model_suite, baseline_models, test)

    test_calibration_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, str | float]] = []
    final_test_probabilities: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        raw_probabilities = raw_test[model_name]
        platt_probabilities = calibrators[model_name].predict_proba(raw_probabilities)
        final_probabilities = (
            platt_probabilities if use_calibrated[model_name] else raw_probabilities
        )
        final_test_probabilities[model_name] = final_probabilities
        raw_metrics = _metric_triplet(test_labels, raw_probabilities)
        final_metrics = _metric_triplet(test_labels, final_probabilities)
        test_rows.append(
            {
                "model": model_name,
                "applied": "yes" if use_calibrated[model_name] else "no",
                "test_raw_log_loss": raw_metrics["log_loss"],
                "test_final_log_loss": final_metrics["log_loss"],
                "test_raw_brier": raw_metrics["brier_score"],
                "test_final_brier": final_metrics["brier_score"],
                "test_raw_ece": expected_calibration_error(
                    test_labels, raw_probabilities
                ),
                "test_final_ece": expected_calibration_error(
                    test_labels, final_probabilities
                ),
            }
        )
        test_calibration_tables.extend(
            [
                _calibration_table_with_tags(
                    model_name, "raw", test_labels, raw_probabilities
                ),
                _calibration_table_with_tags(
                    model_name, "platt", test_labels, platt_probabilities
                ),
            ]
        )

    selected_validation_probabilities = (
        calibrators[selected_model].predict_proba(raw_validation[selected_model])
        if use_calibrated[selected_model]
        else raw_validation[selected_model]
    )
    selected_test_probabilities = final_test_probabilities[selected_model]

    validation_table = pd.concat(validation_calibration_tables, ignore_index=True)
    test_table = pd.concat(test_calibration_tables, ignore_index=True)
    validation_csv = reports_dir / "calibration_analysis_validation.csv"
    test_csv = reports_dir / "calibration_analysis_test.csv"
    validation_plot = reports_dir / "calibration_analysis_validation.png"
    test_plot = reports_dir / "calibration_analysis_test.png"
    validation_table.to_csv(validation_csv, index=False)
    test_table.to_csv(test_csv, index=False)
    selected_validation_raw = _calibration_table_with_tags(
        selected_model, "raw", validation_labels, raw_validation[selected_model]
    )
    selected_validation_final = _calibration_table_with_tags(
        selected_model,
        "final",
        validation_labels,
        selected_validation_probabilities,
    )
    selected_test_raw = _calibration_table_with_tags(
        selected_model, "raw", test_labels, raw_test[selected_model]
    )
    selected_test_final = _calibration_table_with_tags(
        selected_model, "final", test_labels, selected_test_probabilities
    )
    save_calibration_comparison_plot(
        selected_validation_raw,
        selected_validation_final,
        validation_plot,
        title=f"{selected_model}: validation reliability",
    )
    save_calibration_comparison_plot(
        selected_test_raw,
        selected_test_final,
        test_plot,
        title=f"{selected_model}: test reliability",
    )

    selected_test_metrics = _metric_triplet(
        test_labels, selected_test_probabilities
    )
    subgroup_rows = _subgroup_metric_rows(
        test, test_labels, selected_test_probabilities
    )
    subgroup_table = "\n".join(
        f"| {row['Group']} | {row['N']} | {row['accuracy']:.4f} | "
        f"{row['log_loss']:.4f} | {row['rps']:.4f} |"
        for row in subgroup_rows
    )
    calibration_selection_table = "\n".join(
        f"| {row['model']} | {row['applied']} | "
        f"{row['validation_raw_log_loss']:.4f} | "
        f"{row['validation_platt_log_loss']:.4f} | "
        f"{row['validation_raw_brier']:.4f} | "
        f"{row['validation_platt_brier']:.4f} | "
        f"{row['test_raw_log_loss']:.4f} | "
        f"{row['test_final_log_loss']:.4f} | "
        f"{row['test_raw_ece']:.4f} | {row['test_final_ece']:.4f} |"
        for row, test_row in zip(calibration_rows, test_rows)
        for row in [{**row, **test_row}]
    )

    true_probabilities = np.array(
        [
            selected_test_probabilities[index, OUTCOME_ORDER.index(label)]
            for index, label in enumerate(test_labels)
        ]
    )
    miss_indices = np.argsort(true_probabilities)[:10]
    worst_rows: list[str] = []
    for index in miss_indices:
        match = test.iloc[int(index)]
        worst_rows.append(
            f"| {match['Date'].date()} | {match['HomeTeam']} | {match['AwayTeam']} | "
            f"{int(match['FTHG'])}-{int(match['FTAG'])} | {test_labels[index]} | "
            f"{true_probabilities[index]:.4f} |"
        )
    worst_table = "\n".join(worst_rows)
    worst_outcomes = pd.Series(test_labels[miss_indices]).value_counts()
    worst_outcome_summary = ", ".join(
        f"{label}: {count}"
        for label, count in worst_outcomes.items()
    )
    test_log_loss_worse = sum(
        row["applied"] == "yes"
        and row["test_final_log_loss"] > row["test_raw_log_loss"]
        for row in test_rows
    )
    test_brier_worse = sum(
        row["applied"] == "yes"
        and row["test_final_brier"] > row["test_raw_brier"]
        for row in test_rows
    )

    importance, coefficients = _feature_interpretation(model_suite)
    importance_rows = "\n".join(
        f"| {row.feature} | {row.importance:.4f} |"
        for row in importance.head(10).itertuples(index=False)
    )
    coefficient_rows = "\n".join(
        f"| {row.feature} | {row.home_goal_coefficient:.4f} | "
        f"{row.away_goal_coefficient:.4f} | {row.mean_absolute_coefficient:.4f} |"
        for row in coefficients.head(10).itertuples(index=False)
    )
    selected_test_count = len(test)
    report = f"""# Calibration and model analysis

## Evaluation protocol

- Train seasons: {", ".join(TRAIN_SEASONS)}.
- Calibration/selection season: {", ".join(VALIDATION_SEASONS)}.
- Held-out test seasons: {", ".join(TEST_SEASONS)} ({selected_test_count} matches).
- All model parameters were tuned by chronological cross-validation on training seasons. Platt calibrators were fit only on validation predictions; test labels were not used for calibration or model selection.
- A Platt calibrator is applied only when its validation log loss and Brier score are both lower than the uncalibrated values. Validation calibration metrics are in-sample and should not be read as unbiased estimates.
- The final error analysis model was selected by validation log loss after the calibration rule: **{selected_model}**.

## Calibration results

| Model | Platt applied | Validation raw log loss | Validation Platt log loss | Validation raw Brier | Validation Platt Brier | Test raw log loss | Test final log loss | Test raw ECE | Test final ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{calibration_selection_table}

Selected model's test metrics after the validation-only calibration choice: accuracy {selected_test_metrics["accuracy"]:.4f}, log loss {selected_test_metrics["log_loss"]:.4f}, Brier score {selected_test_metrics["brier_score"]:.4f}, RPS {selected_test_metrics["rps"]:.4f}.

Reliability plots show each result class (home win, draw, away win) before and after the validation-fitted calibration:

- [Validation reliability plot]({validation_plot.name})
- [Test reliability plot]({test_plot.name})
- [Validation calibration bins]({validation_csv.name})
- [Test calibration bins]({test_csv.name})

Validation-selected Platt calibration worsened test log loss for {test_log_loss_worse} of {sum(row["applied"] == "yes" for row in test_rows)} models to which it was applied, and worsened test Brier score for {test_brier_worse}. This is why the test comparison is reported separately from the validation selection result.

## Test performance breakdown

Early season is defined as fewer than five prior matches for both sides (the largest of their two season counts is below five); promoted-team involvement is based on league entry relative to the preceding season; the big six are Arsenal, Chelsea, Liverpool, Man City, Man United, and Tottenham.

| Group | N | Accuracy | Log loss | RPS |
|---|---:|---:|---:|---:|
{subgroup_table}

## Worst test misses

Matches are ranked by the final selected model's probability assigned to the true W/D/L result; lower true-outcome probability is a worse miss.
Among the ten lowest-probability true outcomes, the observed result counts were {worst_outcome_summary}. This is a descriptive pattern in this test period, not a general claim about away outcomes.

| Date | Home | Away | Score | Actual result | Probability of actual result |
|---|---|---|---:|---|---:|
{worst_table}

## Feature interpretation

Mean XGBoost feature importance averaged across home-goal, away-goal, and direct-result models:

| Feature | Average importance |
|---|---:|
{importance_rows}

Largest standardized Poisson coefficients (coefficients describe association with the modeled log expected goals):

| Feature | Home-goal coefficient | Away-goal coefficient | Mean absolute coefficient |
|---|---:|---:|---:|
{coefficient_rows}

Feature importance and regression coefficients are descriptive, not causal. Rolling form, team strength, and venue effects may be correlated.

## Limitations

- Football outcomes are high variance; a single season can differ materially from long-run performance.
- The Poisson scoreline grid assumes conditionally independent home and away goals and truncates each side at ten goals before renormalizing.
- Bookmaker odds are an external benchmark, not an input feature for these first-version models.
- Calibration is fit on one validation season; shifts in league scoring or team strength can make it stale.
- Promotion labels are inferred from participating team names in adjacent seasons. Injuries, lineups, tactical changes, and other match-day information are not modeled.
- Test metrics describe the supplied 2324-2526 files only and should not be generalized beyond that period.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved model analysis to {report_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Load engineered matches and run calibration and error analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--cv-splits", type=int, default=3)
    args = parser.parse_args(argv)
    matches = pd.read_parquet(args.input)
    run_model_analysis(
        matches, report_path=args.output, cv_splits=args.cv_splits
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
