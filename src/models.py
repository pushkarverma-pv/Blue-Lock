"""Fit and evaluate Poisson and XGBoost football prediction models."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.baselines import (
    BASELINE_NAMES,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    fit_baselines,
    outcome_labels,
    predict_baselines,
)
from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, REPORTS_DIR
from src.evaluate import OUTCOME_ORDER, evaluate_probabilities
from src.features import feature_columns

INPUT_PATH = PROCESSED_DATA_DIR / "features.parquet"
REPORT_PATH = REPORTS_DIR / "model_comparison.md"
MAX_GOALS = 10
CV_SPLITS = 3

POISSON_ALPHA_GRID = (0.01, 0.1, 1.0)
XGBOOST_GOAL_GRID = (
    {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05},
    {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05},
)
XGBOOST_CLASSIFIER_GRID = (
    {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05},
    {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05},
)


def scoreline_probability_grid(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Return a normalized independent-Poisson scoreline grid for 0..max_goals."""
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    expected = np.asarray([home_expected_goals, away_expected_goals], dtype=float)
    if not np.isfinite(expected).all() or (expected < 0).any():
        raise ValueError("expected goals must be finite and non-negative")
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, expected[0])
    away = poisson.pmf(goals, expected[1])
    grid = np.outer(home, away)
    total = float(grid.sum())
    if total <= 0:
        raise ValueError("scoreline probability grid has no probability mass")
    return grid / total


def scoreline_probability_grids(
    home_expected_goals: Sequence[float],
    away_expected_goals: Sequence[float],
    *,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Vectorized independent-Poisson scoreline grids for multiple matches."""
    home = np.asarray(home_expected_goals, dtype=float)
    away = np.asarray(away_expected_goals, dtype=float)
    if home.ndim != 1 or away.shape != home.shape:
        raise ValueError("expected-goal arrays must be matching one-dimensional arrays")
    if not np.isfinite(home).all() or not np.isfinite(away).all():
        raise ValueError("expected goals must be finite")
    if (home < 0).any() or (away < 0).any():
        raise ValueError("expected goals must be non-negative")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    goals = np.arange(max_goals + 1)
    home_probabilities = poisson.pmf(goals[None, :], home[:, None])
    away_probabilities = poisson.pmf(goals[None, :], away[:, None])
    grids = home_probabilities[:, :, None] * away_probabilities[:, None, :]
    totals = grids.sum(axis=(1, 2), keepdims=True)
    if (totals <= 0).any():
        raise ValueError("scoreline probability grid has no probability mass")
    return grids / totals


def outcome_probabilities_from_grid(grid: np.ndarray) -> np.ndarray:
    """Aggregate a home-goals by away-goals grid to H/D/A probabilities."""
    probabilities = np.asarray(grid, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != probabilities.shape[1]:
        raise ValueError("grid must be a square two-dimensional scoreline matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("grid entries must be finite and non-negative")
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("grid must have positive probability mass")
    probabilities = probabilities / total
    home = np.tril(probabilities, k=-1).sum()
    draw = np.trace(probabilities)
    away = np.triu(probabilities, k=1).sum()
    return np.array([home, draw, away], dtype=float)


def outcome_probabilities_from_grids(grids: np.ndarray) -> np.ndarray:
    """Aggregate scoreline grids for a batch of matches into H/D/A probabilities."""
    values = np.asarray(grids, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("grids must have shape (n_matches, n_goals, n_goals)")
    return np.vstack([outcome_probabilities_from_grid(grid) for grid in values])


def _poisson_estimator(alpha: float) -> Any:
    """Create a scaled Poisson regression pipeline."""
    return make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=alpha, max_iter=500, tol=1e-7),
    )


def _xgboost_regressor(parameters: dict[str, int | float]) -> XGBRegressor:
    """Create a reproducible XGBoost count-Poisson regressor."""
    return XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **parameters,
    )


def _xgboost_classifier(parameters: dict[str, int | float]) -> XGBClassifier:
    """Create a reproducible three-class XGBoost outcome classifier."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(OUTCOME_ORDER),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **parameters,
    )


def _select_goal_parameters(
    estimator_factory: Callable[[dict[str, int | float]], Any],
    candidates: Sequence[dict[str, int | float]],
    features: pd.DataFrame,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    splitter: TimeSeriesSplit,
) -> tuple[dict[str, int | float], float]:
    """Select paired home/away goal-model parameters by chronological CV MAE."""
    best_parameters: dict[str, int | float] | None = None
    best_score = float("inf")
    for candidate in candidates:
        fold_scores: list[float] = []
        for train_indices, validation_indices in splitter.split(features):
            fold_errors: list[float] = []
            for target in (home_goals, away_goals):
                estimator = estimator_factory(candidate)
                estimator.fit(features.iloc[train_indices], target[train_indices])
                prediction = np.clip(
                    estimator.predict(features.iloc[validation_indices]), 1e-6, None
                )
                fold_errors.extend(
                    np.abs(prediction - target[validation_indices]).tolist()
                )
            fold_scores.append(float(np.mean(fold_errors)))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_parameters = dict(candidate)
    if best_parameters is None:
        raise ValueError("No model hyperparameter candidates were evaluated")
    return best_parameters, best_score


def _select_classifier_parameters(
    features: pd.DataFrame,
    outcomes: np.ndarray,
    candidates: Sequence[dict[str, int | float]],
    splitter: TimeSeriesSplit,
) -> tuple[dict[str, int | float], float]:
    """Select direct W/D/L XGBoost parameters by chronological CV log loss."""
    best_parameters: dict[str, int | float] | None = None
    best_score = float("inf")
    numeric_outcomes = np.array([OUTCOME_ORDER.index(str(label)) for label in outcomes])
    for candidate in candidates:
        fold_scores: list[float] = []
        for train_indices, validation_indices in splitter.split(features):
            estimator = _xgboost_classifier(candidate)
            estimator.fit(
                features.iloc[train_indices], numeric_outcomes[train_indices]
            )
            raw_probabilities = estimator.predict_proba(
                features.iloc[validation_indices]
            )
            aligned = np.zeros((len(validation_indices), len(OUTCOME_ORDER)))
            for index, label in enumerate(estimator.classes_):
                aligned[:, int(label)] = raw_probabilities[:, index]
            metrics = evaluate_probabilities(
                outcomes[validation_indices], aligned
            )
            fold_scores.append(metrics["log_loss"])
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_parameters = dict(candidate)
    if best_parameters is None:
        raise ValueError("No classifier hyperparameter candidates were evaluated")
    return best_parameters, best_score


def _fit_goal_pair(
    factory: Callable[[dict[str, int | float]], Any],
    parameters: dict[str, int | float],
    features: pd.DataFrame,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> tuple[Any, Any]:
    """Fit separate home-goal and away-goal estimators."""
    home_model = factory(parameters)
    away_model = factory(parameters)
    home_model.fit(features, home_goals)
    away_model.fit(features, away_goals)
    return home_model, away_model


@dataclass(frozen=True)
class ModelSuite:
    """Fitted estimators, selected parameters, and training-only CV scores."""

    poisson_home: Any
    poisson_away: Any
    xgb_home: Any
    xgb_away: Any
    xgb_classifier: XGBClassifier
    poisson_parameters: dict[str, int | float]
    xgb_goal_parameters: dict[str, int | float]
    xgb_classifier_parameters: dict[str, int | float]
    poisson_cv_mae: float
    xgb_goal_cv_mae: float
    xgb_classifier_cv_log_loss: float
    feature_names: tuple[str, ...]


def fit_model_suite(
    train_features: pd.DataFrame,
    home_goals: Sequence[float],
    away_goals: Sequence[float],
    outcomes: Sequence[str],
    *,
    cv_splits: int = CV_SPLITS,
) -> ModelSuite:
    """Tune and fit all models using training data and chronological CV only."""
    features = _feature_matrix(train_features)
    home_targets = np.asarray(home_goals, dtype=float)
    away_targets = np.asarray(away_goals, dtype=float)
    labels = np.asarray(outcomes, dtype=str)
    if len(features) == 0 or len(home_targets) != len(features):
        raise ValueError("training features and goal targets must have matching non-empty lengths")
    if len(away_targets) != len(features) or len(labels) != len(features):
        raise ValueError("training features, goals, and outcomes must have matching lengths")
    if not np.isfinite(home_targets).all() or not np.isfinite(away_targets).all():
        raise ValueError("training goal targets must be finite")
    if not set(labels).issubset(OUTCOME_ORDER):
        raise ValueError("training outcomes must be H, D, or A")
    if cv_splits < 2 or len(features) <= cv_splits:
        raise ValueError("not enough training matches for the requested time-series CV")

    splitter = TimeSeriesSplit(n_splits=cv_splits)
    poisson_parameters, poisson_cv_mae = _select_goal_parameters(
        lambda candidate: _poisson_estimator(float(candidate["alpha"])),
        [{"alpha": alpha} for alpha in POISSON_ALPHA_GRID],
        features,
        home_targets,
        away_targets,
        splitter,
    )
    xgb_goal_parameters, xgb_goal_cv_mae = _select_goal_parameters(
        _xgboost_regressor,
        XGBOOST_GOAL_GRID,
        features,
        home_targets,
        away_targets,
        splitter,
    )
    xgb_classifier_parameters, xgb_classifier_cv_log_loss = (
        _select_classifier_parameters(
            features,
            labels,
            XGBOOST_CLASSIFIER_GRID,
            splitter,
        )
    )

    poisson_home, poisson_away = _fit_goal_pair(
        lambda candidate: _poisson_estimator(float(candidate["alpha"])),
        poisson_parameters,
        features,
        home_targets,
        away_targets,
    )
    xgb_home, xgb_away = _fit_goal_pair(
        _xgboost_regressor,
        xgb_goal_parameters,
        features,
        home_targets,
        away_targets,
    )
    numeric_outcomes = np.array([OUTCOME_ORDER.index(label) for label in labels])
    direct_classifier = _xgboost_classifier(xgb_classifier_parameters)
    direct_classifier.fit(features, numeric_outcomes)
    return ModelSuite(
        poisson_home=poisson_home,
        poisson_away=poisson_away,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        xgb_classifier=direct_classifier,
        poisson_parameters=poisson_parameters,
        xgb_goal_parameters=xgb_goal_parameters,
        xgb_classifier_parameters=xgb_classifier_parameters,
        poisson_cv_mae=poisson_cv_mae,
        xgb_goal_cv_mae=xgb_goal_cv_mae,
        xgb_classifier_cv_log_loss=xgb_classifier_cv_log_loss,
        feature_names=tuple(feature_columns()),
    )


def predict_model_suite(
    model_suite: ModelSuite,
    matches: pd.DataFrame,
    *,
    max_goals: int = MAX_GOALS,
) -> dict[str, dict[str, np.ndarray]]:
    """Predict W/D/L probabilities and expected goals from a fitted model suite."""
    features = _feature_matrix(matches)
    poisson_home = np.clip(model_suite.poisson_home.predict(features), 1e-6, None)
    poisson_away = np.clip(model_suite.poisson_away.predict(features), 1e-6, None)
    xgb_home = np.clip(model_suite.xgb_home.predict(features), 1e-6, None)
    xgb_away = np.clip(model_suite.xgb_away.predict(features), 1e-6, None)
    direct_raw = model_suite.xgb_classifier.predict_proba(features)
    direct = np.zeros((len(features), len(OUTCOME_ORDER)))
    for class_index, label in enumerate(model_suite.xgb_classifier.classes_):
        direct[:, int(label)] = direct_raw[:, class_index]
    return {
        "Poisson regression": {
            "probabilities": outcome_probabilities_from_grids(
                scoreline_probability_grids(
                    poisson_home, poisson_away, max_goals=max_goals
                )
            ),
            "home_expected_goals": poisson_home,
            "away_expected_goals": poisson_away,
        },
        "XGBoost Poisson goals": {
            "probabilities": outcome_probabilities_from_grids(
                scoreline_probability_grids(
                    xgb_home, xgb_away, max_goals=max_goals
                )
            ),
            "home_expected_goals": xgb_home,
            "away_expected_goals": xgb_away,
        },
        "XGBoost direct W/D/L": {"probabilities": direct},
    }


def _goal_model_metrics(
    labels: np.ndarray,
    home_actual: np.ndarray,
    away_actual: np.ndarray,
    home_prediction: np.ndarray,
    away_prediction: np.ndarray,
    max_goals: int = MAX_GOALS,
) -> dict[str, float]:
    """Calculate outcome metrics and both goal MAEs for a goal model."""
    grids = scoreline_probability_grids(
        home_prediction, away_prediction, max_goals=max_goals
    )
    outcomes = outcome_probabilities_from_grids(grids)
    actual_goals = np.column_stack([home_actual, away_actual])
    predicted_goals = np.column_stack([home_prediction, away_prediction])
    return evaluate_probabilities(
        labels,
        outcomes,
        actual_goals=actual_goals,
        predicted_goals=predicted_goals,
    )


def _prediction_row(
    model: str,
    split: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    home_actual: np.ndarray | None = None,
    away_actual: np.ndarray | None = None,
    home_prediction: np.ndarray | None = None,
    away_prediction: np.ndarray | None = None,
    max_goals: int = MAX_GOALS,
) -> dict[str, str | int | float]:
    """Create one metrics-table row for a model and data split."""
    if home_actual is not None:
        metrics = _goal_model_metrics(
            labels,
            home_actual,
            away_actual,
            home_prediction,
            away_prediction,
            max_goals,
        )
    else:
        metrics = evaluate_probabilities(labels, probabilities)
    return {"Model": model, "Split": split, "N": len(labels), **metrics}


def _feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the numeric features used by all prediction models."""
    columns = feature_columns()
    missing = [column for column in columns if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing generated model features: {', '.join(missing)}")
    features = matches.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain missing or non-finite values")
    return features.astype(float)


def run_model_comparison(
    matches: pd.DataFrame,
    *,
    report_path: Path = REPORT_PATH,
    cv_splits: int = CV_SPLITS,
    max_goals: int = MAX_GOALS,
) -> pd.DataFrame:
    """Tune on training seasons, fit models, and report validation/test results."""
    required = {"season", "Date", "FTHG", "FTAG"}
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Missing columns required for model comparison: {', '.join(missing)}")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    frame = matches.copy()
    frame["season"] = frame["season"].astype("string")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("Model data contains invalid dates")
    frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    features = _feature_matrix(frame)
    home_goals = pd.to_numeric(frame["FTHG"], errors="coerce").to_numpy(dtype=float)
    away_goals = pd.to_numeric(frame["FTAG"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(home_goals).all() or not np.isfinite(away_goals).all():
        raise ValueError("Model data contains missing goal targets")
    labels = outcome_labels(frame)

    train_mask = frame["season"].isin(TRAIN_SEASONS).to_numpy()
    train_indices = np.flatnonzero(train_mask)
    if len(train_indices) <= cv_splits:
        raise ValueError("Not enough training matches for the requested time-series CV")
    train_features = features.iloc[train_indices].reset_index(drop=True)
    train_home_goals = home_goals[train_indices]
    train_away_goals = away_goals[train_indices]
    train_labels = labels[train_indices]
    model_suite = fit_model_suite(
        train_features,
        train_home_goals,
        train_away_goals,
        train_labels,
        cv_splits=cv_splits,
    )

    baseline_models = fit_baselines(frame.loc[train_mask])
    # Features parquet already includes pre-match Elo; baseline mapping is fit
    # using only the training-season rows.
    records: list[dict[str, str | int | float]] = []
    for split_name, seasons in (
        ("Validation", VALIDATION_SEASONS),
        ("Test", TEST_SEASONS),
    ):
        split_mask = frame["season"].isin(seasons).to_numpy()
        split_indices = np.flatnonzero(split_mask)
        if len(split_indices) == 0:
            raise ValueError(f"No data found for {split_name.lower()} seasons {seasons}")
        split_labels = labels[split_indices]
        split_home = home_goals[split_indices]
        split_away = away_goals[split_indices]
        split_matches = frame.iloc[split_indices]
        baseline_predictions = predict_baselines(split_matches, baseline_models)
        for baseline in BASELINE_NAMES:
            probabilities = baseline_predictions[baseline]
            if baseline == "Bookmaker":
                valid = np.isfinite(probabilities).all(axis=1)
                if valid.any():
                    records.append(
                        _prediction_row(
                            baseline,
                            split_name,
                            split_labels[valid],
                            probabilities[valid],
                        )
                    )
                continue
            records.append(
                _prediction_row(
                    baseline,
                    split_name,
                    split_labels,
                    probabilities,
                )
            )

        fitted_predictions = predict_model_suite(
            model_suite, split_matches, max_goals=max_goals
        )
        poisson_home_pred = fitted_predictions["Poisson regression"][
            "home_expected_goals"
        ]
        poisson_away_pred = fitted_predictions["Poisson regression"][
            "away_expected_goals"
        ]
        records.append(
            _prediction_row(
                "Poisson regression",
                split_name,
                split_labels,
                fitted_predictions["Poisson regression"]["probabilities"],
                home_actual=split_home,
                away_actual=split_away,
                home_prediction=poisson_home_pred,
                away_prediction=poisson_away_pred,
                max_goals=max_goals,
            )
        )

        xgb_home_pred = fitted_predictions["XGBoost Poisson goals"][
            "home_expected_goals"
        ]
        xgb_away_pred = fitted_predictions["XGBoost Poisson goals"][
            "away_expected_goals"
        ]
        records.append(
            _prediction_row(
                "XGBoost Poisson goals",
                split_name,
                split_labels,
                fitted_predictions["XGBoost Poisson goals"]["probabilities"],
                home_actual=split_home,
                away_actual=split_away,
                home_prediction=xgb_home_pred,
                away_prediction=xgb_away_pred,
                max_goals=max_goals,
            )
        )

        records.append(
            _prediction_row(
                "XGBoost direct W/D/L",
                split_name,
                split_labels,
                fitted_predictions["XGBoost direct W/D/L"]["probabilities"],
            )
        )

    results = pd.DataFrame.from_records(records)
    metric_columns = (
        "accuracy",
        "log_loss",
        "brier_score",
        "rps",
        "home_goals_mae",
        "away_goals_mae",
    )
    for column in metric_columns:
        if column not in results:
            results[column] = np.nan
    results = results[
        ["Model", "Split", "N", *metric_columns]
    ]

    lines = [
        "# Model comparison",
        "",
        f"- Training seasons: {', '.join(TRAIN_SEASONS)}",
        f"- Validation season: {', '.join(VALIDATION_SEASONS)}",
        f"- Test seasons: {', '.join(TEST_SEASONS)}",
        f"- Features: {len(feature_columns())} generated pre-match features; bookmaker odds are excluded from model inputs.",
        f"- Hyperparameters selected by {cv_splits}-split expanding-window `TimeSeriesSplit` over training seasons only; test seasons were held out until final reporting.",
        f"- Scoreline grids use independent Poisson distributions truncated and renormalized to 0-{max_goals} goals per side.",
        "- Outcome order for probabilities is home win, draw, away win. RPS uses the ordinal order away win, draw, home win.",
        "",
        "## Chronological cross-validation selection",
        "",
        "| Model | Selected parameters | CV objective |",
        "|---|---|---:|",
        f"| Poisson regression | `{model_suite.poisson_parameters}` | Mean home/away goals MAE {model_suite.poisson_cv_mae:.4f} |",
        f"| XGBoost Poisson goals | `{model_suite.xgb_goal_parameters}` | Mean home/away goals MAE {model_suite.xgb_goal_cv_mae:.4f} |",
        f"| XGBoost direct W/D/L | `{model_suite.xgb_classifier_parameters}` | Multiclass log loss {model_suite.xgb_classifier_cv_log_loss:.4f} |",
        "",
        "## Results",
        "",
        "| Model | Split | N | Accuracy | Log loss | Brier score | RPS | Home goals MAE | Away goals MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        rendered = [
            "—" if pd.isna(getattr(row, column)) else f"{getattr(row, column):.4f}"
            for column in metric_columns
        ]
        lines.append(
            f"| {row.Model} | {row.Split} | {row.N} | "
            + " | ".join(rendered)
            + " |"
        )
    lines.extend(
        [
            "",
            "The goal MAEs are omitted for baselines without goal estimates and the direct W/D/L classifier.",
            "Bookmaker scores are computed only for matches with valid bookmaker odds; their `N` is shown separately.",
            "Model estimates are experimental baselines, not guarantees. Results should be interpreted with the high variance of football outcomes in mind.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved model comparison to {report_path}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Load engineered features and run model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--cv-splits", type=int, default=CV_SPLITS)
    args = parser.parse_args(argv)
    matches = pd.read_parquet(args.input)
    run_model_comparison(
        matches,
        report_path=args.output,
        cv_splits=args.cv_splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
