"""Fit the production Poisson models and save deployment artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from src.cleaning import clean_matches
from src.config import (
    MODELS_DIR,
    COLLECTION_SEASONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEASONS,
)
from src.data_loader import (
    combine_season_files,
    download_season_files,
)
from src.features import (
    OUTPUT_PATH as FEATURE_OUTPUT_PATH,
    SUMMARY_PATH as FEATURE_SUMMARY_PATH,
    build_features,
    feature_columns,
    save_feature_summary,
)
from src.models import _poisson_estimator

MODEL_PATH = MODELS_DIR / "poisson_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.md"
MATCHES_PATH = PROCESSED_DATA_DIR / "matches.csv"
CLEANED_MATCHES_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"
PRODUCTION_MODEL = "Poisson regression"


def _parse_number(value: str) -> float | int | None:
    """Parse a model-report cell to a JSON-compatible numeric value."""
    if value.strip() in {"—", "-", ""}:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def read_pre_refit_evaluation(report_path: Path) -> dict[str, object]:
    """Read the recorded validation/test metrics and CV choice from the report."""
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing evaluation report {report_path}; run `python -m src.models` "
            "before fitting the deployment model."
        )
    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    cv_parameters: dict[str, object] | None = None
    cv_goal_mae: float | None = None
    evaluation_rows: list[dict[str, object]] = []
    parsing_results = False

    for line in lines:
        if line.startswith("| Model | Split | N |"):
            parsing_results = True
            continue
        if parsing_results:
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                if evaluation_rows:
                    parsing_results = False
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != 9 or cells[0] == "Model":
                continue
            evaluation_rows.append(
                {
                    "model": cells[0],
                    "split": cells[1],
                    "n": int(cells[2]),
                    "accuracy": _parse_number(cells[3]),
                    "log_loss": _parse_number(cells[4]),
                    "brier_score": _parse_number(cells[5]),
                    "rps": _parse_number(cells[6]),
                    "home_goals_mae": _parse_number(cells[7]),
                    "away_goals_mae": _parse_number(cells[8]),
                }
            )

        match = re.match(
            r"\| Poisson regression \| `(\{.*\})` \| Mean home/away goals MAE ([0-9.]+) \|",
            line,
        )
        if match:
            parsed_parameters = ast.literal_eval(match.group(1))
            if not isinstance(parsed_parameters, dict):
                raise ValueError("Poisson CV parameters in report are malformed")
            cv_parameters = parsed_parameters
            cv_goal_mae = float(match.group(2))

    if cv_parameters is None or cv_goal_mae is None:
        raise ValueError(f"Could not find Poisson CV selection in {report_path}")
    if not evaluation_rows:
        raise ValueError(f"Could not find pre-refit evaluation rows in {report_path}")
    return {
        "source_report": str(report_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv_selected_parameters": cv_parameters,
        "cv_goal_mae": cv_goal_mae,
        "reported_metrics": evaluation_rows,
        "interpretation": (
            "These are evaluation metrics from before deployment refitting. "
            "The deployment estimator is refit on all seasons with the CV-selected "
            "hyperparameters; the held-out metrics remain historical and are not "
            "estimates for the refitted artifact."
        ),
    }


def _available_raw_seasons(league: str, raw_dir: Path) -> list[str]:
    """Return configured and locally available season codes for a league."""
    discovered = {
        match.group(1)
        for path in raw_dir.glob(f"{league}_*.csv")
        if (match := re.fullmatch(rf"{re.escape(league)}_([0-9]{{4}})\.csv", path.name))
    }
    return sorted(set(SEASONS).union(discovered))


def _update_processed_matches(
    league: str,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    matches_path: Path = MATCHES_PATH,
    cleaned_path: Path = CLEANED_MATCHES_PATH,
    feature_path: Path = FEATURE_OUTPUT_PATH,
) -> pd.DataFrame:
    """Download configured missing sources, append new matches, and rebuild features.

    Raw source files remain read-only. Files for additional seasons already
    present in ``raw_dir`` are automatically included in the rebuild.
    """
    download_season_files(
        leagues=(league,), seasons=COLLECTION_SEASONS, raw_dir=raw_dir
    )
    seasons = _available_raw_seasons(league, raw_dir)
    incoming = combine_season_files((league,), seasons, raw_dir)
    if matches_path.is_file():
        existing = pd.read_csv(matches_path, low_memory=False)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    if combined["Date"].isna().any():
        raise ValueError("Update source contains invalid dates")
    combined["season"] = combined["season"].astype("string")
    combined["league"] = combined["league"].astype("string")
    for column in ("HomeTeam", "AwayTeam"):
        combined[column] = combined[column].astype("string").str.strip()
    keys = ["league", "season", "Date", "HomeTeam", "AwayTeam"]
    missing_keys = [column for column in keys if column not in combined]
    if missing_keys:
        raise ValueError(
            f"Cannot deduplicate collected matches; missing keys: {', '.join(missing_keys)}"
        )
    combined = (
        combined.drop_duplicates(subset=keys, keep="last")
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(matches_path, index=False, date_format="%Y-%m-%d")
    cleaned, _ = clean_matches(combined)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False, date_format="%Y-%m-%d")
    features = build_features(cleaned)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    save_feature_summary(features, FEATURE_SUMMARY_PATH)
    return features


def train_production_model(
    features: pd.DataFrame,
    *,
    league: str,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    evaluation_report_path: Path = MODEL_COMPARISON_PATH,
) -> dict[str, object]:
    """Refit both Poisson goal estimators on all supplied seasons.

    The Poisson regularization value is read from the already-recorded
    chronological-CV evaluation report. No model selection or test evaluation
    is performed in this deployment-refit step.
    """
    evaluation = read_pre_refit_evaluation(evaluation_report_path)
    parameters = evaluation["cv_selected_parameters"]
    if not isinstance(parameters, dict) or "alpha" not in parameters:
        raise ValueError("Recorded Poisson CV parameters must include alpha")
    alpha = float(parameters["alpha"])
    columns = feature_columns()
    missing = [column for column in columns if column not in features]
    if missing:
        raise ValueError(f"Missing generated training features: {', '.join(missing)}")
    required = {"season", "FTHG", "FTAG"}
    missing_targets = sorted(required.difference(features.columns))
    if missing_targets:
        raise ValueError(f"Missing training target columns: {', '.join(missing_targets)}")

    ordered = features.sort_values("Date", kind="stable") if "Date" in features else features
    X = ordered.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    home_goals = pd.to_numeric(ordered["FTHG"], errors="coerce").to_numpy(dtype=float)
    away_goals = pd.to_numeric(ordered["FTAG"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain missing or non-finite values")
    if not np.isfinite(home_goals).all() or not np.isfinite(away_goals).all():
        raise ValueError("Training targets contain missing or non-finite goals")
    if (home_goals < 0).any() or (away_goals < 0).any():
        raise ValueError("Training goals must be non-negative")
    if len(ordered) == 0:
        raise ValueError("Cannot fit deployment models on an empty dataset")

    home_model = _poisson_estimator(alpha)
    away_model = _poisson_estimator(alpha)
    home_model.fit(X, home_goals)
    away_model.fit(X, away_goals)
    training_seasons = sorted(ordered["season"].astype(str).unique())

    artifact = {
        "model_type": PRODUCTION_MODEL,
        "home_goals_model": home_model,
        "away_goals_model": away_model,
        "feature_names": columns,
        "league": league,
        "max_goals": 10,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metadata: dict[str, object] = {
        "model_type": PRODUCTION_MODEL,
        "league": league,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons": training_seasons,
        "training_rows": int(len(ordered)),
        "feature_names": columns,
        "hyperparameters": evaluation["cv_selected_parameters"],
        "training_procedure": (
            "After pre-refit evaluation, both Poisson goal regressors were fit on all "
            "available seasons with the regularization selected by chronological "
            "cross-validation on training seasons only. No test-set tuning was performed."
        ),
        "pre_refit_evaluation": evaluation,
        "model_artifact": str(model_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Saved {PRODUCTION_MODEL} deployment models for {len(ordered)} matches "
        f"({', '.join(training_seasons)}) to {model_path}"
    )
    print(f"Saved metadata to {metadata_path}")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """Train deployment artifacts from the feature table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="E0")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Append matches available from raw season CSVs, rebuild features, and refit.",
    )
    parser.add_argument("--features", type=Path, default=FEATURE_OUTPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=METADATA_PATH)
    parser.add_argument("--evaluation-report", type=Path, default=MODEL_COMPARISON_PATH)
    args = parser.parse_args(argv)
    if args.update:
        features = _update_processed_matches(args.league, feature_path=args.features)
    else:
        features = pd.read_parquet(args.features)
    train_production_model(
        features,
        league=args.league,
        model_path=args.model_output,
        metadata_path=args.metadata_output,
        evaluation_report_path=args.evaluation_report,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
