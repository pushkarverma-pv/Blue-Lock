import json

import pandas as pd
import pytest

from src.features import build_features
from src.config import COLLECTION_SEASONS, LIVE_SEASONS, SEASONS
from src.train import (
    _update_processed_matches,
    read_pre_refit_evaluation,
    train_production_model,
)


_COMPARISON_REPORT = """# Model comparison

| Model | Selected parameters | CV objective |
|---|---|---:|
| Poisson regression | `{'alpha': 1.0}` | Mean home/away goals MAE 0.9230 |

| Model | Split | N | Accuracy | Log loss | Brier score | RPS | Home goals MAE | Away goals MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Elo logistic | Test | 1140 | 0.5289 | 0.9827 | 0.5865 | 0.2013 | — | — |
| Bookmaker | Test | 1140 | 0.5421 | 0.9650 | 0.5740 | 0.1955 | — | — |
| Poisson regression | Test | 1140 | 0.5404 | 0.9834 | 0.5848 | 0.2006 | 0.9821 | 0.8794 |
"""


def _training_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-08",
                    "2023-08-15",
                    "2023-08-22",
                    "2023-08-29",
                    "2023-09-05",
                    "2023-09-12",
                    "2023-09-19",
                ]
            ),
            "HomeTeam": ["A", "B", "A", "C", "A", "B", "C", "A"],
            "AwayTeam": ["B", "A", "C", "A", "B", "C", "A", "C"],
            "FTHG": [2, 1, 0, 3, 1, 2, 0, 1],
            "FTAG": [0, 1, 1, 0, 2, 1, 2, 1],
            "HS": [12, 10, 8, 14, 11, 12, 8, 10],
            "AS": [4, 9, 8, 5, 10, 8, 12, 11],
            "HST": [5, 4, 2, 7, 4, 5, 2, 4],
            "AST": [2, 3, 3, 2, 4, 3, 4, 3],
            "season": ["2526"] * 8,
            "league": ["E0"] * 8,
        }
    )


def test_read_pre_refit_evaluation_preserves_recorded_test_metrics(tmp_path) -> None:
    report_path = tmp_path / "model_comparison.md"
    report_path.write_text(_COMPARISON_REPORT, encoding="utf-8")

    evaluation = read_pre_refit_evaluation(report_path)
    test_metrics = {
        row["model"]: row
        for row in evaluation["reported_metrics"]
        if row["split"] == "Test"
    }

    assert evaluation["cv_selected_parameters"] == {"alpha": 1.0}
    assert evaluation["cv_goal_mae"] == pytest.approx(0.9230)
    assert test_metrics["Poisson regression"]["log_loss"] == pytest.approx(0.9834)
    assert test_metrics["Elo logistic"]["log_loss"] == pytest.approx(0.9827)
    assert test_metrics["Bookmaker"]["log_loss"] == pytest.approx(0.9650)


def test_train_refits_poisson_on_all_rows_and_saves_metadata(tmp_path) -> None:
    matches = _training_matches()
    features = build_features(matches)
    report_path = tmp_path / "model_comparison.md"
    report_path.write_text(_COMPARISON_REPORT, encoding="utf-8")
    model_path = tmp_path / "models" / "poisson.joblib"
    metadata_path = tmp_path / "models" / "metadata.json"

    metadata = train_production_model(
        features,
        league="E0",
        model_path=model_path,
        metadata_path=metadata_path,
        evaluation_report_path=report_path,
    )

    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert model_path.is_file()
    assert metadata["training_rows"] == len(features)
    assert saved["training_seasons"] == ["2526"]
    assert saved["hyperparameters"] == {"alpha": 1.0}
    assert "pre_refit_evaluation" in saved
    test_metrics = saved["pre_refit_evaluation"]["reported_metrics"]
    assert any(
        row["model"] == "Poisson regression"
        and row["split"] == "Test"
        and row["log_loss"] == pytest.approx(0.9834)
        for row in test_metrics
    )


def test_update_appends_new_matches_and_deduplicates_without_editing_raw(
    tmp_path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    original_raw: dict[str, bytes] = {}
    existing_rows = []
    for index, season in enumerate(SEASONS):
        start_year = 2000 + 15 + index
        date = f"{start_year}-08-01"
        row = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
            f"{date},Team {index},Opponent {index},1,0\n"
        )
        raw_path = raw_dir / f"E0_{season}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(row, encoding="utf-8")
        original_raw[season] = raw_path.read_bytes()
        if index == 0:
            existing_rows.append(
                {
                    "Date": date,
                    "HomeTeam": "Team 0",
                    "AwayTeam": "Opponent 0",
                    "FTHG": 1,
                    "FTAG": 0,
                    "season": season,
                    "league": "E0",
                }
            )
            existing_rows.append(
                {
                    "Date": f"{start_year}-08-08",
                    "HomeTeam": "Extra",
                    "AwayTeam": "Opponent",
                    "FTHG": 0,
                    "FTAG": 1,
                    "season": season,
                    "league": "E0",
                }
            )
    live_path = raw_dir / "E0_2627.csv"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
        "21/08/2026,Promoted,Opponent,1,0\n",
        encoding="utf-8",
    )
    original_raw["2627"] = live_path.read_bytes()
    matches_path = tmp_path / "processed" / "matches.csv"
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(existing_rows).to_csv(matches_path, index=False)
    download_calls = []
    monkeypatch.setattr(
        "src.train.download_season_files",
        lambda **kwargs: download_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "src.train.FEATURE_SUMMARY_PATH",
        tmp_path / "reports" / "feature_summary.md",
    )

    updated = _update_processed_matches(
        "E0",
        raw_dir=raw_dir,
        matches_path=matches_path,
        cleaned_path=tmp_path / "processed" / "cleaned.csv",
        feature_path=tmp_path / "processed" / "features.parquet",
    )
    combined = pd.read_csv(matches_path)

    assert len(combined) == len(SEASONS) + 2
    assert len(updated) == len(SEASONS) + 2
    assert "2627" in set(combined["season"].astype(str))
    assert download_calls[0]["seasons"] == COLLECTION_SEASONS
    assert LIVE_SEASONS == ("2627",)
    assert combined.duplicated(
        subset=["league", "season", "Date", "HomeTeam", "AwayTeam"]
    ).sum() == 0
    assert all(
        (raw_dir / f"E0_{season}.csv").read_bytes() == contents
        for season, contents in original_raw.items()
    )
