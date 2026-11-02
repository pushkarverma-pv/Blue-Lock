from pathlib import Path

import pandas as pd
import pytest

from src.config import COLLECTION_SEASONS
from src.data_loader import combine_season_files, collect_data


def _write_source(path: Path, date: str, home_team: str) -> None:
    """Write a minimal football-data CSV fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Date,HomeTeam,AwayTeam,FTHG,FTAG\n{date},{home_team},Away,1,0\n",
        encoding="utf-8-sig",
    )


def test_combine_season_files_adds_season_and_sorts_dates(tmp_path: Path) -> None:
    """Combined matches are chronological with season metadata."""
    _write_source(tmp_path / "E0_2223.csv", "05/08/2022", "Later")
    _write_source(tmp_path / "E0_2122.csv", "15/08/2021", "Earlier")

    matches = combine_season_files(
        leagues=("E0",), seasons=("2223", "2122"), raw_dir=tmp_path
    )

    assert matches["HomeTeam"].tolist() == ["Earlier", "Later"]
    assert matches["season"].tolist() == ["2122", "2223"]
    assert matches["Date"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(matches["Date"])


def test_collect_data_saves_combined_csv_and_reports(tmp_path: Path, capsys) -> None:
    """Collection uses existing raw files and emits the summary."""
    raw_dir = tmp_path / "raw"
    _write_source(raw_dir / "E0_2324.csv", "11/08/2023", "Home")
    output_path = tmp_path / "processed" / "matches.csv"

    matches = collect_data(
        leagues=("E0",),
        seasons=("2324",),
        raw_dir=raw_dir,
        output_path=output_path,
    )

    saved = pd.read_csv(output_path, parse_dates=["Date"])
    report = capsys.readouterr().out
    assert len(matches) == len(saved) == 1
    assert saved.loc[0, "HomeTeam"] == "Home"
    assert "Rows per league-season:" in report
    assert "Date range:" in report
    assert "Missing values by column:" in report


def test_combine_reports_all_missing_raw_files(tmp_path: Path) -> None:
    """Missing source files are explicit and include their expected paths."""
    with pytest.raises(FileNotFoundError, match="E0_2324.csv"):
        combine_season_files(
            leagues=("E0",), seasons=("2324",), raw_dir=tmp_path
        )


def test_combine_season_files_preserves_iso_calendar_dates(tmp_path: Path) -> None:
    _write_source(tmp_path / "E0_2223.csv", "2022-08-05", "Home")

    matches = combine_season_files(
        leagues=("E0",), seasons=("2223",), raw_dir=tmp_path
    )

    assert matches.loc[0, "Date"] == pd.Timestamp("2022-08-05")


def test_default_collection_includes_live_2026_27_season() -> None:
    assert COLLECTION_SEASONS[-1] == "2627"
