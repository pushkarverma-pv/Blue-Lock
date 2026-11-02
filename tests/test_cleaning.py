import pandas as pd
import pytest

from src.cleaning import clean_matches, normalize_team_name


def _matches() -> pd.DataFrame:
    """Build minimal match rows with mixed date formats."""
    return pd.DataFrame(
        {
            "Date": ["05/08/2022", "2022-08-06"],
            "HomeTeam": ["  Man Utd  ", "Arsenal"],
            "AwayTeam": ["Spurs", "Liverpool"],
            "FTHG": [2, None],
            "FTAG": [1, 0],
            "FTR": ["H", "H"],
            "season": ["2223", "2223"],
            "league": ["E0", "E0"],
        }
    )


def test_clean_matches_normalizes_types_and_reports_dropped_goal_rows() -> None:
    cleaned, report = clean_matches(_matches())

    assert len(cleaned) == 1
    assert cleaned.loc[0, "HomeTeam"] == "Man United"
    assert cleaned.loc[0, "AwayTeam"] == "Tottenham"
    assert cleaned.loc[0, "FTHG"] == 2
    assert cleaned.loc[0, "Date"] == pd.Timestamp("2022-08-05")
    assert str(cleaned["season"].dtype) == "string"
    assert report.input_rows == 2
    assert report.output_rows == 1
    assert report.missing_goal_rows_dropped == 1
    assert report.team_names_normalized == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("  Manchester City  ", "Man City"),
        ("Wolverhampton Wanderers", "Wolves"),
        ("Arsenal", "Arsenal"),
    ],
)
def test_normalize_team_name(name: str, expected: str) -> None:
    assert normalize_team_name(name) == expected


def test_clean_matches_rejects_invalid_dates() -> None:
    frame = _matches()
    frame.loc[0, "Date"] = "not-a-date"

    with pytest.raises(ValueError, match="invalid dates"):
        clean_matches(frame)


def test_clean_matches_preserves_iso_date_month_and_day() -> None:
    frame = _matches()
    frame.loc[1, "FTHG"] = 1

    cleaned, _ = clean_matches(frame)

    assert cleaned["Date"].tolist() == [
        pd.Timestamp("2022-08-05"),
        pd.Timestamp("2022-08-06"),
    ]


def test_clean_matches_rejects_negative_or_fractional_goals() -> None:
    frame = _matches().iloc[:1].copy()
    frame.loc[0, "FTHG"] = -1

    with pytest.raises(ValueError, match="non-negative whole-number goals"):
        clean_matches(frame)
