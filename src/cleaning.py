"""Clean collected football match data without changing raw source files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import PROCESSED_DATA_DIR

REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
GOAL_COLUMNS = ("FTHG", "FTAG")
TEAM_COLUMNS = ("HomeTeam", "AwayTeam")
DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "matches.csv"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "cleaned_matches.csv"

_TEAM_ALIASES = {
    "brighton & hove albion": "Brighton",
    "manchester city": "Man City",
    "manchester united": "Man United",
    "man utd": "Man United",
    "nottingham forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "west bromwich albion": "West Brom",
    "wolverhampton wanderers": "Wolves",
}


@dataclass(frozen=True)
class CleaningSummary:
    """Counts describing data removed or normalized during cleaning."""

    input_rows: int
    output_rows: int
    missing_goal_rows_dropped: int
    team_names_normalized: int


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates and day-first source dates without swapping components."""
    text = values.astype("string").str.strip()
    iso_dates = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:$|[ T])", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso_dates] = pd.to_datetime(
        text.loc[iso_dates], format="mixed", errors="coerce"
    )
    parsed.loc[~iso_dates] = pd.to_datetime(
        text.loc[~iso_dates], format="mixed", dayfirst=True, errors="coerce"
    )
    return parsed


def normalize_team_name(name: object) -> object:
    """Normalize whitespace and known team-name aliases, preserving missing values."""
    if pd.isna(name):
        return name
    cleaned = " ".join(str(name).strip().split())
    return _TEAM_ALIASES.get(cleaned.casefold(), cleaned)


def clean_matches(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Validate, normalize, and type match rows.

    Rows with either missing or non-numeric full-time goal values are excluded
    and counted. Invalid dates and missing team names are treated as data
    errors rather than silently discarded.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in matches]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = matches.copy()
    input_rows = len(cleaned)
    original_teams = cleaned.loc[:, list(TEAM_COLUMNS)].copy()

    cleaned["Date"] = _parse_dates(cleaned["Date"])
    invalid_dates = cleaned["Date"].isna()
    if invalid_dates.any():
        raise ValueError(f"Found {int(invalid_dates.sum())} rows with invalid dates")

    for column in TEAM_COLUMNS:
        cleaned[column] = cleaned[column].map(normalize_team_name).astype("string")
    invalid_teams = cleaned.loc[:, list(TEAM_COLUMNS)].isna().any(axis=1)
    if invalid_teams.any():
        raise ValueError(f"Found {int(invalid_teams.sum())} rows with missing team names")
    empty_teams = cleaned.loc[:, list(TEAM_COLUMNS)].apply(
        lambda column: column.str.strip().eq("")
    ).any(axis=1)
    if empty_teams.any():
        raise ValueError(f"Found {int(empty_teams.sum())} rows with empty team names")

    for column in GOAL_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    missing_goals = cleaned.loc[:, list(GOAL_COLUMNS)].isna().any(axis=1)
    dropped_goal_rows = int(missing_goals.sum())
    cleaned = cleaned.loc[~missing_goals].copy()

    for column in GOAL_COLUMNS:
        values = cleaned[column]
        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain non-negative whole-number goals")
        cleaned[column] = values.astype("int64")

    if "season" in cleaned:
        cleaned["season"] = cleaned["season"].astype("string")
    if "league" in cleaned:
        cleaned["league"] = cleaned["league"].astype("string")
    if "FTR" in cleaned:
        cleaned["FTR"] = cleaned["FTR"].astype("string").str.upper()

    changed_teams = cleaned.loc[:, list(TEAM_COLUMNS)].ne(original_teams).any(axis=1)
    summary = CleaningSummary(
        input_rows=input_rows,
        output_rows=len(cleaned),
        missing_goal_rows_dropped=dropped_goal_rows,
        team_names_normalized=int(changed_teams.sum()),
    )
    cleaned = cleaned.sort_values("Date", kind="stable").reset_index(drop=True)
    return cleaned, summary


def clean_matches_file(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Load the combined CSV, clean it, and save a processed CSV."""
    matches = pd.read_csv(input_path, low_memory=False)
    cleaned, summary = clean_matches(matches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"Input rows: {summary.input_rows}")
    print(f"Rows with missing goals dropped: {summary.missing_goal_rows_dropped}")
    print(f"Team-name rows normalized: {summary.team_names_normalized}")
    print(f"Clean rows: {summary.output_rows}")
    print(f"Saved cleaned data to {output_path}")
    return cleaned, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Clean the default collected dataset."""
    del argv
    clean_matches_file()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
