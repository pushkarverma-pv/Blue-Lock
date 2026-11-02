"""Download and combine football-data.co.uk season files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.config import (
    COLLECTION_SEASONS,
    DEFAULT_LEAGUES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
OUTPUT_PATH = PROCESSED_DATA_DIR / "matches.csv"
DATE_COLUMN = "Date"


class DataCollectionError(RuntimeError):
    """Raised when one or more requested source files cannot be collected."""


def _validate_source_code(league: str, season: str) -> None:
    """Reject malformed URL components before making a source request."""
    if not re.fullmatch(r"[A-Z][0-9]", league):
        raise ValueError(f"Invalid league code: {league!r}")
    if not re.fullmatch(r"[0-9]{4}", season):
        raise ValueError(f"Invalid season code: {season!r}")


def season_file_path(league: str, season: str, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Return the local path used for one league-season CSV."""
    _validate_source_code(league, season)
    return raw_dir / f"{league}_{season}.csv"


def download_season_file(
    league: str,
    season: str,
    raw_dir: Path = RAW_DATA_DIR,
    timeout: float = 30,
) -> Path:
    """Download a single season CSV without replacing an existing raw file."""
    path = season_file_path(league, season, raw_dir)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{season}/{league}.csv"
    request = Request(url, headers={"User-Agent": "football-predictor/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DataCollectionError(f"Could not download {url}: {exc}") from exc

    if not payload:
        raise DataCollectionError(f"Downloaded an empty file from {url}")

    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        temporary_path.write_bytes(payload)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def download_season_files(
    leagues: Sequence[str] = DEFAULT_LEAGUES,
    seasons: Sequence[str] = COLLECTION_SEASONS,
    raw_dir: Path = RAW_DATA_DIR,
    timeout: float = 30,
) -> list[Path]:
    """Download all requested files, preserving existing local downloads.

    Attempts every league-season pair and raises one error listing any failures.
    """
    paths: list[Path] = []
    failures: list[str] = []
    for league in leagues:
        for season in seasons:
            try:
                paths.append(download_season_file(league, season, raw_dir, timeout))
            except DataCollectionError as exc:
                failures.append(str(exc))

    if failures:
        raise DataCollectionError(
            "Some season files could not be collected:\n- " + "\n- ".join(failures)
        )
    return paths


def _read_season_file(path: Path, league: str, season: str) -> pd.DataFrame:
    """Read a source CSV and attach its league and season identifiers."""
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1")
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"{path} does not contain the required {DATE_COLUMN!r} column")
    metadata = pd.DataFrame(
        {"season": season, "league": league},
        index=frame.index,
    )
    frame = pd.concat([frame, metadata], axis=1)
    date_text = frame[DATE_COLUMN].astype("string").str.strip()
    iso_dates = date_text.str.match(
        r"^\d{4}-\d{1,2}-\d{1,2}(?:$|[ T])", na=False
    )
    parsed_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    parsed_dates.loc[iso_dates] = pd.to_datetime(
        date_text.loc[iso_dates], format="mixed", errors="coerce"
    )
    parsed_dates.loc[~iso_dates] = pd.to_datetime(
        date_text.loc[~iso_dates],
        dayfirst=True,
        format="mixed",
        errors="coerce",
    )
    frame[DATE_COLUMN] = parsed_dates
    return frame


def combine_season_files(
    leagues: Sequence[str] = DEFAULT_LEAGUES,
    seasons: Sequence[str] = COLLECTION_SEASONS,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Combine requested local files and sort matches by date.

    Missing files are reported together so manually downloaded files can be
    added to ``data/raw`` before retrying.
    """
    expected = [
        (league, season, season_file_path(league, season, raw_dir))
        for league in leagues
        for season in seasons
    ]
    missing = [path for _, _, path in expected if not path.is_file()]
    if missing:
        paths = "\n- ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing season files:\n- {paths}")

    frames = [
        _read_season_file(path, league, season)
        for league, season, path in expected
    ]
    if not frames:
        raise ValueError("At least one league-season pair must be configured")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    return matches.sort_values(DATE_COLUMN, kind="stable", na_position="last").reset_index(
        drop=True
    )


def print_collection_summary(matches: pd.DataFrame) -> None:
    """Print match counts, date range, and missing values by column."""
    print("Rows per league-season:")
    counts = matches.groupby(["league", "season"], dropna=False).size()
    print(counts.to_string())

    dates = matches[DATE_COLUMN].dropna()
    if dates.empty:
        print("Date range: unavailable (no dates could be parsed)")
    else:
        print(f"Date range: {dates.min().date()} to {dates.max().date()}")

    print("Missing values by column:")
    print(matches.isna().sum().to_string())


def collect_data(
    leagues: Sequence[str] = DEFAULT_LEAGUES,
    seasons: Sequence[str] = COLLECTION_SEASONS,
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = OUTPUT_PATH,
    timeout: float = 30,
) -> pd.DataFrame:
    """Download, combine, summarize, and save the requested match data."""
    download_season_files(leagues, seasons, raw_dir, timeout)
    matches = combine_season_files(leagues, seasons, raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print_collection_summary(matches)
    print(f"Saved {len(matches)} matches to {output_path}")
    return matches


def main(argv: Sequence[str] | None = None) -> int:
    """Run data collection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", nargs="+", default=list(DEFAULT_LEAGUES))
    parser.add_argument("--season", nargs="+", default=list(COLLECTION_SEASONS))
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        collect_data(leagues=args.league, seasons=args.season, timeout=args.timeout)
    except (DataCollectionError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
