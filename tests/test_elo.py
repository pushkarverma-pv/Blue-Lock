import pandas as pd
import pytest

from src.elo import pre_match_ratings


def _matches() -> pd.DataFrame:
    """Create two seasons with a same-day pair and a follow-up match."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2023-08-01", "2023-08-01", "2023-08-08", "2024-08-01"]
            ),
            "HomeTeam": ["A", "C", "A", "A"],
            "AwayTeam": ["B", "D", "B", "B"],
            "FTHG": [3, 0, 0, 1],
            "FTAG": [0, 0, 2, 1],
            "season": ["2324", "2324", "2324", "2425"],
        },
        index=[8, 3, 9, 2],
    )


def test_elo_returns_ratings_before_current_match_and_preserves_input_index() -> None:
    matches = _matches()
    ratings = pre_match_ratings(matches)

    assert ratings.index.tolist() == matches.index.tolist()
    assert ratings.loc[8, "home_elo"] == 1500
    assert ratings.loc[8, "away_elo"] == 1500
    assert ratings.loc[3, "home_elo"] == 1500
    assert ratings.loc[3, "away_elo"] == 1500
    assert ratings.loc[9, "home_elo"] != 1500
    one_season = _matches().iloc[[0]].copy()
    next_season = one_season.copy()
    next_season["Date"] = pd.Timestamp("2024-08-01")
    next_season["season"] = "2425"
    two_seasons = pd.concat([one_season, next_season], ignore_index=True)
    both_ratings = pre_match_ratings(two_seasons)
    without_regression = pre_match_ratings(two_seasons, season_regression=0)
    assert both_ratings.iloc[1]["home_elo"] == pytest.approx(
        1500 + (without_regression.iloc[1]["home_elo"] - 1500) * 0.75
    )


def test_elo_same_date_result_does_not_change_other_same_date_rating() -> None:
    matches = _matches()
    changed = matches.copy()
    changed.loc[8, ["FTHG", "FTAG"]] = [0, 5]

    before = pre_match_ratings(matches)
    after = pre_match_ratings(changed)

    pd.testing.assert_frame_equal(before.loc[[8, 3]], after.loc[[8, 3]])


def test_elo_without_goal_difference_multiplier_is_supported() -> None:
    matches = _matches().iloc[:1]
    ratings = pre_match_ratings(matches, goal_difference_scale=0)

    assert ratings.loc[8, "elo_diff"] == 0
    with pytest.raises(ValueError, match="non-negative"):
        pre_match_ratings(matches, goal_difference_scale=-1)
