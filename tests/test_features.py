import pandas as pd
import pytest

from src.features import build_features, feature_columns, feature_nan_counts


def _matches() -> pd.DataFrame:
    """Build a compact match history with same-day fixtures and two seasons."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-01",
                    "2023-08-08",
                    "2023-08-15",
                    "2024-08-01",
                ]
            ),
            "HomeTeam": ["A", "C", "A", "B", "A"],
            "AwayTeam": ["B", "D", "B", "A", "C"],
            "FTHG": [2, 0, 1, 0, 1],
            "FTAG": [0, 0, 1, 2, 1],
            "HS": [10, 8, 12, 9, 11],
            "AS": [5, 7, 9, 13, 10],
            "HST": [4, 2, 5, 3, 4],
            "AST": [2, 1, 3, 4, 3],
            "season": ["2324", "2324", "2324", "2324", "2425"],
            "league": ["E0"] * 5,
        }
    )


def test_features_use_only_matches_before_the_current_date() -> None:
    matches = _matches()
    complete = build_features(matches)
    match_date = matches.loc[2, "Date"]
    prior_matches = matches.loc[matches["Date"] < match_date]
    query = pd.concat([prior_matches, matches.iloc[[2]]], ignore_index=True)
    recomputed = build_features(query)
    columns = feature_columns()

    pd.testing.assert_series_equal(
        complete.loc[2, columns].reset_index(drop=True),
        recomputed.iloc[-1][columns].reset_index(drop=True),
        check_names=False,
    )


def test_changing_a_match_result_does_not_change_its_own_features() -> None:
    matches = _matches()
    baseline = build_features(matches)
    changed = matches.copy()
    changed.loc[2, ["FTHG", "FTAG"]] = [0, 4]
    changed_features = build_features(changed)

    pd.testing.assert_series_equal(
        baseline.loc[2, feature_columns()],
        changed_features.loc[2, feature_columns()],
    )


def test_same_day_results_do_not_leak_between_fixtures() -> None:
    matches = _matches()
    before = build_features(matches)
    changed = matches.copy()
    changed.loc[0, ["FTHG", "FTAG"]] = [0, 5]
    after = build_features(changed)

    pd.testing.assert_series_equal(
        before.loc[1, feature_columns()],
        after.loc[1, feature_columns()],
    )


def test_first_appearances_use_fallback_and_features_have_no_nans() -> None:
    features = build_features(_matches())

    assert features.loc[0, "home_first_appearance"]
    assert features.loc[0, "away_first_appearance"]
    assert features.loc[0, "home_overall_goals_scored_last_5"] == pytest.approx(1.35)
    assert (feature_nan_counts(features) == 0).all()
    assert features.columns.is_unique


def test_form_windows_include_only_available_prior_matches() -> None:
    features = build_features(_matches())

    # Before A's third match, only its first match is in its overall history.
    assert features.loc[2, "home_overall_goals_scored_last_5"] == 2
    assert features.loc[2, "home_overall_goals_scored_last_10"] == 2
    assert features.loc[2, "home_matches_played_season"] == 1
    assert features.loc[2, "home_days_rest"] == 7


def test_required_columns_are_validated() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        build_features(pd.DataFrame({"Date": []}))


def test_feature_position_mode_matches_full_pipeline() -> None:
    matches = _matches()
    full = build_features(matches)
    selected = build_features(matches, feature_positions=[2])

    pd.testing.assert_series_equal(
        full.loc[2, feature_columns()].reset_index(drop=True),
        selected.iloc[0][feature_columns()].reset_index(drop=True),
        check_names=False,
    )
    assert len(selected) == 1
