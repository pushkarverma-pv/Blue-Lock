# Feature engineering summary

- Matches: 4260
- Date range: 2015-08-08 to 2026-10-25
- Generated features: 69
- All match features are calculated from matches strictly before the match date.
- Same-day fixtures share a pre-date snapshot; their results are applied only after every match on that date receives features.
- Form windows use available prior matches up to their window size. First appearances use an expanding prior league/context average, or a neutral cold-start prior before any league results exist.
- Elo is also recorded before the current match result is applied.

## Missing values by generated feature

| Feature | Missing values |
|---|---:|
| home_overall_goals_scored_last_5 | 0 |
| home_overall_goals_conceded_last_5 | 0 |
| home_overall_shots_last_5 | 0 |
| home_overall_shots_on_target_last_5 | 0 |
| home_overall_points_last_5 | 0 |
| home_overall_goals_scored_last_10 | 0 |
| home_overall_goals_conceded_last_10 | 0 |
| home_overall_shots_last_10 | 0 |
| home_overall_shots_on_target_last_10 | 0 |
| home_overall_points_last_10 | 0 |
| home_home_goals_scored_last_5 | 0 |
| home_home_goals_conceded_last_5 | 0 |
| home_home_shots_last_5 | 0 |
| home_home_shots_on_target_last_5 | 0 |
| home_home_points_last_5 | 0 |
| home_home_goals_scored_last_10 | 0 |
| home_home_goals_conceded_last_10 | 0 |
| home_home_shots_last_10 | 0 |
| home_home_shots_on_target_last_10 | 0 |
| home_home_points_last_10 | 0 |
| home_away_goals_scored_last_5 | 0 |
| home_away_goals_conceded_last_5 | 0 |
| home_away_shots_last_5 | 0 |
| home_away_shots_on_target_last_5 | 0 |
| home_away_points_last_5 | 0 |
| home_away_goals_scored_last_10 | 0 |
| home_away_goals_conceded_last_10 | 0 |
| home_away_shots_last_10 | 0 |
| home_away_shots_on_target_last_10 | 0 |
| home_away_points_last_10 | 0 |
| away_overall_goals_scored_last_5 | 0 |
| away_overall_goals_conceded_last_5 | 0 |
| away_overall_shots_last_5 | 0 |
| away_overall_shots_on_target_last_5 | 0 |
| away_overall_points_last_5 | 0 |
| away_overall_goals_scored_last_10 | 0 |
| away_overall_goals_conceded_last_10 | 0 |
| away_overall_shots_last_10 | 0 |
| away_overall_shots_on_target_last_10 | 0 |
| away_overall_points_last_10 | 0 |
| away_home_goals_scored_last_5 | 0 |
| away_home_goals_conceded_last_5 | 0 |
| away_home_shots_last_5 | 0 |
| away_home_shots_on_target_last_5 | 0 |
| away_home_points_last_5 | 0 |
| away_home_goals_scored_last_10 | 0 |
| away_home_goals_conceded_last_10 | 0 |
| away_home_shots_last_10 | 0 |
| away_home_shots_on_target_last_10 | 0 |
| away_home_points_last_10 | 0 |
| away_away_goals_scored_last_5 | 0 |
| away_away_goals_conceded_last_5 | 0 |
| away_away_shots_last_5 | 0 |
| away_away_shots_on_target_last_5 | 0 |
| away_away_points_last_5 | 0 |
| away_away_goals_scored_last_10 | 0 |
| away_away_goals_conceded_last_10 | 0 |
| away_away_shots_last_10 | 0 |
| away_away_shots_on_target_last_10 | 0 |
| away_away_points_last_10 | 0 |
| home_elo | 0 |
| away_elo | 0 |
| elo_diff | 0 |
| home_days_rest | 0 |
| away_days_rest | 0 |
| home_matches_played_season | 0 |
| away_matches_played_season | 0 |
| home_first_appearance | 0 |
| away_first_appearance | 0 |
