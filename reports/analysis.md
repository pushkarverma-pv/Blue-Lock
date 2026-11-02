# Calibration and model analysis

## Evaluation protocol

- Train seasons: 1516, 1617, 1718, 1819, 1920, 2021, 2122.
- Calibration/selection season: 2223.
- Held-out test seasons: 2324, 2425, 2526 (1140 matches).
- All model parameters were tuned by chronological cross-validation on training seasons. Platt calibrators were fit only on validation predictions; test labels were not used for calibration or model selection.
- A Platt calibrator is applied only when its validation log loss and Brier score are both lower than the uncalibrated values. Validation calibration metrics are in-sample and should not be read as unbiased estimates.
- The final error analysis model was selected by validation log loss after the calibration rule: **XGBoost Poisson goals**.

## Calibration results

| Model | Platt applied | Validation raw log loss | Validation Platt log loss | Validation raw Brier | Validation Platt Brier | Test raw log loss | Test final log loss | Test raw ECE | Test final ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Elo logistic | yes | 0.9743 | 0.9688 | 0.5795 | 0.5760 | 0.9827 | 0.9897 | 0.0269 | 0.0375 |
| Poisson regression | yes | 0.9793 | 0.9745 | 0.5827 | 0.5797 | 0.9834 | 0.9917 | 0.0225 | 0.0393 |
| XGBoost Poisson goals | yes | 0.9646 | 0.9586 | 0.5727 | 0.5693 | 0.9898 | 0.9952 | 0.0264 | 0.0355 |
| XGBoost direct W/D/L | yes | 0.9798 | 0.9693 | 0.5826 | 0.5760 | 0.9935 | 0.9948 | 0.0258 | 0.0366 |

Selected model's test metrics after the validation-only calibration choice: accuracy 0.5228, log loss 0.9952, Brier score 0.5938, RPS 0.2045.

Reliability plots show each result class (home win, draw, away win) before and after the validation-fitted calibration:

- [Validation reliability plot](calibration_analysis_validation.png)
- [Test reliability plot](calibration_analysis_test.png)
- [Validation calibration bins](calibration_analysis_validation.csv)
- [Test calibration bins](calibration_analysis_test.csv)

Validation-selected Platt calibration worsened test log loss for 4 of 4 models to which it was applied, and worsened test Brier score for 4. This is why the test comparison is reported separately from the validation selection result.

## Test performance breakdown

Early season is defined as fewer than five prior matches for both sides (the largest of their two season counts is below five); promoted-team involvement is based on league entry relative to the preceding season; the big six are Arsenal, Chelsea, Liverpool, Man City, Man United, and Tottenham.

| Group | N | Accuracy | Log loss | RPS |
|---|---:|---:|---:|---:|
| Season 2324 | 380 | 0.5579 | 0.9442 | 0.1946 |
| Season 2425 | 380 | 0.5184 | 1.0058 | 0.2077 |
| Season 2526 | 380 | 0.4921 | 1.0357 | 0.2113 |
| Early season (<5 prior matches) | 149 | 0.5503 | 0.9456 | 0.1854 |
| Later season | 991 | 0.5187 | 1.0027 | 0.2074 |
| Promoted team involved | 216 | 0.5972 | 0.9447 | 0.1837 |
| No promoted team involved | 924 | 0.5054 | 1.0071 | 0.2094 |
| Big-six match | 594 | 0.5774 | 0.9315 | 0.1911 |
| Other match | 546 | 0.4634 | 1.0646 | 0.2192 |

## Worst test misses

Matches are ranked by the final selected model's probability assigned to the true W/D/L result; lower true-outcome probability is a worse miss.
Among the ten lowest-probability true outcomes, the observed result counts were A: 10. This is a descriptive pattern in this test period, not a general claim about away outcomes.

| Date | Home | Away | Score | Actual result | Probability of actual result |
|---|---|---|---:|---|---:|
| 2024-04-14 | Liverpool | Crystal Palace | 0-1 | A | 0.0540 |
| 2024-04-14 | Arsenal | Aston Villa | 0-2 | A | 0.0687 |
| 2024-11-10 | Tottenham | Ipswich | 1-2 | A | 0.0706 |
| 2025-04-02 | Bournemouth | Ipswich | 1-2 | A | 0.0756 |
| 2026-05-24 | Man City | Aston Villa | 1-2 | A | 0.0845 |
| 2024-12-26 | Chelsea | Fulham | 1-2 | A | 0.0883 |
| 2025-02-22 | Bournemouth | Wolves | 0-1 | A | 0.0896 |
| 2025-02-22 | Arsenal | West Ham | 0-1 | A | 0.0915 |
| 2023-12-26 | Newcastle | Nottingham Forest | 1-3 | A | 0.0942 |
| 2024-12-15 | Man City | Man United | 1-2 | A | 0.0942 |

## Feature interpretation

Mean XGBoost feature importance averaged across home-goal, away-goal, and direct-result models:

| Feature | Average importance |
|---|---:|
| elo_diff | 0.1417 |
| home_elo | 0.0431 |
| home_home_shots_on_target_last_10 | 0.0418 |
| away_overall_shots_last_5 | 0.0317 |
| away_overall_shots_on_target_last_5 | 0.0312 |
| home_away_shots_last_10 | 0.0287 |
| home_overall_shots_on_target_last_10 | 0.0283 |
| home_home_shots_last_10 | 0.0272 |
| away_overall_goals_conceded_last_10 | 0.0272 |
| away_home_shots_on_target_last_10 | 0.0271 |

Largest standardized Poisson coefficients (coefficients describe association with the modeled log expected goals):

| Feature | Home-goal coefficient | Away-goal coefficient | Mean absolute coefficient |
|---|---:|---:|---:|
| elo_diff | 0.0446 | -0.0393 | 0.0420 |
| home_elo | 0.0491 | -0.0317 | 0.0404 |
| home_home_shots_last_10 | 0.0233 | -0.0177 | 0.0205 |
| away_elo | -0.0154 | 0.0251 | 0.0202 |
| away_away_shots_last_5 | 0.0085 | 0.0308 | 0.0197 |
| away_overall_shots_last_5 | -0.0012 | 0.0357 | 0.0185 |
| away_home_shots_last_10 | -0.0168 | 0.0194 | 0.0181 |
| away_away_goals_conceded_last_10 | 0.0239 | -0.0118 | 0.0179 |
| home_away_goals_scored_last_10 | 0.0110 | -0.0230 | 0.0170 |
| away_overall_shots_last_10 | 0.0053 | 0.0253 | 0.0153 |

Feature importance and regression coefficients are descriptive, not causal. Rolling form, team strength, and venue effects may be correlated.

## Limitations

- Football outcomes are high variance; a single season can differ materially from long-run performance.
- The Poisson scoreline grid assumes conditionally independent home and away goals and truncates each side at ten goals before renormalizing.
- Bookmaker odds are an external benchmark, not an input feature for these first-version models.
- Calibration is fit on one validation season; shifts in league scoring or team strength can make it stale.
- Promotion labels are inferred from participating team names in adjacent seasons. Injuries, lineups, tactical changes, and other match-day information are not modeled.
- Test metrics describe the supplied 2324-2526 files only and should not be generalized beyond that period.
