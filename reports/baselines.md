# Baseline evaluation

- Training seasons: 1516, 1617, 1718, 1819, 1920, 2021, 2122
- Validation season: 2223
- Test seasons: 2324, 2425, 2526
- All predictions and probability mappings were fit using training seasons only.
- Elo ratings are generated pre-match using K=20, a 60-point home advantage, a 1.0 goal-margin scale, and 25% season regression; past results update subsequent ratings.
- Bookmaker probabilities use inverse odds normalized to remove the overround; bookmaker rows without valid odds are excluded.
- RPS uses the ordered outcomes away win, draw, home win. Brier score sums squared errors across the three classes.

## Results

| Baseline | Split | N | Accuracy | Log loss | Brier score | RPS |
|---|---|---:|---:|---:|---:|---:|
| Always home win | Validation | 380 | 0.4842 | 17.8147 | 1.0316 | 0.4013 |
| Class frequency | Validation | 380 | 0.4842 | 1.0510 | 0.6339 | 0.2287 |
| Elo logistic | Validation | 380 | 0.5526 | 0.9743 | 0.5795 | 0.2015 |
| Bookmaker | Validation | 380 | 0.5605 | 0.9658 | 0.5737 | 0.1991 |
| Always home win | Test | 1140 | 0.4316 | 19.6326 | 1.1368 | 0.4461 |
| Class frequency | Test | 1140 | 0.4316 | 1.0726 | 0.6493 | 0.2322 |
| Elo logistic | Test | 1140 | 0.5289 | 0.9827 | 0.5865 | 0.2013 |
| Bookmaker | Test | 1140 | 0.5421 | 0.9650 | 0.5740 | 0.1955 |

## Validation Elo calibration

Reliability plot: [calibration_validation.png](calibration_validation.png)
Calibration data: [calibration_validation.csv](calibration_validation.csv)


## Test Elo calibration

Reliability plot: [calibration_test.png](calibration_test.png)
Calibration data: [calibration_test.csv](calibration_test.csv)

