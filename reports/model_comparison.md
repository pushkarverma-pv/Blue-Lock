# Model comparison

- Training seasons: 1516, 1617, 1718, 1819, 1920, 2021, 2122
- Validation season: 2223
- Test seasons: 2324, 2425, 2526
- Features: 69 generated pre-match features; bookmaker odds are excluded from model inputs.
- Hyperparameters selected by 3-split expanding-window `TimeSeriesSplit` over training seasons only; test seasons were held out until final reporting.
- Scoreline grids use independent Poisson distributions truncated and renormalized to 0-10 goals per side.
- Outcome order for probabilities is home win, draw, away win. RPS uses the ordinal order away win, draw, home win.

## Chronological cross-validation selection

| Model | Selected parameters | CV objective |
|---|---|---:|
| Poisson regression | `{'alpha': 1.0}` | Mean home/away goals MAE 0.9230 |
| XGBoost Poisson goals | `{'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05}` | Mean home/away goals MAE 0.9355 |
| XGBoost direct W/D/L | `{'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05}` | Multiclass log loss 0.9811 |

## Results

| Model | Split | N | Accuracy | Log loss | Brier score | RPS | Home goals MAE | Away goals MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Always home win | Validation | 380 | 0.4842 | 17.8147 | 1.0316 | 0.4013 | — | — |
| Class frequency | Validation | 380 | 0.4842 | 1.0510 | 0.6339 | 0.2287 | — | — |
| Elo logistic | Validation | 380 | 0.5526 | 0.9743 | 0.5795 | 0.2015 | — | — |
| Bookmaker | Validation | 380 | 0.5605 | 0.9658 | 0.5737 | 0.1991 | — | — |
| Poisson regression | Validation | 380 | 0.5342 | 0.9793 | 0.5827 | 0.2032 | 0.9955 | 0.9233 |
| XGBoost Poisson goals | Validation | 380 | 0.5500 | 0.9646 | 0.5727 | 0.1989 | 0.9987 | 0.9092 |
| XGBoost direct W/D/L | Validation | 380 | 0.5421 | 0.9798 | 0.5826 | 0.2017 | — | — |
| Always home win | Test | 1140 | 0.4316 | 19.6326 | 1.1368 | 0.4461 | — | — |
| Class frequency | Test | 1140 | 0.4316 | 1.0726 | 0.6493 | 0.2322 | — | — |
| Elo logistic | Test | 1140 | 0.5289 | 0.9827 | 0.5865 | 0.2013 | — | — |
| Bookmaker | Test | 1140 | 0.5421 | 0.9650 | 0.5740 | 0.1955 | — | — |
| Poisson regression | Test | 1140 | 0.5404 | 0.9834 | 0.5848 | 0.2006 | 0.9821 | 0.8794 |
| XGBoost Poisson goals | Test | 1140 | 0.5246 | 0.9898 | 0.5901 | 0.2028 | 0.9780 | 0.8705 |
| XGBoost direct W/D/L | Test | 1140 | 0.5298 | 0.9935 | 0.5906 | 0.2019 | — | — |

The goal MAEs are omitted for baselines without goal estimates and the direct W/D/L classifier.
Bookmaker scores are computed only for matches with valid bookmaker odds; their `N` is shown separately.
Model estimates are experimental baselines, not guarantees. Results should be interpreted with the high variance of football outcomes in mind.
