# Football Match Score Predictor

A Python application for estimating expected home/away goals, scoreline
probabilities, and home-win/draw/away-win probabilities from Premier League
match history. It uses no bookmaker odds as model inputs. These estimates
describe uncertainty; football outcomes are not guarantees.

## Project layout

- `src/`: data collection, cleaning, leakage-safe features, evaluation,
  baselines, models, prediction API, and training CLI.
- `data/raw/`: untouched football-data.co.uk CSV downloads.
- `data/processed/`: combined, cleaned, and feature-engineered data.
- `notebooks/`: EDA and model-analysis notebooks; reusable logic stays in `src/`.
- `models/`: deployment model and metadata (ignored by Git).
- `reports/`: computed evaluation reports and plots.
- `app/streamlit_app.py`: interactive probability-focused demo.

## Setup

Python 3.10 or later is required. From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On systems with a different supported Python version, create the environment
with that interpreter instead. Raw files are never edited. If a data download
fails, download the listed CSVs into `data/raw/` using the expected filename,
for example `E0_1516.csv`.

## Build data and evaluate

Run each command in order:

```powershell
python -m src.data_loader
python -m src.eda
python -m src.baselines
python -m src.features
python -m src.models
python -m src.analysis
python -m pytest -q
```

The historical evaluation seasons are `1516` through `2526`. Collection also
includes the live `2627` season; keeping it separate leaves the fixed
train/validation/test split unchanged.
The fixed time split is:

- Training and model selection: `1516`–`2122`
- Validation and calibration choice: `2223`
- Final historical test report: `2324`–`2526`

`src.models` tunes only on chronological folds within training data. The
evaluation report is recorded before deployment refitting. Do not use the test
seasons to select further experiments; future improvement experiments should
use rolling-origin validation on training and validation seasons and reserve
the test set for a single final report.

Generated files include:

- `data/processed/matches.csv`
- `data/processed/cleaned_matches.csv`
- `data/processed/features.parquet`
- `reports/eda_summary.md`
- `reports/baselines.md`
- `reports/model_comparison.md`
- `reports/analysis.md`
- Calibration tables and plots in `reports/`

## Model selection and reported results

**Production model: Poisson regression.** Its training-only chronological
cross-validation goals MAE was `0.9230`, better than the XGBoost Poisson
model's `0.9355`, and it directly provides expected goals for constructing a
scoreline probability grid. This choice is based on the cross-validation
results, not the held-out test metrics. The deployment regressors are refit
using all available seasons after the pre-refit evaluation is recorded.

| Model | Validation accuracy | Validation log loss | Test accuracy | Test log loss | Test home goals MAE | Test away goals MAE |
|---|---:|---:|---:|---:|---:|---:|
| Poisson regression (production) | 53.42% | 0.9793 | 54.04% | 0.9834 | 0.9821 | 0.8794 |
| Elo logistic (reference/fallback) | 55.26% | 0.9743 | 52.89% | 0.9827 | — | — |
| Bookmaker no-vig benchmark | 56.05% | 0.9658 | 54.21% | 0.9650 | — | — |

On the test set, Poisson regression is **effectively tied with Elo logistic**
by log-loss point estimate (`0.9834` vs `0.9827`); it does **not** beat Elo.
No statistical significance test was performed, so this is not evidence that
the models are statistically equivalent. Poisson regression is clearly behind
the bookmaker benchmark (`0.9650`). The Elo logistic baseline is the
documented reference/fallback model; the current prediction API ships the
Poisson scoreline model. Accuracy and log loss are based on the metrics
actually computed in `reports/model_comparison.md`.

Validation-fitted calibration improved validation metrics but worsened test
metrics for every evaluated model. Therefore **calibrated models are not
shipped**. The production predictor returns uncalibrated Poisson probabilities
and explicitly labels them as estimates, not guarantees.

## Train the deployment model

After running the evaluation pipeline above, fit the deployment estimators:

```powershell
python -m src.train --league E0
```

This command reads the already-recorded Poisson CV-selected regularization
parameter from `reports/model_comparison.md` and refits the two goal
regressors on **all available seasons**, including any `2627` matches already
collected. It does not tune on or re-evaluate the test set. Pre-refit
validation/test metrics and CV details are retained in
`models/model_metadata.json`; the model artifact is
`models/poisson_model.joblib`.

### Refresh and retrain

Place new or updated season CSVs in `data/raw/` using names such as
`E0_2627.csv`, then run:

```powershell
python -m src.train --league E0 --update
```

The collector and update path include the configured live `2627` season. The
update path combines available raw seasons with existing processed data,
deduplicates matches by league/season/date/teams, rebuilds cleaned data and
features, and refits the deployment model. Raw source files remain read-only.
This update step does not rerun model selection or alter the saved pre-refit
evaluation metrics.

## Predict a match

After training:

```python
from src.predict import predict

result = predict("Arsenal", "Liverpool", "2026-08-15")
print(result["expected_goals"])
print(result["outcome_probabilities"])
print(result["top_scorelines"])
```

The `date` argument is optional; when omitted, it defaults to the day after
the latest available match. The function only uses history strictly before
the prediction date and raises a helpful error for unknown teams. The returned
dictionary includes expected goals, home/draw/away probabilities, the top five
scorelines with probabilities, the most-likely score, and the complete
0–10-goal probability grid.

## Run the Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

Select home and away teams and a date to view expected goals, a W/D/L
probability chart, the five most likely scorelines, and a scoreline-probability
heatmap. The UI emphasizes that probabilities are not guarantees and does not
present a single score as certain. Team selectors use only teams in the latest
season present in the cleaned data. When the selected match date is more than
14 days after the latest match in the history, the app displays a warning that
the history is stale and predictions may be less reliable.

The app saves each prediction to `reports/prediction_log.csv` by default; turn
off **Save this prediction to the prediction log** to skip an individual
entry. Once matches have been added to the cleaned history, settle matching
fixtures and refresh cumulative metrics with:

```powershell
python -m src.prediction_log --settle
```

Settlement fills actual goals/results without overwriting already settled
entries and writes running log loss, multiclass Brier score, and accuracy to
`reports/prediction_log_metrics.md`. Multiple predictions for the same fixture
are retained and scored independently. The command accepts `--log`, `--matches`,
and `--metrics-report` overrides for alternate files.

## Limitations

- Football outcomes are high variance; one season may differ substantially from
  long-run results.
- Independent Poisson goals simplify score dependence and the scoreline grid
  truncates each team's goals at ten before renormalizing.
- The model does not include injuries, lineups, tactical changes, or
  match-day information.
- The reported test metrics apply only to seasons `2324`–`2526`; future
  experiments must avoid test-guided tuning.
- Calibration was not deployed because validation-fitted calibration
  worsened held-out test metrics.
