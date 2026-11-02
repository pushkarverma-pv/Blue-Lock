"""Streamlit interface for probabilistic football match predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.prediction_log import append_prediction
from src.predict import (
    HISTORY_PATH,
    history_is_stale,
    latest_match_date,
    latest_season_teams,
    predict,
)

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")


@st.cache_data
def _available_teams(history_path: str) -> list[str]:
    """Load teams appearing in the most recent season."""
    history = pd.read_csv(
        history_path, usecols=["season", "HomeTeam", "AwayTeam"]
    )
    return latest_season_teams(history)


@st.cache_data
def _last_match_date(history_path: str) -> pd.Timestamp:
    """Load the latest historical match date."""
    history = pd.read_csv(history_path, usecols=["Date"])
    return latest_match_date(history)


def _scoreline_probabilities(
    grid: np.ndarray, home_team: str, away_team: str
) -> dict[str, float]:
    """Derive goal-market probabilities from a normalized scoreline grid."""
    probabilities = np.asarray(grid, dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != probabilities.shape[1]
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-6)
    ):
        raise ValueError("Scoreline grid must be a normalized square probability matrix")

    home_goals, away_goals = np.indices(probabilities.shape)
    return {
        "Over 2.5 goals": float(probabilities[home_goals + away_goals >= 3].sum()),
        "Under 2.5 goals": float(probabilities[home_goals + away_goals <= 2].sum()),
        "Both teams to score": float(probabilities[1:, 1:].sum()),
        f"{home_team} clean sheet": float(probabilities[:, 0].sum()),
        f"{away_team} clean sheet": float(probabilities[0, :].sum()),
    }


def main() -> None:
    """Render the prediction form and probability-focused results."""
    st.title("Football Match Score Predictor")
    st.write(
        "Explore estimated goals, match-outcome probabilities, and a full "
        "scoreline distribution."
    )
    st.info(
        "These values are probabilities, not guarantees. Football matches are "
        "high-variance; the most-likely scoreline is not a certain result."
    )

    history_path = Path(HISTORY_PATH)
    if not history_path.is_file():
        st.error(
            f"Cleaned match history is missing at `{history_path}`. "
            "Run `python -m src.eda` first."
        )
        return
    try:
        teams = _available_teams(str(history_path))
        last_match = _last_match_date(str(history_path))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load team list from match history: {exc}")
        return
    if not teams:
        st.error("No teams were found in the cleaned match history.")
        return

    left, right = st.columns(2)
    with left:
        home_team = st.selectbox("Home team", teams, key="home_team")
    with right:
        away_options = [team for team in teams if team != home_team]
        away_team = st.selectbox("Away team", away_options, key="away_team")
    match_date = st.date_input("Match date")
    if history_is_stale(match_date, last_match):
        days_old = (match_date - last_match.date()).days
        st.warning(
            f"History ends on {last_match.date().isoformat()} ({days_old} days "
            "before the selected match date). Predictions may be less reliable."
        )

    save_to_log = st.checkbox("Save this prediction to the prediction log", value=True)
    if st.button("Predict match", type="primary"):
        try:
            result = predict(home_team, away_team, match_date.isoformat())
            if save_to_log:
                append_prediction(result)
                st.success("Prediction saved to reports/prediction_log.csv.")
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return

        st.subheader(f"{result['home_team']} vs {result['away_team']}")
        expected = result["expected_goals"]
        st.metric(
            "Expected goals",
            f"{result['home_team']} {expected['home']:.2f} – "
            f"{expected['away']:.2f} {result['away_team']}",
        )

        outcome = result["outcome_probabilities"]
        outcome_labels = [
            f"{result['home_team']} win",
            "Draw",
            f"{result['away_team']} win",
        ]
        outcome_values = [
            outcome["home_win"],
            outcome["draw"],
            outcome["away_win"],
        ]
        chart_column, score_column = st.columns(2)
        with chart_column:
            st.markdown("#### Win / draw / loss probabilities")
            figure, axis = plt.subplots(figsize=(7, 2.6))
            bars = axis.barh(
                outcome_labels,
                outcome_values,
                color=["#2563eb", "#a855f7", "#f97316"],
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.1%}" for value in outcome_values],
                padding=5,
            )
            axis.set_xlim(0, 1.08)
            axis.set_xlabel("Probability")
            axis.invert_yaxis()
            figure.tight_layout()
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
        with score_column:
            st.markdown("#### Five most likely scorelines")
            scorelines = pd.DataFrame(result["top_scorelines"])[
                ["score", "probability"]
            ].copy()
            scorelines["probability"] = scorelines["probability"].map(
                lambda value: f"{value:.1%}"
            )
            st.dataframe(
                scorelines,
                hide_index=True,
                use_container_width=True,
            )

        grid = pd.DataFrame(
            result["scoreline_grid"],
            index=result["goals_range"],
            columns=result["goals_range"],
        )
        grid_probabilities = _scoreline_probabilities(
            grid.to_numpy(), result["home_team"], result["away_team"]
        )
        st.markdown("#### Goal probabilities")
        metric_columns = st.columns(len(grid_probabilities))
        for column, (label, probability) in zip(
            metric_columns, grid_probabilities.items()
        ):
            column.metric(label, f"{probability:.1%}")

        with st.expander("Show detailed scoreline grid", expanded=False):
            st.markdown("#### Scoreline probability grid")
            figure, axis = plt.subplots(figsize=(9, 7))
            image = axis.imshow(grid.to_numpy(), cmap="Blues", origin="upper")
            figure.colorbar(image, ax=axis, label="Probability")
            for home_goals in range(len(grid.index)):
                for away_goals in range(len(grid.columns)):
                    axis.text(
                        away_goals,
                        home_goals,
                        f"{grid.iat[home_goals, away_goals]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            axis.set(
                xticks=range(len(grid.columns)),
                yticks=range(len(grid.index)),
                xticklabels=grid.columns,
                yticklabels=grid.index,
                xlabel=f"{result['away_team']} goals",
                ylabel=f"{result['home_team']} goals",
                title="Probability of each scoreline",
            )
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
        st.caption(
            f"Most-likely scoreline: **{result['most_likely_score']}**. "
            "This is the maximum-probability score, not a guarantee."
        )


if __name__ == "__main__":
    main()
