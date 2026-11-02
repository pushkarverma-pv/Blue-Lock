from pathlib import Path

import pandas as pd

from src.eda import generate_eda_report


def test_generate_eda_report_contains_computed_summaries_and_plots(
    tmp_path: Path,
) -> None:
    matches = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-08-01", "2023-08-02", "2024-08-01"]),
            "HomeTeam": ["A", "B", "A"],
            "AwayTeam": ["B", "A", "B"],
            "FTHG": [2, 0, 1],
            "FTAG": [1, 0, 3],
            "FTR": ["H", "D", "A"],
            "season": ["2324", "2324", "2425"],
        }
    )

    report_path = tmp_path / "reports" / "eda_summary.md"
    goal_plot_path = tmp_path / "reports" / "goals.png"
    advantage_plot_path = tmp_path / "reports" / "advantage.png"
    report = generate_eda_report(
        matches, report_path, goal_plot_path, advantage_plot_path
    )

    assert "Matches: 3" in report
    assert "| Home win | 33.33% |" in report
    assert "| A | 1.000 | 1.333 |" in report
    assert "Home advantage by season" in report
    assert report_path.is_file()
    assert goal_plot_path.is_file()
    assert advantage_plot_path.is_file()
