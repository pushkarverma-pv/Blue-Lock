from pathlib import Path

import numpy as np

from src.calibration import MulticlassPlattCalibrator, expected_calibration_error
from src.evaluate import (
    calibration_table,
    save_calibration_comparison_plot,
)


def _validation_sample() -> tuple[np.ndarray, np.ndarray]:
    labels = np.array(["H", "D", "A"] * 20)
    probabilities = np.tile(
        np.array([[0.55, 0.25, 0.20], [0.30, 0.45, 0.25], [0.25, 0.25, 0.50]]),
        (20, 1),
    )
    return labels, probabilities


def test_platt_calibrator_returns_normalized_three_class_probabilities() -> None:
    labels, probabilities = _validation_sample()
    calibrator = MulticlassPlattCalibrator().fit(probabilities, labels)

    calibrated = calibrator.predict_proba(probabilities)

    assert calibrated.shape == probabilities.shape
    assert np.isfinite(calibrated).all()
    assert (calibrated >= 0).all()
    assert np.allclose(calibrated.sum(axis=1), 1)


def test_calibration_error_and_comparison_plot(tmp_path: Path) -> None:
    labels, probabilities = _validation_sample()
    calibrator = MulticlassPlattCalibrator().fit(probabilities, labels)
    calibrated = calibrator.predict_proba(probabilities)
    raw_table = calibration_table(labels, probabilities)
    calibrated_table = calibration_table(labels, calibrated)
    output = tmp_path / "reliability.png"

    error = expected_calibration_error(labels, calibrated)
    save_calibration_comparison_plot(raw_table, calibrated_table, output)

    assert 0 <= error <= 1
    assert output.is_file()
