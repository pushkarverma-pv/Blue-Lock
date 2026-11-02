"""Multiclass probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.evaluate import OUTCOME_ORDER, calibration_table


@dataclass
class MulticlassPlattCalibrator:
    """One-vs-rest sigmoid calibrators for H/D/A class probabilities."""

    models: list[LogisticRegression] | None = None

    def fit(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> "MulticlassPlattCalibrator":
        """Fit one binary sigmoid mapping per outcome from validation predictions."""
        probabilities = self._validate(probabilities)
        outcomes = np.asarray(labels, dtype=str)
        if len(outcomes) != len(probabilities) or len(outcomes) == 0:
            raise ValueError("labels and probabilities must have matching non-empty lengths")
        if not set(outcomes).issubset(OUTCOME_ORDER):
            raise ValueError("labels must be H, D, or A")
        models: list[LogisticRegression] = []
        for class_index, outcome in enumerate(OUTCOME_ORDER):
            binary_target = (outcomes == outcome).astype(int)
            if np.unique(binary_target).size != 2:
                raise ValueError(f"validation data must contain positive and negative {outcome} labels")
            values = np.clip(probabilities[:, class_index], 1e-6, 1.0 - 1e-6)
            logits = np.log(values / (1.0 - values)).reshape(-1, 1)
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )
            model.fit(logits, binary_target)
            models.append(model)
        self.models = models
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply each outcome's sigmoid and normalize to a probability simplex."""
        if self.models is None:
            raise ValueError("calibrator must be fitted before predicting")
        values = self._validate(probabilities)
        calibrated = np.zeros_like(values)
        for class_index, model in enumerate(self.models):
            clipped = np.clip(values[:, class_index], 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            calibrated[:, class_index] = model.predict_proba(logits)[:, 1]
        totals = calibrated.sum(axis=1, keepdims=True)
        if (totals <= 0).any() or not np.isfinite(totals).all():
            raise ValueError("calibration produced invalid probability totals")
        return calibrated / totals

    @staticmethod
    def _validate(probabilities: np.ndarray) -> np.ndarray:
        """Check H/D/A input matrix shape and values."""
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(OUTCOME_ORDER):
            raise ValueError("probabilities must have shape (n_samples, 3) in H/D/A order")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("probabilities must be finite and non-negative")
        totals = values.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError("each probability row must have positive mass")
        return values / totals[:, None]


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10
) -> float:
    """Return mean one-vs-rest expected calibration error across H/D/A."""
    table = calibration_table(labels, probabilities, n_bins=n_bins)
    class_errors: list[float] = []
    for outcome in OUTCOME_ORDER:
        rows = table.loc[table["outcome"] == outcome]
        count = int(rows["count"].sum())
        if count:
            error = np.average(
                np.abs(rows["mean_predicted"] - rows["observed_frequency"]),
                weights=rows["count"],
            )
            class_errors.append(float(error))
    return float(np.mean(class_errors)) if class_errors else float("nan")


__all__ = ["MulticlassPlattCalibrator", "expected_calibration_error"]
