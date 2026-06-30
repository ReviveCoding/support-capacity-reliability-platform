from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class ForecastOutput:
    model_name: str
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame({"q10": self.q10, "q50": self.q50, "q90": self.q90}, index=index)


class ForecastModel(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> ForecastModel: ...

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput: ...


def residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.quantile(residuals, 0.1)), float(np.quantile(residuals, 0.9))


def enforce_quantile_order(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.stack([q10, q50, q90], axis=1)
    ordered = np.sort(stacked, axis=1)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2]
