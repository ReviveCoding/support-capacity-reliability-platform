from __future__ import annotations

import numpy as np
import pandas as pd

from support_capacity_reliability.forecasting.base import ForecastOutput


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / max(denominator, 1e-9))


def pinball(y_true: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = y_true - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def forecast_metrics(y_true: np.ndarray, forecast: ForecastOutput) -> dict[str, float]:
    """Evaluate point forecasts and the nominal 80% prediction interval."""
    observed_values = np.asarray(y_true, dtype=float)
    interval_coverage_rate = interval_coverage(
        observed_values, forecast.q10, forecast.q90
    )
    return {
        "mae": float(np.mean(np.abs(observed_values - forecast.q50))),
        "rmse": float(np.sqrt(np.mean((observed_values - forecast.q50) ** 2))),
        "wape": wape(observed_values, forecast.q50),
        "signed_bias": float(np.mean(forecast.q50 - observed_values)),
        "pinball_q10": pinball(observed_values, forecast.q10, 0.1),
        "pinball_q50": pinball(observed_values, forecast.q50, 0.5),
        "pinball_q90": pinball(observed_values, forecast.q90, 0.9),
        "interval_coverage_80": interval_coverage_rate,
        "coverage_error": abs(interval_coverage_rate - 0.8),
        "mean_interval_width": float(np.mean(forecast.q90 - forecast.q10)),
    }


def slice_metrics(frame: pd.DataFrame, target: str, forecast: ForecastOutput) -> pd.DataFrame:
    working = frame[["regime", "region", "skill"]].copy()
    working["target"] = frame[target].to_numpy(float)
    working["prediction"] = forecast.q50
    rows: list[dict[str, object]] = []
    for dimension in ["regime", "region", "skill"]:
        for value, group in working.groupby(dimension):
            rows.append(
                {
                    "dimension": dimension,
                    "slice": str(value),
                    "count": len(group),
                    "wape": wape(
                        group["target"].to_numpy(float), group["prediction"].to_numpy(float)
                    ),
                    "mae": float(np.mean(np.abs(group["target"] - group["prediction"]))),
                }
            )
    return pd.DataFrame(rows)
