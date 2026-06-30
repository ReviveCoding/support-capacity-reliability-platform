from __future__ import annotations

import numpy as np

from support_capacity_reliability.forecasting.base import ForecastOutput, enforce_quantile_order


class IntervalCalibrator:
    def __init__(
        self,
        target_coverage: float = 0.8,
        peak_upper_coverage: float = 0.9,
    ) -> None:
        self.target_coverage = target_coverage
        self.peak_upper_coverage = peak_upper_coverage
        self.expansion_: float = 1.0
        self.peak_upper_adjustment_: float = 0.0
        self.peak_activation_threshold_: float | None = None

    def fit(
        self,
        y_true: np.ndarray,
        forecast: ForecastOutput,
        peak_mask: np.ndarray | None = None,
        peak_activation_threshold: float | None = None,
    ) -> IntervalCalibrator:
        y = np.asarray(y_true, dtype=float)
        median = np.asarray(forecast.q50, dtype=float)
        base_half = np.maximum((forecast.q90 - forecast.q10) / 2.0, 1e-6)
        normalized_error = np.abs(y - median) / base_half
        self.expansion_ = max(0.25, float(np.quantile(normalized_error, self.target_coverage)))
        if peak_activation_threshold is not None:
            self.peak_activation_threshold_ = float(peak_activation_threshold)
        if peak_mask is not None:
            peak = np.asarray(peak_mask, dtype=bool)
            if len(peak) != len(y):
                raise ValueError("peak_mask length must match y_true")
            if int(peak.sum()) >= 5:
                upper_residual = y[peak] - np.asarray(forecast.q90, dtype=float)[peak]
                self.peak_upper_adjustment_ = max(
                    0.0,
                    float(np.quantile(upper_residual, self.peak_upper_coverage)),
                )
        return self

    def transform(self, forecast: ForecastOutput) -> ForecastOutput:
        half = np.maximum((forecast.q90 - forecast.q10) / 2.0, 1e-6) * self.expansion_
        q10 = np.clip(forecast.q50 - half, 0, None)
        upper_adjustment = np.zeros_like(np.asarray(forecast.q90, dtype=float))
        if self.peak_upper_adjustment_ > 0:
            if self.peak_activation_threshold_ is None:
                upper_adjustment[:] = self.peak_upper_adjustment_
            else:
                peak_candidate = (
                    np.asarray(forecast.q90, dtype=float) >= self.peak_activation_threshold_
                )
                upper_adjustment[peak_candidate] = self.peak_upper_adjustment_
        q90 = np.clip(
            forecast.q50 + half + upper_adjustment,
            0,
            None,
        )
        q10, q50, q90 = enforce_quantile_order(q10, forecast.q50, q90)
        return ForecastOutput(f"{forecast.model_name}+calibrated", q10, q50, q90)
