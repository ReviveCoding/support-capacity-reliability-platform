import numpy as np
import pytest

from support_capacity_reliability.forecasting.base import ForecastOutput
from support_capacity_reliability.reliability.calibration import IntervalCalibrator


def test_peak_aware_calibration_expands_upper_tail_from_calibration_only():
    y = np.array([1, 2, 3, 4, 5, 8, 9, 10, 12, 14], dtype=float)
    forecast = ForecastOutput(
        "base",
        q10=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float),
        q50=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
        q90=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float),
    )
    peak_mask = y >= 8
    calibrator = IntervalCalibrator(target_coverage=0.8, peak_upper_coverage=0.9).fit(
        y,
        forecast,
        peak_mask=peak_mask,
    )
    transformed = calibrator.transform(forecast)

    assert calibrator.peak_upper_adjustment_ > 0
    assert np.all(transformed.q90 >= forecast.q90)
    assert np.mean(y[peak_mask] <= transformed.q90[peak_mask]) >= np.mean(
        y[peak_mask] <= forecast.q90[peak_mask]
    )


def test_peak_mask_length_is_validated():
    forecast = ForecastOutput(
        "base",
        q10=np.array([0.0, 1.0]),
        q50=np.array([1.0, 2.0]),
        q90=np.array([2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="peak_mask length"):
        IntervalCalibrator().fit(
            np.array([1.0, 2.0]),
            forecast,
            peak_mask=np.array([True]),
        )


def test_peak_adjustment_is_applied_only_to_predicted_peak_candidates():
    y = np.array([1, 2, 3, 4, 5, 8, 9, 10, 12, 14], dtype=float)
    forecast = ForecastOutput(
        "base",
        q10=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float),
        q50=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
        q90=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float),
    )
    calibrator = IntervalCalibrator().fit(
        y,
        forecast,
        peak_mask=y >= 8,
        peak_activation_threshold=8.0,
    )
    transformed = calibrator.transform(forecast)
    base_half = np.maximum((forecast.q90 - forecast.q10) / 2.0, 1e-6)
    symmetric_q90 = forecast.q50 + base_half * calibrator.expansion_

    non_peak = forecast.q90 < 8.0
    peak = ~non_peak
    assert np.allclose(transformed.q90[non_peak], symmetric_q90[non_peak])
    assert np.all(transformed.q90[peak] >= symmetric_q90[peak])
