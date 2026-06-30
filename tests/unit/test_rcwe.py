import numpy as np

from support_capacity_reliability.config import load_config
from support_capacity_reliability.data.features import build_supervised_frame, temporal_split
from support_capacity_reliability.data.synthetic import generate_synthetic_acd
from support_capacity_reliability.forecasting.models import SeasonalNaiveForecaster
from support_capacity_reliability.reliability.rcwe import ReferenceConditionedWorkloadEnvelope


def test_rcwe_returns_support_and_ordered_intervals():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=13)
    frame, features = build_supervised_frame(
        bundle.intervals,
        config.forecast.target,
        config.forecast.lags,
        config.forecast.rolling_windows,
    )
    train, calibration, _ = temporal_split(frame, 0.65, 0.17)
    base = (
        SeasonalNaiveForecaster()
        .fit(train, features, config.forecast.target)
        .predict(calibration, features)
    )
    state_features = [
        column
        for column in features
        if column.startswith("lag_") or column.startswith("roll_mean_")
    ][:8]
    rcwe = ReferenceConditionedWorkloadEnvelope(neighbors=6).fit(
        train, state_features, config.forecast.target
    )
    result = rcwe.transform(calibration, base)
    assert len(result.support) == len(calibration)
    assert np.all((result.support >= 0) & (result.support <= 1))
    assert np.all(result.forecast.q10 <= result.forecast.q50)
    assert np.all(result.forecast.q50 <= result.forecast.q90)


def test_rcwe_support_is_density_calibrated_and_detects_novel_states():
    import pandas as pd

    from support_capacity_reliability.forecasting.base import ForecastOutput

    rng = np.random.default_rng(123)
    train = pd.DataFrame(
        {
            "region": ["north"] * 80,
            "skill": ["billing"] * 80,
            "state_a": rng.normal(0.0, 0.25, 80),
            "state_b": rng.normal(0.0, 0.25, 80),
            "target": rng.poisson(5.0, 80).astype(float),
        }
    )
    query = pd.DataFrame(
        {
            "region": ["north", "north"],
            "skill": ["billing", "billing"],
            "state_a": [0.05, 8.0],
            "state_b": [-0.05, 8.0],
        }
    )
    base = ForecastOutput(
        "base",
        q10=np.array([3.0, 3.0]),
        q50=np.array([5.0, 5.0]),
        q90=np.array([7.0, 7.0]),
    )
    result = (
        ReferenceConditionedWorkloadEnvelope(
            neighbors=8,
            minimum_support=0.2,
        )
        .fit(train, ["state_a", "state_b"], "target")
        .transform(query, base)
    )

    assert result.support[0] > 0.2
    assert result.support[0] > result.support[1]
    assert not result.low_support[0]
    assert result.low_support[1]
