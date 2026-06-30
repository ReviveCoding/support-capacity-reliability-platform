import numpy as np

from support_capacity_reliability.config import load_config
from support_capacity_reliability.data.features import build_supervised_frame, temporal_split
from support_capacity_reliability.data.synthetic import generate_synthetic_acd
from support_capacity_reliability.forecasting.models import (
    PoissonForecaster,
    SeasonalNaiveForecaster,
)


def _data():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=5)
    frame, features = build_supervised_frame(
        bundle.intervals,
        config.forecast.target,
        config.forecast.lags,
        config.forecast.rolling_windows,
    )
    return config, features, temporal_split(frame, 0.65, 0.17)


def test_seasonal_quantiles_are_ordered():
    config, features, (train, calibration, _) = _data()
    model = SeasonalNaiveForecaster().fit(train, features, config.forecast.target)
    prediction = model.predict(calibration, features)
    assert np.all(prediction.q10 <= prediction.q50)
    assert np.all(prediction.q50 <= prediction.q90)


def test_poisson_predictions_are_nonnegative():
    config, features, (train, calibration, _) = _data()
    model = PoissonForecaster().fit(train, features, config.forecast.target)
    prediction = model.predict(calibration, features)
    assert np.all(prediction.q10 >= 0)
    assert np.all(prediction.q50 >= 0)
