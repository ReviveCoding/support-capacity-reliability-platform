from support_capacity_reliability.config import load_config
from support_capacity_reliability.data.features import (
    OPERATIONAL_COLUMNS,
    build_recursive_feature_rows,
    build_supervised_frame,
    temporal_split,
)
from support_capacity_reliability.data.synthetic import generate_synthetic_acd


def test_feature_builder_is_leakage_oriented():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=2)
    frame, features = build_supervised_frame(
        bundle.intervals,
        config.forecast.target,
        config.forecast.lags,
        config.forecast.rolling_windows,
    )
    assert "lag_1" in features
    assert config.forecast.target not in features
    assert not frame[features].isna().any().any()


def test_temporal_split_is_ordered():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=2)
    frame, _ = build_supervised_frame(
        bundle.intervals,
        config.forecast.target,
        config.forecast.lags,
        config.forecast.rolling_windows,
    )
    train, calibration, test = temporal_split(frame, 0.65, 0.17)
    assert train["timestamp"].max() < calibration["timestamp"].min()
    assert calibration["timestamp"].max() < test["timestamp"].min()


def test_current_operational_values_and_regime_are_not_predictors():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=3)
    _, features = build_supervised_frame(
        bundle.intervals,
        config.forecast.target,
        config.forecast.lags,
        config.forecast.rolling_windows,
    )
    for column in OPERATIONAL_COLUMNS:
        assert column not in features
    assert not any(column.startswith("regime_") for column in features)


def test_recursive_features_use_history_and_planning_assumptions_only():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=4)
    history = bundle.intervals.iloc[: -len(config.data.regions) * len(config.data.skills)].copy()
    timestamp = bundle.intervals["timestamp"].max()
    planning = history.groupby(["region", "skill"], as_index=False)[OPERATIONAL_COLUMNS].median()
    rows, features = build_recursive_feature_rows(
        history=history,
        timestamp=timestamp,
        target=config.forecast.target,
        lags=config.forecast.lags,
        rolling_windows=config.forecast.rolling_windows,
        regions=config.data.regions,
        skills=config.data.skills,
        planning_parameters=planning,
    )
    assert len(rows) == len(config.data.regions) * len(config.data.skills)
    assert not rows[features].isna().any().any()
    for column in OPERATIONAL_COLUMNS:
        assert column not in features
