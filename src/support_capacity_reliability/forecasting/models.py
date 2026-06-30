from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from support_capacity_reliability.forecasting.base import (
    ForecastOutput,
    enforce_quantile_order,
    residual_quantiles,
)


class SeasonalNaiveForecaster:
    name = "seasonal"

    def __init__(self, seasonal_lag: int = 48) -> None:
        self.seasonal_lag = seasonal_lag
        self.fallback_: float = 0.0
        self.residual_low_: float = -1.0
        self.residual_high_: float = 1.0
        self.target_: str | None = None

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> SeasonalNaiveForecaster:
        del features
        self.target_ = target
        self.fallback_ = float(frame[target].median())
        lag_column = f"lag_{self.seasonal_lag}"
        point = (
            frame[lag_column].to_numpy(float)
            if lag_column in frame
            else np.full(len(frame), self.fallback_)
        )
        point = np.nan_to_num(point, nan=self.fallback_)
        self.residual_low_, self.residual_high_ = residual_quantiles(
            frame[target].to_numpy(float), point
        )
        return self

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput:
        del features
        lag_column = f"lag_{self.seasonal_lag}"
        point = (
            frame[lag_column].to_numpy(float)
            if lag_column in frame
            else np.full(len(frame), self.fallback_)
        )
        point = np.clip(np.nan_to_num(point, nan=self.fallback_), 0, None)
        q10 = np.clip(point + self.residual_low_, 0, None)
        q90 = np.clip(point + self.residual_high_, 0, None)
        q10, point, q90 = enforce_quantile_order(q10, point, q90)
        return ForecastOutput(self.name, q10, point, q90)


class PoissonForecaster:
    name = "poisson"

    def __init__(self, alpha: float = 0.1, max_iter: int = 500) -> None:
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=alpha, max_iter=max_iter)),
            ]
        )
        self.residual_low_ = -1.0
        self.residual_high_ = 1.0

    def fit(self, frame: pd.DataFrame, features: list[str], target: str) -> PoissonForecaster:
        x = frame[features]
        y = np.clip(frame[target].to_numpy(float), 0, None)
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            self.pipeline.fit(x, y)
        point = np.clip(self.pipeline.predict(x), 0, None)
        self.residual_low_, self.residual_high_ = residual_quantiles(y, point)
        return self

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput:
        point = np.clip(self.pipeline.predict(frame[features]), 0, None)
        q10 = np.clip(point + self.residual_low_, 0, None)
        q90 = np.clip(point + self.residual_high_, 0, None)
        q10, point, q90 = enforce_quantile_order(q10, point, q90)
        return ForecastOutput(self.name, q10, point, q90)


class LightGBMQuantileForecaster:
    name = "lightgbm"

    def __init__(self, seed: int = 42) -> None:
        common = dict(
            n_estimators=180,
            learning_rate=0.045,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.2,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        self.models = {
            0.1: LGBMRegressor(objective="quantile", alpha=0.1, **common),
            0.5: LGBMRegressor(objective="quantile", alpha=0.5, **common),
            0.9: LGBMRegressor(objective="quantile", alpha=0.9, **common),
        }

    def fit(
        self, frame: pd.DataFrame, features: list[str], target: str
    ) -> LightGBMQuantileForecaster:
        x = frame[features]
        y = np.clip(frame[target].to_numpy(float), 0, None)
        for model in self.models.values():
            model.fit(x, y)
        return self

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput:
        x = frame[features]
        q10 = np.clip(self.models[0.1].predict(x), 0, None)
        q50 = np.clip(self.models[0.5].predict(x), 0, None)
        q90 = np.clip(self.models[0.9].predict(x), 0, None)
        q10, q50, q90 = enforce_quantile_order(q10, q50, q90)
        return ForecastOutput(self.name, q10, q50, q90)
