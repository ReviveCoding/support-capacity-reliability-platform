from __future__ import annotations

from support_capacity_reliability.config import AppConfig
from support_capacity_reliability.forecasting.chronos_adapter import Chronos2Adapter
from support_capacity_reliability.forecasting.models import (
    LightGBMQuantileForecaster,
    PoissonForecaster,
    SeasonalNaiveForecaster,
)


def build_model(name: str, config: AppConfig):
    if name == "seasonal":
        return SeasonalNaiveForecaster(seasonal_lag=48)
    if name == "poisson":
        return PoissonForecaster()
    if name == "lightgbm":
        return LightGBMQuantileForecaster(seed=config.project.seed)
    if name == "torch_quantile":
        try:
            from support_capacity_reliability.forecasting.torch_model import TorchQuantileForecaster
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "torch_quantile requires the optional torch extra; "
                    "install with support-capacity-reliability[torch]"
                ) from exc
            raise
        settings = config.forecast.torch
        return TorchQuantileForecaster(
            epochs=settings.epochs,
            hidden_dim=settings.hidden_dim,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            device=settings.device,
            seed=config.project.seed,
        )
    if name == "chronos2":
        return Chronos2Adapter(
            config.forecast.chronos.model_id,
            context_length=config.forecast.chronos.context_length,
        )
    raise KeyError(f"Unknown forecast model: {name}")
