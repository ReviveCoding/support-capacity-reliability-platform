from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from support_capacity_reliability.forecasting.base import ForecastOutput


@dataclass
class ChronosAvailability:
    available: bool
    reason: str
    package_version: str | None = None


class Chronos2Adapter:
    """Optional official Chronos-2 zero-shot adapter.

    The adapter requires a dense historical context represented by ``lag_1`` through
    ``lag_<context_length>``. Values are passed oldest-to-newest. Smoke mode never downloads
    model weights; full mode can enable the official package and cached model.
    """

    name = "chronos2"

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device_map: str = "auto",
        context_length: int = 96,
    ) -> None:
        if context_length < 8:
            raise ValueError("Chronos context_length must be at least 8")
        self.model_id = model_id
        self.device_map = device_map
        self.context_length = int(context_length)
        self.pipeline: Any | None = None

    @staticmethod
    def availability() -> ChronosAvailability:
        try:
            import importlib.metadata

            package_version = importlib.metadata.version("chronos-forecasting")
            from chronos import BaseChronosPipeline  # noqa: F401

            return ChronosAvailability(True, "available", package_version)
        except Exception as exc:  # optional dependency and model API may vary
            return ChronosAvailability(False, f"{type(exc).__name__}: {exc}")

    def fit(
        self,
        frame: pd.DataFrame | None,
        features: list[str],
        target: str,
    ) -> Chronos2Adapter:
        del frame, features, target
        status = self.availability()
        if not status.available:
            raise RuntimeError(
                "Chronos-2 is optional. Install `.[chronos]`, cache the official model, "
                f"and retry. Availability check: {status.reason}"
            )
        from chronos import BaseChronosPipeline

        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device_map,
        )
        return self

    def _contexts(self, frame: pd.DataFrame) -> list[np.ndarray]:
        columns = [f"lag_{lag}" for lag in range(self.context_length, 0, -1)]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                "Chronos-2 requires a dense lag context. Missing columns include: " + preview
            )
        matrix = frame[columns].to_numpy(dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Chronos context contains non-finite values")
        return [row.copy() for row in matrix]

    @staticmethod
    def _normalize_quantile_shape(raw: Any, batch_size: int) -> np.ndarray:
        quantile_tensor = raw[0] if isinstance(raw, tuple) else raw
        if hasattr(quantile_tensor, "detach"):
            quantile_tensor = quantile_tensor.detach().cpu().numpy()
        values = np.asarray(quantile_tensor, dtype=float)
        # Official pipelines commonly return [batch, prediction_length, quantile] or
        # [batch, quantile, prediction_length]. Only a one-step prediction is requested.
        if values.ndim == 3 and values.shape == (batch_size, 1, 3):
            return values[:, 0, :]
        if values.ndim == 3 and values.shape == (batch_size, 3, 1):
            return values[:, :, 0]
        if values.ndim == 2 and values.shape == (batch_size, 3):
            return values
        raise ValueError(f"Unexpected Chronos quantile output shape: {values.shape}")

    def predict(self, frame: pd.DataFrame, features: list[str]) -> ForecastOutput:
        del features
        if self.pipeline is None:
            raise RuntimeError("Chronos adapter is not initialized")
        contexts = self._contexts(frame)
        raw = self.pipeline.predict_quantiles(
            contexts,
            prediction_length=1,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        quantiles = self._normalize_quantile_shape(raw, len(frame))
        quantiles = np.maximum.accumulate(np.clip(quantiles, 0.0, None), axis=1)
        return ForecastOutput(
            self.name,
            quantiles[:, 0],
            quantiles[:, 1],
            quantiles[:, 2],
        )
