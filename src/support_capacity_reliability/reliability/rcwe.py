from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from support_capacity_reliability.forecasting.base import ForecastOutput, enforce_quantile_order


@dataclass
class RCWEOutput:
    forecast: ForecastOutput
    reference_mean: np.ndarray
    reference_std: np.ndarray
    support: np.ndarray
    low_support: np.ndarray


class ReferenceConditionedWorkloadEnvelope:
    """Leakage-safe reference-state forecast correction.

    Reference candidates are drawn only from the frame passed to ``fit``. The caller is responsible
    for fitting on a historical partition that ends before the forecast origin. A NumPy/SciPy batched
    distance implementation is used to keep behavior deterministic across CPU environments.
    """

    def __init__(
        self,
        neighbors: int = 12,
        blend_strength: float = 0.45,
        minimum_support: float = 0.20,
        distance_temperature: float = 1.0,
        low_support_interval_inflation: float = 1.30,
    ) -> None:
        self.neighbors = neighbors
        self.blend_strength = blend_strength
        self.minimum_support = minimum_support
        self.distance_temperature = distance_temperature
        self.low_support_interval_inflation = low_support_interval_inflation
        self.scaler = StandardScaler()
        self.states: dict[tuple[str, str], np.ndarray] = {}
        self.targets: dict[tuple[str, str], np.ndarray] = {}
        self.support_scales: dict[tuple[str, str], float] = {}
        self.state_features: list[str] = []
        self.global_target: float = 0.0

    def fit(
        self, frame: pd.DataFrame, state_features: list[str], target: str
    ) -> ReferenceConditionedWorkloadEnvelope:
        self.state_features = list(state_features)
        matrix = frame[self.state_features].to_numpy(float)
        self.scaler.fit(matrix)
        scaled = self.scaler.transform(matrix)
        self.global_target = float(frame[target].median())
        working = frame[["region", "skill", target]].copy()
        working["_row"] = np.arange(len(working))
        for key, group in working.groupby(["region", "skill"], sort=False):
            row_ids = group["_row"].to_numpy(int)
            if len(row_ids) < 2:
                continue
            group_states = scaled[row_ids]
            self.states[key] = group_states
            self.targets[key] = group[target].to_numpy(float)

            # Calibrate support relative to the density of each historical region-skill
            # state library. Raw Euclidean distances grow with feature dimension, so a
            # fixed exp(-distance) score makes nearly every realistic query look OOD.
            # The robust 90th percentile of leakage-safe leave-one-out neighbor
            # distances defines the familiar-state scale for this group.
            sample_count = min(len(group_states), 512)
            sample_ids = np.linspace(0, len(group_states) - 1, sample_count, dtype=int)
            calibration_distances = cdist(group_states[sample_ids], group_states)
            calibration_distances[np.arange(sample_count), sample_ids] = np.inf
            support_neighbors = min(self.neighbors, len(group_states) - 1)
            nearest = np.partition(calibration_distances, kth=support_neighbors - 1, axis=1)[
                :, :support_neighbors
            ]
            mean_neighbor_distance = nearest.mean(axis=1)
            finite = mean_neighbor_distance[np.isfinite(mean_neighbor_distance)]
            scale = float(np.quantile(finite, 0.90)) if len(finite) else 1.0
            self.support_scales[key] = max(scale, 1e-6)
        return self

    def transform(self, frame: pd.DataFrame, base: ForecastOutput) -> RCWEOutput:
        if not self.states:
            raise RuntimeError("RCWE must be fitted before transform")
        scaled = self.scaler.transform(frame[self.state_features].to_numpy(float))
        reference_mean = np.full(len(frame), self.global_target, dtype=float)
        reference_std = np.full(len(frame), max(self.global_target * 0.2, 1.0), dtype=float)
        support = np.zeros(len(frame), dtype=float)

        key_frame = pd.DataFrame(
            {
                "region": frame["region"].astype(str).to_numpy(),
                "skill": frame["skill"].astype(str).to_numpy(),
                "row_id": np.arange(len(frame)),
            }
        )
        for key, group in key_frame.groupby(["region", "skill"], sort=False):
            reference_states = self.states.get(key)
            if reference_states is None:
                continue
            row_ids = group["row_id"].to_numpy(int)
            query_states = scaled[row_ids]
            distances_all = cdist(query_states, reference_states, metric="euclidean")
            k = min(self.neighbors, reference_states.shape[0])
            neighbor_ids = np.argpartition(distances_all, kth=k - 1, axis=1)[:, :k]
            distances = np.take_along_axis(distances_all, neighbor_ids, axis=1)
            order = np.argsort(distances, axis=1)
            neighbor_ids = np.take_along_axis(neighbor_ids, order, axis=1)
            distances = np.take_along_axis(distances, order, axis=1)
            weights = np.exp(-distances / self.distance_temperature)
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            target_values = self.targets[key][neighbor_ids]
            means = np.sum(weights * target_values, axis=1)
            variances = np.sum(weights * (target_values - means[:, None]) ** 2, axis=1)
            reference_mean[row_ids] = means
            reference_std[row_ids] = np.maximum(np.sqrt(variances), 0.5)
            support_scale = self.support_scales.get(key, 1.0)
            support[row_ids] = np.exp(
                -distances.mean(axis=1) / max(support_scale * self.distance_temperature, 1e-12)
            )

        effective_strength = self.blend_strength * support
        corrected_median = base.q50 + effective_strength * (reference_mean - base.q50)
        base_half_width = np.maximum((base.q90 - base.q10) / 2.0, 0.5)
        reference_half_width = 1.2816 * reference_std
        half_width = (
            1 - effective_strength
        ) * base_half_width + effective_strength * reference_half_width
        low_support = support < self.minimum_support
        half_width = np.where(
            low_support, half_width * self.low_support_interval_inflation, half_width
        )
        q10 = np.clip(corrected_median - half_width, 0, None)
        q90 = np.clip(corrected_median + half_width, 0, None)
        q10, corrected_median, q90 = enforce_quantile_order(
            q10, np.clip(corrected_median, 0, None), q90
        )
        forecast = ForecastOutput(f"{base.model_name}+rcwe", q10, corrected_median, q90)
        return RCWEOutput(forecast, reference_mean, reference_std, support, low_support)
