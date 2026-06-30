from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from support_capacity_reliability.forecasting.base import ForecastOutput


@dataclass
class ScenarioBundle:
    leaf_samples: np.ndarray
    leaf_keys: list[str]
    aggregate_samples: pd.DataFrame
    temporal_autocorrelation: float
    cross_sectional_correlation: np.ndarray
    operational_skill_keys: list[str]
    aht_multipliers: np.ndarray
    patience_multipliers: np.ndarray
    shrinkage_rates: np.ndarray
    operational_diagnostics: dict[str, dict[str, float]]


def _safe_correlation(matrix: np.ndarray, shrinkage: float) -> np.ndarray:
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return np.eye(matrix.shape[1])
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (1 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(corrected))
    corrected = corrected / np.outer(diagonal, diagonal)
    return corrected


def _estimate_temporal_autocorrelation(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 4:
        return 0.0
    estimates: list[float] = []
    for column in range(matrix.shape[1]):
        previous = matrix[:-1, column]
        current = matrix[1:, column]
        if np.std(previous) < 1e-9 or np.std(current) < 1e-9:
            continue
        correlation = float(np.corrcoef(previous, current)[0, 1])
        if np.isfinite(correlation):
            estimates.append(correlation)
    if not estimates:
        return 0.0
    return float(np.clip(np.median(estimates), -0.80, 0.80))


def _safe_pair_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 4 or np.std(left[mask]) < 1e-9 or np.std(right[mask]) < 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(left[mask], right[mask])[0, 1], -0.90, 0.90))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _operational_scenarios(
    *,
    working: pd.DataFrame,
    standardized_load_shocks: np.ndarray,
    leaf_keys: list[str],
    skills: list[str],
    scenario_count: int,
    horizon_times: list[pd.Timestamp],
    rng: np.random.Generator,
    operational_history: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Generate AHT, patience, and shrinkage scenarios correlated with load shocks.

    The calibration history estimates each operational parameter's dispersion and its
    relationship with load residuals. The generated paths are scenario-correlated with the
    arrival trajectories but remain bounded to operationally plausible ranges.
    """
    shape = (scenario_count, len(horizon_times), len(skills))
    aht_multipliers = np.ones(shape, dtype=float)
    patience_multipliers = np.ones(shape, dtype=float)
    shrinkage_rates = np.zeros(shape, dtype=float)
    diagnostics: dict[str, dict[str, float]] = {}

    for skill_idx, skill in enumerate(skills):
        leaf_ids = [idx for idx, key in enumerate(leaf_keys) if key.split("::", 1)[1] == skill]
        skill_load = standardized_load_shocks[:, :, leaf_ids].mean(axis=2)
        skill_load = (skill_load - skill_load.mean()) / max(float(skill_load.std()), 1e-9)

        horizon_skill = working[working["skill"].astype(str) == skill]
        base_shrinkage_by_time = (
            horizon_skill.groupby("timestamp")["planning_shrinkage_rate"]
            .mean()
            .reindex(horizon_times)
            .fillna(float(horizon_skill["planning_shrinkage_rate"].median()))
            .to_numpy(float)
        )

        std_aht = 0.10
        std_patience = 0.08
        std_shrinkage_logit = 0.25
        rho_aht = rho_patience = rho_shrinkage = 0.0
        history_rows = 0
        if operational_history is not None and not operational_history.empty:
            history_skill = operational_history[
                operational_history["skill"].astype(str) == skill
            ].copy()
            if not history_skill.empty:
                aggregated = (
                    history_skill.groupby("timestamp", as_index=False)
                    .agg(
                        residual=("residual", "mean"),
                        average_handle_time_seconds=("average_handle_time_seconds", "mean"),
                        patience_mean_seconds=("patience_mean_seconds", "mean"),
                        shrinkage_rate=("shrinkage_rate", "mean"),
                    )
                    .sort_values("timestamp")
                )
                history_rows = len(aggregated)
                load_values = aggregated["residual"].to_numpy(float)
                aht_values = aggregated["average_handle_time_seconds"].to_numpy(float)
                patience_values = aggregated["patience_mean_seconds"].to_numpy(float)
                shrinkage_values = aggregated["shrinkage_rate"].to_numpy(float)
                aht_log = np.log(np.maximum(aht_values, 1.0) / np.median(aht_values))
                patience_log = np.log(np.maximum(patience_values, 1.0) / np.median(patience_values))
                shrinkage_logit = _logit(shrinkage_values)
                shrinkage_logit -= float(np.mean(shrinkage_logit))
                std_aht = float(np.clip(np.std(aht_log, ddof=1), 0.02, 0.35))
                std_patience = float(np.clip(np.std(patience_log, ddof=1), 0.02, 0.35))
                std_shrinkage_logit = float(np.clip(np.std(shrinkage_logit, ddof=1), 0.05, 0.80))
                rho_aht = _safe_pair_correlation(load_values, aht_log)
                rho_patience = _safe_pair_correlation(load_values, patience_log)
                rho_shrinkage = _safe_pair_correlation(load_values, shrinkage_logit)

        def correlated_shock(rho: float, load_shock: np.ndarray = skill_load) -> np.ndarray:
            noise = rng.normal(size=load_shock.shape)
            return rho * load_shock + np.sqrt(max(1.0 - rho**2, 1e-9)) * noise

        aht_multipliers[:, :, skill_idx] = np.clip(
            np.exp(std_aht * correlated_shock(rho_aht)), 0.65, 1.85
        )
        patience_multipliers[:, :, skill_idx] = np.clip(
            np.exp(std_patience * correlated_shock(rho_patience)), 0.55, 1.70
        )
        base_logit = _logit(base_shrinkage_by_time)[None, :]
        shrinkage_rates[:, :, skill_idx] = np.clip(
            _expit(base_logit + std_shrinkage_logit * correlated_shock(rho_shrinkage)),
            0.01,
            0.55,
        )
        diagnostics[skill] = {
            "history_rows": float(history_rows),
            "aht_log_std": std_aht,
            "patience_log_std": std_patience,
            "shrinkage_logit_std": std_shrinkage_logit,
            "load_aht_correlation": rho_aht,
            "load_patience_correlation": rho_patience,
            "load_shrinkage_correlation": rho_shrinkage,
        }

    return aht_multipliers, patience_multipliers, shrinkage_rates, diagnostics


def generate_coherent_scenarios(
    frame: pd.DataFrame,
    forecast: ForecastOutput,
    scenario_count: int,
    seed: int,
    residual_history: pd.DataFrame | None = None,
    operational_history: pd.DataFrame | None = None,
    correlation_shrinkage: float = 0.15,
) -> ScenarioBundle:
    """Generate coherent load paths and correlated operational-parameter scenarios."""
    required_columns = ["timestamp", "region", "skill"]
    optional_columns = [
        "planning_average_handle_time_seconds",
        "planning_patience_mean_seconds",
        "planning_shrinkage_rate",
    ]
    working = frame[
        [*required_columns, *[c for c in optional_columns if c in frame.columns]]
    ].copy()
    for column, default in [
        ("planning_average_handle_time_seconds", 540.0),
        ("planning_patience_mean_seconds", 270.0),
        ("planning_shrinkage_rate", 0.12),
    ]:
        if column not in working:
            working[column] = default
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["q10"] = forecast.q10
    working["q50"] = forecast.q50
    working["q90"] = forecast.q90
    working["leaf_key"] = working["region"].astype(str) + "::" + working["skill"].astype(str)
    leaf_keys = sorted(working["leaf_key"].unique())
    leaf_index = {key: idx for idx, key in enumerate(leaf_keys)}
    horizon_times = [pd.Timestamp(value) for value in sorted(working["timestamp"].unique())]
    rng = np.random.default_rng(seed)

    temporal_autocorrelation = 0.0
    if residual_history is not None and not residual_history.empty:
        pivot = residual_history.pivot_table(
            index="timestamp",
            columns="leaf_key",
            values="residual",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=leaf_keys).fillna(0.0).sort_index()
        residual_matrix = pivot.to_numpy(float)
        correlation = _safe_correlation(residual_matrix, correlation_shrinkage)
        temporal_autocorrelation = _estimate_temporal_autocorrelation(residual_matrix)
    else:
        correlation = np.eye(len(leaf_keys))

    innovations = np.stack(
        [
            rng.multivariate_normal(
                np.zeros(len(leaf_keys)),
                correlation,
                size=scenario_count,
            )
            for _ in horizon_times
        ],
        axis=1,
    )
    standardized = np.zeros_like(innovations)
    innovation_scale = float(np.sqrt(max(1.0 - temporal_autocorrelation**2, 1e-9)))
    standardized[:, 0, :] = innovations[:, 0, :]
    for time_idx in range(1, len(horizon_times)):
        standardized[:, time_idx, :] = (
            temporal_autocorrelation * standardized[:, time_idx - 1, :]
            + innovation_scale * innovations[:, time_idx, :]
        )

    samples = np.zeros((scenario_count, len(horizon_times), len(leaf_keys)), dtype=float)
    aggregate_rows: list[dict[str, object]] = []
    regions = sorted(working["region"].astype(str).unique())
    skills = sorted(working["skill"].astype(str).unique())
    region_leaf_ids = {
        region: [index for index, key in enumerate(leaf_keys) if key.split("::", 1)[0] == region]
        for region in regions
    }
    skill_leaf_ids = {
        skill: [index for index, key in enumerate(leaf_keys) if key.split("::", 1)[1] == skill]
        for skill in skills
    }

    for time_idx, timestamp in enumerate(horizon_times):
        chunk = working[working["timestamp"] == timestamp]
        median = np.zeros(len(leaf_keys), dtype=float)
        sigma = np.full(len(leaf_keys), 0.5, dtype=float)
        for row in chunk.itertuples(index=False):
            idx = leaf_index[row.leaf_key]
            median[idx] = float(row.q50)
            sigma[idx] = max(float(row.q90 - row.q10) / (2 * 1.2816), 0.25)
        values = np.clip(
            median[None, :] + standardized[:, time_idx, :] * sigma[None, :],
            0,
            None,
        )
        samples[:, time_idx, :] = values
        for scenario_id in range(scenario_count):
            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "timestamp": timestamp,
                "global_total": float(values[scenario_id].sum()),
            }
            for region, leaf_ids in region_leaf_ids.items():
                row[f"region::{region}"] = float(values[scenario_id, leaf_ids].sum())
            for skill, leaf_ids in skill_leaf_ids.items():
                row[f"skill::{skill}"] = float(values[scenario_id, leaf_ids].sum())
            aggregate_rows.append(row)

    aht_multipliers, patience_multipliers, shrinkage_rates, operational_diagnostics = (
        _operational_scenarios(
            working=working,
            standardized_load_shocks=standardized,
            leaf_keys=leaf_keys,
            skills=skills,
            scenario_count=scenario_count,
            horizon_times=horizon_times,
            rng=rng,
            operational_history=operational_history,
        )
    )

    return ScenarioBundle(
        leaf_samples=samples,
        leaf_keys=leaf_keys,
        aggregate_samples=(
            pd.DataFrame(aggregate_rows)
            .sort_values(["scenario_id", "timestamp"])
            .reset_index(drop=True)
        ),
        temporal_autocorrelation=temporal_autocorrelation,
        cross_sectional_correlation=correlation,
        operational_skill_keys=skills,
        aht_multipliers=aht_multipliers,
        patience_multipliers=patience_multipliers,
        shrinkage_rates=shrinkage_rates,
        operational_diagnostics=operational_diagnostics,
    )
