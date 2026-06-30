from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from support_capacity_reliability.forecasting.base import ForecastOutput


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    epsilon = 1e-6
    ref_share = np.maximum(ref_counts / max(ref_counts.sum(), 1), epsilon)
    cur_share = np.maximum(cur_counts / max(cur_counts.sum(), 1), epsilon)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _severity(psi: float) -> str:
    if not np.isfinite(psi):
        return "UNKNOWN"
    if psi >= 0.25:
        return "ALERT"
    if psi >= 0.10:
        return "WARN"
    return "OK"


def build_monitoring_snapshot(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    operational_columns: list[str],
    forecast: ForecastOutput,
    rcwe_support: np.ndarray,
    rcwe_low_support: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    columns = [target, *operational_columns]
    for column in columns:
        for partition_name, partition in [("calibration", calibration), ("test", test)]:
            psi = population_stability_index(
                train[column].to_numpy(float), partition[column].to_numpy(float)
            )
            rows.append(
                {
                    "metric": "psi",
                    "field": column,
                    "partition": partition_name,
                    "value": psi,
                    "status": _severity(psi),
                }
            )

    y = test[target].to_numpy(float)
    residual = y - np.asarray(forecast.q50, dtype=float)
    rows.extend(
        [
            {
                "metric": "forecast_signed_bias",
                "field": target,
                "partition": "test",
                "value": float(np.mean(residual)),
                "status": "INFO",
            },
            {
                "metric": "interval_coverage_80",
                "field": target,
                "partition": "test",
                "value": float(np.mean((y >= forecast.q10) & (y <= forecast.q90))),
                "status": "INFO",
            },
            {
                "metric": "mean_rcwe_support",
                "field": "rcwe_support",
                "partition": "test",
                "value": float(np.mean(rcwe_support)),
                "status": "INFO",
            },
            {
                "metric": "low_rcwe_support_rate",
                "field": "rcwe_support",
                "partition": "test",
                "value": float(np.mean(rcwe_low_support)),
                "status": "WARN" if float(np.mean(rcwe_low_support)) > 0.20 else "OK",
            },
        ]
    )
    frame = pd.DataFrame(rows)
    status_counts = frame["status"].value_counts().to_dict()
    snapshot = {
        "schema_version": "1.0",
        "status": "ALERT"
        if status_counts.get("ALERT", 0)
        else "WARN"
        if status_counts.get("WARN", 0)
        else "OK",
        "thresholds": {
            "psi_warn": 0.10,
            "psi_alert": 0.25,
            "low_rcwe_support_warn": 0.20,
        },
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "maximum_test_psi": float(
            frame.loc[(frame["metric"] == "psi") & (frame["partition"] == "test"), "value"].max()
        ),
        "forecast_signed_bias": float(np.mean(residual)),
        "interval_coverage_80": float(np.mean((y >= forecast.q10) & (y <= forecast.q90))),
        "mean_rcwe_support": float(np.mean(rcwe_support)),
        "low_rcwe_support_rate": float(np.mean(rcwe_low_support)),
        "interpretation": (
            "Monitoring diagnostics are warning signals, not causal evidence or automatic release blockers."
        ),
    }
    return snapshot, frame
