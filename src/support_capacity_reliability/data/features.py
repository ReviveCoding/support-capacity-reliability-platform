from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

ID_COLUMNS = ["timestamp", "region", "skill", "regime"]
OPERATIONAL_COLUMNS = [
    "average_handle_time_seconds",
    "patience_mean_seconds",
    "shrinkage_rate",
]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only forecast-origin-available calendar features.

    Regime labels are intentionally excluded. In the synthetic generator they describe the
    realized future state and therefore are evaluation slices, not deployable forecast inputs.
    """
    out = frame.copy()
    timestamp = pd.to_datetime(out["timestamp"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    dow = timestamp.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def _operational_feature_names() -> list[str]:
    names: list[str] = []
    for column in OPERATIONAL_COLUMNS:
        names.extend([f"{column}_lag_1", f"{column}_lag_48", f"{column}_roll_mean_48"])
    return names


def build_supervised_frame(
    intervals: pd.DataFrame,
    target: str,
    lags: list[int],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Build leakage-safe rolling-origin supervised rows.

    Only calendar values known for the target timestamp and lagged historical quantities are
    features. Realized current-interval AHT, patience, shrinkage, and regime are never used as
    predictive inputs.
    """
    if target not in intervals.columns:
        raise KeyError(f"Target column {target!r} is missing")
    missing_operational = [
        column for column in OPERATIONAL_COLUMNS if column not in intervals.columns
    ]
    if missing_operational:
        raise KeyError(f"Operational columns are missing: {missing_operational}")

    out = add_time_features(intervals)
    out = out.sort_values(["region", "skill", "timestamp"]).reset_index(drop=True)
    target_groups = out.groupby(["region", "skill"], sort=False)[target]
    feature_columns = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]

    for lag in sorted(set(lags)):
        column = f"lag_{lag}"
        out[column] = target_groups.shift(lag)
        feature_columns.append(column)

    for window in sorted(set(rolling_windows)):
        shifted = target_groups.shift(1)
        mean_column = f"roll_mean_{window}"
        std_column = f"roll_std_{window}"
        out[mean_column] = shifted.groupby([out["region"], out["skill"]]).transform(
            lambda series, window=window: series.rolling(
                window, min_periods=max(2, min(window, 4))
            ).mean()
        )
        out[std_column] = shifted.groupby([out["region"], out["skill"]]).transform(
            lambda series, window=window: series.rolling(
                window, min_periods=max(2, min(window, 4))
            ).std()
        )
        feature_columns.extend([mean_column, std_column])

    for operational_column in OPERATIONAL_COLUMNS:
        operational_groups = out.groupby(["region", "skill"], sort=False)[operational_column]
        lag_1 = f"{operational_column}_lag_1"
        lag_48 = f"{operational_column}_lag_48"
        rolling = f"{operational_column}_roll_mean_48"
        out[lag_1] = operational_groups.shift(1)
        out[lag_48] = operational_groups.shift(48)
        shifted_operational = operational_groups.shift(1)
        out[rolling] = shifted_operational.groupby([out["region"], out["skill"]]).transform(
            lambda series: series.rolling(48, min_periods=4).mean()
        )
        feature_columns.extend([lag_1, lag_48, rolling])

    regions = sorted(out["region"].astype(str).unique())
    skills = sorted(out["skill"].astype(str).unique())
    region_map = {value: index for index, value in enumerate(regions)}
    skill_map = {value: index for index, value in enumerate(skills)}
    out["region_code"] = out["region"].astype(str).map(region_map).astype(int)
    out["skill_code"] = out["skill"].astype(str).map(skill_map).astype(int)
    feature_columns.extend(["region_code", "skill_code"])

    out = out.dropna(subset=feature_columns + [target]).reset_index(drop=True)
    return out, feature_columns


def _last_or_nan(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) < lag:
        return float("nan")
    return float(values[-lag])


def _rolling_stat(values: np.ndarray, window: int, statistic: str) -> float:
    if len(values) < 2:
        return float("nan")
    selected = values[-min(window, len(values)) :]
    if statistic == "mean":
        return float(np.mean(selected))
    if len(selected) < 2:
        return float("nan")
    return float(np.std(selected, ddof=1))


def build_recursive_feature_rows(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    target: str,
    lags: list[int],
    rolling_windows: list[int],
    regions: Iterable[str],
    skills: Iterable[str],
    planning_parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Create one fixed-origin recursive feature row per region-skill leaf.

    ``history`` must contain only observations or prior recursive predictions available before
    ``timestamp``. ``planning_parameters`` contains forecast-time assumptions for AHT, patience,
    and shrinkage and is not allowed to contain realized future values.
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    regions_list = sorted(str(value) for value in regions)
    skills_list = sorted(str(value) for value in skills)
    region_map = {value: index for index, value in enumerate(regions_list)}
    skill_map = {value: index for index, value in enumerate(skills_list)}
    parameter_lookup = planning_parameters.set_index(["region", "skill"])

    rows: list[dict[str, object]] = []
    for region in regions_list:
        for skill in skills_list:
            leaf_history = history[
                (history["region"].astype(str) == region)
                & (history["skill"].astype(str) == skill)
                & (pd.to_datetime(history["timestamp"], utc=True) < ts)
            ].sort_values("timestamp")
            if leaf_history.empty:
                raise ValueError(f"No historical rows available for {region}/{skill} before {ts}")

            row: dict[str, object] = {
                "timestamp": ts,
                "region": region,
                "skill": skill,
                "regime": "unknown_future",
                "region_code": region_map[region],
                "skill_code": skill_map[skill],
            }
            hour = ts.hour + ts.minute / 60.0
            dow = ts.dayofweek
            row.update(
                {
                    "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                    "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
                    "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
                    "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
                    "is_weekend": int(dow >= 5),
                }
            )

            target_values = leaf_history[target].to_numpy(float)
            for lag in sorted(set(lags)):
                row[f"lag_{lag}"] = _last_or_nan(target_values, lag)
            for window in sorted(set(rolling_windows)):
                row[f"roll_mean_{window}"] = _rolling_stat(target_values, window, "mean")
                row[f"roll_std_{window}"] = _rolling_stat(target_values, window, "std")

            for operational_column in OPERATIONAL_COLUMNS:
                operational_values = leaf_history[operational_column].to_numpy(float)
                row[f"{operational_column}_lag_1"] = _last_or_nan(operational_values, 1)
                row[f"{operational_column}_lag_48"] = _last_or_nan(operational_values, 48)
                row[f"{operational_column}_roll_mean_48"] = _rolling_stat(
                    operational_values, 48, "mean"
                )

            parameter_row = parameter_lookup.loc[(region, skill)]
            for operational_column in OPERATIONAL_COLUMNS:
                row[f"planning_{operational_column}"] = float(parameter_row[operational_column])
            rows.append(row)

    frame = pd.DataFrame(rows)
    feature_columns = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]
    feature_columns.extend(f"lag_{lag}" for lag in sorted(set(lags)))
    for window in sorted(set(rolling_windows)):
        feature_columns.extend([f"roll_mean_{window}", f"roll_std_{window}"])
    feature_columns.extend(_operational_feature_names())
    feature_columns.extend(["region_code", "skill_code"])
    return frame, feature_columns


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = np.array(sorted(frame["timestamp"].unique()))
    if len(timestamps) < 10:
        raise ValueError("At least ten unique timestamps are required for temporal splitting")
    train_index = max(1, int(len(timestamps) * train_fraction))
    calibration_index = max(
        train_index + 1, int(len(timestamps) * (train_fraction + calibration_fraction))
    )
    calibration_index = min(calibration_index, len(timestamps) - 1)
    train_end = timestamps[train_index - 1]
    calibration_end = timestamps[calibration_index - 1]
    train = frame[frame["timestamp"] <= train_end].copy()
    calibration = frame[
        (frame["timestamp"] > train_end) & (frame["timestamp"] <= calibration_end)
    ].copy()
    test = frame[frame["timestamp"] > calibration_end].copy()
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition")
    return train, calibration, test
