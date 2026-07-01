from __future__ import annotations

import pandas as pd


def shift_band(shift: str) -> str:
    """Return the availability/preference band for a concrete decision shift."""
    return str(shift).split("_", 1)[0]


def ordered_shifts(shift_mapping: dict[pd.Timestamp, str]) -> list[str]:
    """Return concrete shifts in chronological order without duplicates."""
    return list(dict.fromkeys(shift_mapping.values()))


def resolve_shift_duration_hours(
    horizon: pd.DataFrame,
    *,
    interval_minutes: int,
    configured_shift_duration_hours: float | None,
) -> float:
    """Resolve legacy two-shift duration or an explicit micro-shift duration."""
    timestamps = pd.to_datetime(horizon["timestamp"], utc=True).unique()
    interval_count = max(len(timestamps), 1)
    if configured_shift_duration_hours is None:
        return interval_count * interval_minutes / 60.0 / 2.0
    return float(configured_shift_duration_hours)


def build_shift_mapping(
    horizon: pd.DataFrame,
    *,
    interval_minutes: int,
    configured_shift_duration_hours: float | None,
) -> dict[pd.Timestamp, str]:
    """Map timestamps to legacy two shifts or explicit band-preserving micro-shifts."""
    timestamps = sorted(pd.to_datetime(horizon["timestamp"], utc=True).unique())
    if not timestamps:
        return {}

    if configured_shift_duration_hours is None:
        midpoint = max(1, len(timestamps) // 2)
        return {
            pd.Timestamp(timestamp): ("early" if index < midpoint else "late")
            for index, timestamp in enumerate(timestamps)
        }

    shift_steps_float = configured_shift_duration_hours * 60.0 / interval_minutes
    shift_steps = int(round(shift_steps_float))
    if shift_steps <= 0 or abs(shift_steps_float - shift_steps) > 1e-9:
        raise ValueError(
            "configured_shift_duration_hours must contain an integer number of decision intervals"
        )
    if len(timestamps) % shift_steps != 0:
        raise ValueError(
            "decision horizon length must be divisible by configured_shift_duration_hours"
        )

    shift_count = len(timestamps) // shift_steps
    if shift_count < 2 or shift_count % 2 != 0:
        raise ValueError("explicit micro-shift mode requires an even number of at least two shifts")

    shifts_per_band = shift_count // 2
    mapping: dict[pd.Timestamp, str] = {}
    for index, timestamp in enumerate(timestamps):
        shift_index = index // shift_steps
        band = "early" if shift_index < shifts_per_band else "late"
        band_index = shift_index + 1 if band == "early" else shift_index - shifts_per_band + 1
        mapping[pd.Timestamp(timestamp)] = f"{band}_{band_index}"
    return mapping
