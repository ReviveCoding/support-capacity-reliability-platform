from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContractResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


def _finite_numeric_errors(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
            errors.append(f"Non-finite or non-numeric values found in {column}")
    return errors


def validate_intervals(
    frame: pd.DataFrame,
    *,
    target_column: str = "offered_load_estimate",
) -> ContractResult:
    """Validate the interval schema consumed by forecasting and queue planning."""
    required = {
        "timestamp",
        "region",
        "skill",
        "regime",
        "latent_demand",
        "offered_contacts",
        "served_contacts",
        "abandoned_contacts",
        "average_handle_time_seconds",
        "patience_mean_seconds",
        "shrinkage_rate",
        "source_type",
        target_column,
    }
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(required.difference(frame.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")
        return ContractResult(False, errors, warnings)
    if frame.empty:
        errors.append("Interval frame is empty")
        return ContractResult(False, errors, warnings)

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        errors.append("Missing or unparseable timestamps")
    if frame.duplicated(["timestamp", "region", "skill"]).any():
        errors.append("Duplicate timestamp-region-skill rows detected")
    if frame["region"].isna().any() or (frame["region"].astype(str).str.strip() == "").any():
        errors.append("Region values must be non-empty")
    if frame["skill"].isna().any() or (frame["skill"].astype(str).str.strip() == "").any():
        errors.append("Skill values must be non-empty")
    if frame["regime"].isna().any() or (frame["regime"].astype(str).str.strip() == "").any():
        errors.append("Regime values must be non-empty")

    numeric_columns = [
        "latent_demand",
        "offered_contacts",
        "served_contacts",
        "abandoned_contacts",
        "average_handle_time_seconds",
        "patience_mean_seconds",
        "shrinkage_rate",
        target_column,
    ]
    errors.extend(_finite_numeric_errors(frame, numeric_columns))
    if not errors:
        nonnegative = [
            "latent_demand",
            "offered_contacts",
            "served_contacts",
            "abandoned_contacts",
            target_column,
        ]
        for column in nonnegative:
            if (frame[column] < 0).any():
                errors.append(f"Negative values found in {column}")
        if (frame["average_handle_time_seconds"] <= 0).any():
            errors.append("average_handle_time_seconds must be positive")
        if (frame["patience_mean_seconds"] <= 0).any():
            errors.append("patience_mean_seconds must be positive")
        if ((frame["shrinkage_rate"] < 0) | (frame["shrinkage_rate"] >= 1)).any():
            errors.append("shrinkage_rate must be in [0, 1)")

    imbalance = frame["served_contacts"] + frame["abandoned_contacts"] - frame["offered_contacts"]
    if not (imbalance == 0).all():
        errors.append("Flow conservation failed: served + abandoned != offered")
    if (
        frame["source_type"].isna().any()
        or (frame["source_type"].astype(str).str.strip() == "").any()
    ):
        errors.append("Missing provenance in source_type")
    if frame["offered_contacts"].sum() == 0:
        warnings.append("No offered contacts were generated")
    if (frame["offered_contacts"] > frame["latent_demand"]).any():
        warnings.append("Some offered contacts exceed latent demand; verify proxy-target semantics")
    return ContractResult(not errors, errors, warnings)


def validate_agents(
    frame: pd.DataFrame,
    *,
    allowed_skills: Iterable[str] | None = None,
    allowed_shifts: Iterable[str] = ("early", "late"),
) -> ContractResult:
    """Validate the workforce schema required by scheduling, simulation, and recourse."""
    required = {
        "agent_id",
        "home_region",
        "skills",
        "primary_skill",
        "proficiency",
        "regular_hourly_cost",
        "overtime_hourly_cost",
        "preferred_shift",
        "available_shifts",
        "overtime_eligible",
        "max_daily_hours",
        "absence_probability",
        "source_type",
    }
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(required.difference(frame.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")
        return ContractResult(False, errors, warnings)
    if frame.empty:
        errors.append("Agent frame is empty")
        return ContractResult(False, errors, warnings)
    if frame["agent_id"].isna().any() or (frame["agent_id"].astype(str).str.strip() == "").any():
        errors.append("Agent IDs must be non-empty")
    if frame["agent_id"].duplicated().any():
        errors.append("Duplicate agent IDs")

    numeric_columns = [
        "proficiency",
        "regular_hourly_cost",
        "overtime_hourly_cost",
        "max_daily_hours",
        "absence_probability",
    ]
    errors.extend(_finite_numeric_errors(frame, numeric_columns))
    if not errors:
        if (frame["proficiency"] <= 0).any():
            errors.append("Agent proficiency must be positive")
        if (frame["regular_hourly_cost"] <= 0).any():
            errors.append("Agent regular cost must be positive")
        if (frame["overtime_hourly_cost"] < frame["regular_hourly_cost"]).any():
            errors.append("Overtime cost must be at least regular cost")
        if (frame["max_daily_hours"] <= 0).any():
            errors.append("max_daily_hours must be positive")
        if ((frame["absence_probability"] < 0) | (frame["absence_probability"] > 1)).any():
            errors.append("absence_probability must be in [0, 1]")

    allowed_skill_set = set(str(value) for value in allowed_skills) if allowed_skills else None
    allowed_shift_set = set(str(value) for value in allowed_shifts)
    for row in frame.itertuples(index=False):
        skills = {value for value in str(row.skills).split("|") if value}
        shifts = {value for value in str(row.available_shifts).split("|") if value}
        if not skills:
            errors.append(f"Agent {row.agent_id} must have at least one skill")
        if allowed_skill_set is not None and not skills.issubset(allowed_skill_set):
            errors.append(
                f"Agent {row.agent_id} has unsupported skills: {sorted(skills - allowed_skill_set)}"
            )
        if str(row.primary_skill) not in skills:
            errors.append(f"Agent {row.agent_id} primary_skill must be included in skills")
        if not shifts:
            errors.append(f"Agent {row.agent_id} must have at least one available shift")
        if not shifts.issubset(allowed_shift_set):
            errors.append(
                f"Agent {row.agent_id} has unsupported shifts: {sorted(shifts - allowed_shift_set)}"
            )
        if str(row.preferred_shift) not in shifts:
            warnings.append(f"Agent {row.agent_id} preferred shift is unavailable")
    return ContractResult(not errors, errors, warnings)
