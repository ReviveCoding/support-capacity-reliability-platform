from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class CapacityPlan:
    skills: list[str]
    regular_capacity_units: np.ndarray
    expected_shortage: np.ndarray
    expected_excess: np.ndarray
    objective_value: float
    success: bool
    status: str

    @property
    def regular_fte(self) -> np.ndarray:
        """Backward-compatible alias; values are concurrent capacity units, not weekly FTE."""
        return self.regular_capacity_units

    def to_records(
        self,
        *,
        planning_horizon_hours: float,
        standard_week_hours: float = 40.0,
    ) -> list[dict[str, object]]:
        return [
            {
                "skill": skill,
                "regular_capacity_units": int(round(float(self.regular_capacity_units[idx]))),
                "fte_equivalent_for_horizon": float(
                    self.regular_capacity_units[idx] * planning_horizon_hours / standard_week_hours
                ),
                "planning_horizon_hours": float(planning_horizon_hours),
                "expected_shortage_capacity_units": float(self.expected_shortage[idx]),
                "expected_excess_capacity_units": float(self.expected_excess[idx]),
            }
            for idx, skill in enumerate(self.skills)
        ]


class StrategicCapacityPlanner:
    """Two-stage integer capacity planning with shortage and excess recourse.

    The first-stage variable is regular concurrent capacity by skill. For every scenario and skill,
    shortage and excess variables reconcile regular capacity with scenario demand:

        regular_capacity + shortage - excess = required_capacity

    This prevents an unused excess-capacity penalty and makes the cost trade-off
    explicit instead of implicitly treating overstaffing as free beyond labor cost.
    """

    def __init__(
        self,
        regular_cost: float,
        shortage_penalty: float,
        excess_penalty: float = 0.0,
        time_limit_seconds: float = 20.0,
    ) -> None:
        self.regular_cost = float(regular_cost)
        self.shortage_penalty = float(shortage_penalty)
        self.excess_penalty = float(excess_penalty)
        self.time_limit_seconds = float(time_limit_seconds)

    def solve(self, required_fte_scenarios: np.ndarray, skills: list[str]) -> CapacityPlan:
        required = np.asarray(required_fte_scenarios, dtype=float)
        if required.ndim != 2 or required.shape[1] != len(skills):
            raise ValueError("required_fte_scenarios must have shape [scenario, skill]")
        if required.shape[0] == 0:
            raise ValueError("at least one scenario is required")
        if not np.isfinite(required).all() or (required < 0).any():
            raise ValueError("required_fte_scenarios must be finite and nonnegative")

        scenarios, skill_count = required.shape
        recourse_count = scenarios * skill_count
        variable_count = skill_count + 2 * recourse_count
        shortage_start = skill_count
        excess_start = skill_count + recourse_count

        c = np.concatenate(
            [
                np.full(skill_count, self.regular_cost, dtype=float),
                np.full(recourse_count, self.shortage_penalty / scenarios, dtype=float),
                np.full(recourse_count, self.excess_penalty / scenarios, dtype=float),
            ]
        )
        integrality = np.concatenate([np.ones(skill_count), np.zeros(2 * recourse_count)])
        lower = np.zeros(variable_count)
        upper = np.full(variable_count, np.inf)

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for scenario in range(scenarios):
            for skill in range(skill_count):
                offset = scenario * skill_count + skill
                row = np.zeros(variable_count)
                row[skill] = 1.0
                row[shortage_start + offset] = 1.0
                row[excess_start + offset] = -1.0
                rows.append(row)
                rhs.append(required[scenario, skill])

        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(np.asarray(rows), np.asarray(rhs), np.asarray(rhs)),
            options={"time_limit": self.time_limit_seconds},
        )

        if result.x is None:
            fallback = np.ceil(np.quantile(required, 0.9, axis=0))
            shortage = np.maximum(required - fallback[None, :], 0).mean(axis=0)
            excess = np.maximum(fallback[None, :] - required, 0).mean(axis=0)
            objective = (
                self.regular_cost * fallback.sum()
                + self.shortage_penalty * shortage.sum()
                + self.excess_penalty * excess.sum()
            )
            return CapacityPlan(
                skills,
                fallback,
                shortage,
                excess,
                float(objective),
                False,
                str(result.message),
            )

        regular = result.x[:skill_count]
        shortage_values = result.x[shortage_start:excess_start].reshape(scenarios, skill_count)
        excess_values = result.x[excess_start:].reshape(scenarios, skill_count)
        return CapacityPlan(
            skills=skills,
            regular_capacity_units=regular,
            expected_shortage=shortage_values.mean(axis=0),
            expected_excess=excess_values.mean(axis=0),
            objective_value=float(result.fun),
            success=bool(result.success),
            status=str(result.message),
        )
