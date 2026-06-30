from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class RecourseAction:
    action: str
    amount: int
    reason: str
    estimated_cost: float
    agent_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScheduleRepairResult:
    schedule: pd.DataFrame
    actions: pd.DataFrame
    remaining_hard_violations: int


def intraday_recourse(
    scheduled_agents: int,
    required_agents: int,
    reserve_available: int,
    overtime_hourly_cost: float,
    excess_capacity_threshold: int = 2,
) -> list[RecourseAction]:
    actions: list[RecourseAction] = []
    gap = required_agents - scheduled_agents
    if gap > 0:
        reserve = min(gap, reserve_available)
        if reserve:
            actions.append(
                RecourseAction(
                    "activate_reserve",
                    reserve,
                    "forecast_or_realized_undercoverage",
                    0.0,
                )
            )
        remaining = gap - reserve
        if remaining > 0:
            actions.append(
                RecourseAction(
                    "offer_overtime",
                    remaining,
                    "reserve_capacity_exhausted",
                    remaining * overtime_hourly_cost * 4,
                )
            )
    elif gap < -excess_capacity_threshold:
        actions.append(
            RecourseAction(
                "offer_voluntary_time_off",
                abs(gap),
                "material_excess_capacity",
                -abs(gap) * overtime_hourly_cost * 2,
            )
        )
    else:
        actions.append(RecourseAction("hold_schedule", 0, "capacity_within_tolerance", 0.0))
    return actions


def apply_intraday_recourse(
    *,
    schedule: pd.DataFrame,
    agents: pd.DataFrame,
    required_coverage: dict[tuple[str, str], int],
    regular_hourly_cost: float,
    overtime_hourly_cost: float,
    shift_duration_hours: float = 8.0,
) -> ScheduleRepairResult:
    """Apply deterministic reserve, overtime, and VTO schedule repair.

    Every added assignment is simulated as available for a full decision-horizon shift and is
    therefore costed using ``shift_duration_hours``. Material excess coverage is actually removed;
    it is not merely emitted as an unapplied recommendation.
    """
    repaired = schedule.copy()
    if "assignment_type" not in repaired.columns:
        repaired["assignment_type"] = repaired["assigned"].map({1: "regular", 0: "unassigned"})
    repaired["agent_id"] = repaired["agent_id"].astype(str)

    skill_map = {
        str(row.agent_id): set(str(row.skills).split("|")) for row in agents.itertuples(index=False)
    }
    preference_map = {
        str(row.agent_id): str(row.preferred_shift) for row in agents.itertuples(index=False)
    }
    availability_map = {
        str(row.agent_id): set(str(getattr(row, "available_shifts", "early|late")).split("|"))
        for row in agents.itertuples(index=False)
    }
    overtime_eligible_map = {
        str(row.agent_id): bool(getattr(row, "overtime_eligible", True))
        for row in agents.itertuples(index=False)
    }
    max_daily_hours_map = {
        str(row.agent_id): float(getattr(row, "max_daily_hours", 2.0 * shift_duration_hours))
        for row in agents.itertuples(index=False)
    }
    proficiency_map = {
        str(row.agent_id): float(row.proficiency) for row in agents.itertuples(index=False)
    }
    regular_cost_map = {
        str(row.agent_id): float(getattr(row, "regular_hourly_cost", regular_hourly_cost))
        for row in agents.itertuples(index=False)
    }
    overtime_cost_map = {
        str(row.agent_id): float(getattr(row, "overtime_hourly_cost", overtime_hourly_cost))
        for row in agents.itertuples(index=False)
    }
    action_rows: list[dict[str, object]] = []

    def achieved(shift: str, skill: str) -> int:
        return int(
            (
                (repaired["shift"] == shift)
                & (repaired["assigned"] == 1)
                & (repaired["assigned_skill"] == skill)
            ).sum()
        )

    for shift, skill in sorted(required_coverage):
        requirement = int(required_coverage[(shift, skill)])
        gap = requirement - achieved(shift, skill)

        if gap < -2:
            removable = repaired[
                (repaired["shift"] == shift)
                & (repaired["assigned"] == 1)
                & (repaired["assigned_skill"] == skill)
            ].copy()
            removable["preference_mismatch"] = removable["agent_id"].map(
                lambda agent_id, shift=shift: int(preference_map.get(agent_id) != shift)
            )
            removable["proficiency"] = removable["agent_id"].map(
                lambda agent_id: proficiency_map.get(agent_id, 1.0)
            )
            removable = removable.sort_values(
                ["preference_mismatch", "proficiency", "agent_id"],
                ascending=[False, True, True],
            )
            for row in removable.head(abs(gap)).itertuples(index=False):
                agent_id = str(row.agent_id)
                row_mask = (repaired["agent_id"] == agent_id) & (repaired["shift"] == shift)
                repaired.loc[row_mask, "assigned"] = 0
                repaired.loc[row_mask, "assigned_skill"] = None
                repaired.loc[row_mask, "assignment_type"] = "vto"
                action_rows.append(
                    {
                        "action": "apply_voluntary_time_off",
                        "amount": 1,
                        "reason": "material_excess_capacity",
                        "estimated_cost": -shift_duration_hours
                        * regular_cost_map.get(agent_id, regular_hourly_cost),
                        "agent_id": agent_id,
                        "shift": shift,
                        "skill": skill,
                    }
                )
            continue

        if gap <= 0:
            action_rows.append(
                {
                    "action": "hold_schedule",
                    "amount": 0,
                    "reason": "capacity_within_tolerance",
                    "estimated_cost": 0.0,
                    "agent_id": None,
                    "shift": shift,
                    "skill": skill,
                }
            )
            continue

        # First repair undercoverage by moving a cross-trained agent from a skill that
        # currently has true surplus in the same shift. This changes routing capacity
        # without creating a double shift or adding labor cost.
        reassignment_candidates: list[tuple[int, float, str, str]] = []
        assigned_rows = repaired[
            (repaired["shift"] == shift)
            & (repaired["assigned"] == 1)
            & (repaired["assigned_skill"] != skill)
        ]
        for row in assigned_rows.itertuples(index=False):
            agent_id = str(row.agent_id)
            source_skill = str(row.assigned_skill)
            if skill not in skill_map.get(agent_id, set()):
                continue
            source_surplus = achieved(shift, source_skill) - int(
                required_coverage.get((shift, source_skill), 0)
            )
            if source_surplus <= 0:
                continue
            reassignment_candidates.append(
                (
                    -source_surplus,
                    -proficiency_map.get(agent_id, 1.0),
                    agent_id,
                    source_skill,
                )
            )
        reassignment_candidates.sort()
        for _, _, agent_id, source_skill in reassignment_candidates:
            if gap <= 0:
                break
            if achieved(shift, source_skill) <= int(
                required_coverage.get((shift, source_skill), 0)
            ):
                continue
            row_mask = (repaired["agent_id"] == agent_id) & (repaired["shift"] == shift)
            repaired.loc[row_mask, "assigned_skill"] = skill
            repaired.loc[row_mask, "assignment_type"] = "cross_skill_reassignment"
            action_rows.append(
                {
                    "action": "cross_skill_reassignment",
                    "amount": 1,
                    "reason": "same_shift_cross_trained_surplus",
                    "estimated_cost": 0.0,
                    "agent_id": agent_id,
                    "shift": shift,
                    "skill": skill,
                    "source_skill": source_skill,
                }
            )
            gap -= 1

        if gap <= 0:
            continue

        target_rows = repaired[(repaired["shift"] == shift) & (repaired["assigned"] == 0)].copy()
        candidates: list[tuple[int, int, float, str]] = []
        for row in target_rows.itertuples(index=False):
            agent_id = str(row.agent_id)
            if skill not in skill_map.get(agent_id, set()):
                continue
            if shift not in availability_map.get(agent_id, {"early", "late"}):
                continue
            if shift_duration_hours > max_daily_hours_map.get(agent_id, 2.0 * shift_duration_hours):
                continue
            total_assignments = int(
                repaired[(repaired["agent_id"] == agent_id) & (repaired["assigned"] == 1)].shape[0]
            )
            if total_assignments > 0:
                projected_hours = (total_assignments + 1) * shift_duration_hours
                if not overtime_eligible_map.get(agent_id, True):
                    continue
                if projected_hours > max_daily_hours_map.get(agent_id, 2.0 * shift_duration_hours):
                    continue
            # Prefer unused reserve agents, then agents whose target shift matches preference,
            # then higher proficiency. The tuple is deterministic because agent_id is final.
            preference_mismatch = int(preference_map.get(agent_id) != shift)
            candidates.append(
                (
                    total_assignments,
                    preference_mismatch,
                    -proficiency_map.get(agent_id, 1.0),
                    agent_id,
                )
            )
        candidates.sort()

        for total_assignments, _, _, agent_id in candidates[:gap]:
            row_mask = (repaired["agent_id"] == agent_id) & (repaired["shift"] == shift)
            assignment_type = "reserve" if total_assignments == 0 else "overtime"
            repaired.loc[row_mask, "assigned"] = 1
            repaired.loc[row_mask, "assigned_skill"] = skill
            repaired.loc[row_mask, "assignment_type"] = assignment_type
            estimated_cost = (
                shift_duration_hours * regular_cost_map.get(agent_id, regular_hourly_cost)
                if assignment_type == "reserve"
                else shift_duration_hours * overtime_cost_map.get(agent_id, overtime_hourly_cost)
            )
            action_rows.append(
                {
                    "action": (
                        "activate_reserve" if assignment_type == "reserve" else "offer_overtime"
                    ),
                    "amount": 1,
                    "reason": "realized_shift_skill_undercoverage",
                    "estimated_cost": estimated_cost,
                    "agent_id": agent_id,
                    "shift": shift,
                    "skill": skill,
                }
            )

    remaining = sum(
        max(int(requirement) - achieved(shift, skill), 0)
        for (shift, skill), requirement in required_coverage.items()
    )
    actions = pd.DataFrame(action_rows)
    if actions.empty:
        actions = pd.DataFrame(
            columns=[
                "action",
                "amount",
                "reason",
                "estimated_cost",
                "agent_id",
                "shift",
                "skill",
                "source_skill",
            ]
        )
    return ScheduleRepairResult(repaired, actions, int(remaining))
