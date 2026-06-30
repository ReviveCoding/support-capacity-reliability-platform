from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ortools.sat.python import cp_model


@dataclass
class ScheduleResult:
    schedule: pd.DataFrame
    status: str
    objective_value: float
    hard_violations: int
    required_coverage: dict[tuple[str, str], int]
    achieved_coverage: dict[tuple[str, str], int]

    @property
    def feasibility(self) -> float:
        total = sum(self.required_coverage.values())
        if total == 0:
            return 1.0
        met = sum(
            min(self.achieved_coverage.get(key, 0), requirement)
            for key, requirement in self.required_coverage.items()
        )
        return met / total


class TacticalShiftScheduler:
    def __init__(
        self,
        time_limit_seconds: int = 15,
        preference_penalty: int = 2,
        shortage_penalty: int = 1000,
    ) -> None:
        self.time_limit_seconds = time_limit_seconds
        self.preference_penalty = preference_penalty
        self.shortage_penalty = shortage_penalty

    def solve(
        self,
        agents: pd.DataFrame,
        required_coverage: dict[tuple[str, str], int],
        shifts: list[str],
        skills: list[str],
        shift_duration_hours: float = 8.0,
    ) -> ScheduleResult:
        model = cp_model.CpModel()
        agent_ids = agents["agent_id"].astype(str).tolist()
        skill_map = {
            str(row.agent_id): set(str(row.skills).split("|"))
            for row in agents.itertuples(index=False)
        }
        preference_map = {
            str(row.agent_id): str(row.preferred_shift) for row in agents.itertuples(index=False)
        }
        availability_map = {
            str(row.agent_id): set(str(getattr(row, "available_shifts", "early|late")).split("|"))
            for row in agents.itertuples(index=False)
        }
        max_daily_hours_map = {
            str(row.agent_id): float(getattr(row, "max_daily_hours", shift_duration_hours))
            for row in agents.itertuples(index=False)
        }
        assign: dict[tuple[str, str, str], cp_model.IntVar] = {}
        works: dict[tuple[str, str], cp_model.IntVar] = {}
        for agent in agent_ids:
            for shift in shifts:
                works[(agent, shift)] = model.NewBoolVar(f"works_{agent}_{shift}")
                if (
                    shift not in availability_map[agent]
                    or shift_duration_hours > max_daily_hours_map[agent]
                ):
                    model.Add(works[(agent, shift)] == 0)
                    continue
                eligible = []
                for skill in skills:
                    if skill in skill_map[agent]:
                        variable = model.NewBoolVar(f"assign_{agent}_{shift}_{skill}")
                        assign[(agent, shift, skill)] = variable
                        eligible.append(variable)
                if eligible:
                    model.Add(sum(eligible) == works[(agent, shift)])
                else:
                    model.Add(works[(agent, shift)] == 0)
            model.Add(sum(works[(agent, shift)] for shift in shifts) <= 1)

        shortages: dict[tuple[str, str], cp_model.IntVar] = {}
        for shift in shifts:
            for skill in skills:
                requirement = int(required_coverage.get((shift, skill), 0))
                shortage = model.NewIntVar(0, max(requirement, 0), f"shortage_{shift}_{skill}")
                shortages[(shift, skill)] = shortage
                eligible_vars = [
                    variable
                    for (agent_id, shift_id, skill_id), variable in assign.items()
                    if shift_id == shift and skill_id == skill
                ]
                model.Add(sum(eligible_vars) + shortage >= requirement)

        labor_cost_terms = []
        preference_terms = []
        for row in agents.itertuples(index=False):
            agent = str(row.agent_id)
            integer_cost = int(round(float(row.regular_hourly_cost) * shift_duration_hours))
            for shift in shifts:
                labor_cost_terms.append(integer_cost * works[(agent, shift)])
                if preference_map[agent] != shift:
                    preference_terms.append(self.preference_penalty * works[(agent, shift)])
        shortage_terms = [self.shortage_penalty * shortage for shortage in shortages.values()]
        model.Minimize(sum(labor_cost_terms) + sum(preference_terms) + sum(shortage_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.time_limit_seconds)
        solver.parameters.num_search_workers = 1
        solver.parameters.max_deterministic_time = 1.0
        status_code = solver.Solve(model)
        status = solver.StatusName(status_code)
        rows = []
        achieved: dict[tuple[str, str], int] = {}
        hard_violations = 0
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for agent in agent_ids:
                for shift in shifts:
                    assigned_skill = None
                    for skill in skills:
                        variable = assign.get((agent, shift, skill))
                        if variable is not None and solver.Value(variable) == 1:
                            assigned_skill = skill
                            achieved[(shift, skill)] = achieved.get((shift, skill), 0) + 1
                    rows.append(
                        {
                            "agent_id": agent,
                            "shift": shift,
                            "assigned": int(solver.Value(works[(agent, shift)])),
                            "assigned_skill": assigned_skill,
                        }
                    )
            hard_violations = int(sum(solver.Value(value) for value in shortages.values()))
            objective = float(solver.ObjectiveValue())
        else:
            for agent in agent_ids:
                for shift in shifts:
                    rows.append(
                        {"agent_id": agent, "shift": shift, "assigned": 0, "assigned_skill": None}
                    )
            objective = float("inf")
            hard_violations = sum(required_coverage.values())
        return ScheduleResult(
            schedule=pd.DataFrame(rows),
            status=status,
            objective_value=objective,
            hard_violations=hard_violations,
            required_coverage=required_coverage,
            achieved_coverage=achieved,
        )
