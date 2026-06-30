from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import simpy


@dataclass
class SimAgent:
    agent_id: str
    skills: set[str]
    proficiency: float


@dataclass
class SimulationResult:
    offered: int
    served: int
    abandoned: int
    answered_within_threshold: int
    average_wait_seconds: float
    p95_wait_seconds: float
    service_level: float
    service_level_answered: float
    utilization: float
    flow_conservation: bool
    waits: list[float]

    @property
    def abandonment_rate(self) -> float:
        return self.abandoned / max(self.offered, 1)


class MultiSkillVoiceSimulator:
    """Finite multi-skill voice queue simulation.

    ``service_level`` uses offered contacts as the denominator. This prevents abandoned contacts
    from disappearing from the KPI. ``service_level_answered`` is retained as a diagnostic. Busy
    time is clipped to the staffed operating horizon so utilization cannot exceed one merely
    because contacts finish after the final arrival interval.
    """

    def __init__(self, service_level_seconds: float = 120, seed: int = 42) -> None:
        self.service_level_seconds = service_level_seconds
        self.seed = seed

    def run(
        self,
        interval_plan: pd.DataFrame,
        agents: list[SimAgent],
        interval_minutes: int = 30,
    ) -> SimulationResult:
        if interval_plan.empty:
            return SimulationResult(0, 0, 0, 0, 0.0, 0.0, 1.0, 1.0, 0.0, True, [])
        if not agents:
            offered = int(round(float(interval_plan["offered_contacts"].sum())))
            return SimulationResult(
                offered,
                0,
                offered,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                True,
                [],
            )

        rng = np.random.default_rng(self.seed)
        env = simpy.Environment()
        store = simpy.FilterStore(env, capacity=len(agents))
        for agent in agents:
            store.items.append(agent)

        waits: list[float] = []
        served = 0
        abandoned = 0
        answered_within_threshold = 0
        busy_time_within_horizon = 0.0
        unique_times = sorted(pd.to_datetime(interval_plan["timestamp"], utc=True).unique())
        simulation_end = len(unique_times) * interval_minutes * 60.0

        def contact_process(skill: str, service_time: float, patience: float, arrival_time: float):
            nonlocal served, abandoned, answered_within_threshold, busy_time_within_horizon
            yield env.timeout(max(arrival_time - env.now, 0.0))
            request = store.get(lambda agent: skill in agent.skills)
            remaining_open_seconds = max(simulation_end - arrival_time, 0.0)
            allowed_wait = min(patience, remaining_open_seconds)
            outcome = yield request | env.timeout(allowed_wait)
            wait = float(env.now - arrival_time)
            if request not in outcome:
                request.cancel()
                abandoned += 1
                return

            agent = outcome[request]
            waits.append(wait)
            served += 1
            if wait <= self.service_level_seconds:
                answered_within_threshold += 1
            adjusted_service = service_time / max(agent.proficiency, 0.25)
            service_start = float(env.now)
            service_end = service_start + adjusted_service
            overlap = max(0.0, min(service_end, simulation_end) - max(service_start, 0.0))
            busy_time_within_horizon += overlap
            yield env.timeout(adjusted_service)
            yield store.put(agent)

        time_index = {
            pd.Timestamp(timestamp).value: idx for idx, timestamp in enumerate(unique_times)
        }
        offered = 0
        for row in interval_plan.itertuples(index=False):
            timestamp = pd.Timestamp(row.timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            count = int(round(float(row.offered_contacts)))
            offered += count
            interval_start = time_index[timestamp.value] * interval_minutes * 60.0
            for _ in range(count):
                offset = float(rng.uniform(0, interval_minutes * 60.0))
                service_sigma = 0.30
                service_mean = max(float(row.average_handle_time_seconds), 30.0)
                service_mu = np.log(service_mean) - 0.5 * service_sigma**2
                service_time = float(
                    np.clip(
                        rng.lognormal(service_mu, service_sigma),
                        30.0,
                        8 * 3600.0,
                    )
                )
                patience = float(
                    np.clip(
                        rng.exponential(max(row.patience_mean_seconds, 5.0)),
                        5.0,
                        4 * 3600.0,
                    )
                )
                env.process(
                    contact_process(
                        str(row.skill),
                        service_time,
                        patience,
                        interval_start + offset,
                    )
                )

        # Contacts may finish service after close, but no queued contact can begin a new service
        # after the staffed horizon. Running to exhaustion still guarantees exact flow accounting.
        env.run()
        wait_array = np.asarray(waits, dtype=float)
        average_wait = float(wait_array.mean()) if len(wait_array) else 0.0
        p95_wait = float(np.quantile(wait_array, 0.95)) if len(wait_array) else 0.0
        service_level = answered_within_threshold / max(offered, 1)
        service_level_answered = answered_within_threshold / max(served, 1)
        utilization = busy_time_within_horizon / max(len(agents) * simulation_end, 1e-9)
        flow_conservation = served + abandoned == offered
        return SimulationResult(
            offered=offered,
            served=served,
            abandoned=abandoned,
            answered_within_threshold=answered_within_threshold,
            average_wait_seconds=average_wait,
            p95_wait_seconds=p95_wait,
            service_level=float(service_level),
            service_level_answered=float(service_level_answered),
            utilization=float(np.clip(utilization, 0.0, 1.0)),
            flow_conservation=flow_conservation,
            waits=waits,
        )


def agents_from_schedule(
    schedule: pd.DataFrame,
    agent_frame: pd.DataFrame,
    shift: str,
) -> list[SimAgent]:
    """Create the simulator pool from the actual shift-skill assignment.

    A cross-trained agent may be eligible for several skills, but the tactical scheduler assigns
    one skill for a given shift. Letting the simulator use every certified skill would create
    capacity that was not represented in the schedule and inflate service-level estimates.
    """
    selected = schedule[(schedule["shift"] == shift) & (schedule["assigned"] == 1)].copy()
    selected["agent_id"] = selected["agent_id"].astype(str)
    agent_lookup = agent_frame.assign(agent_id=agent_frame["agent_id"].astype(str)).set_index(
        "agent_id"
    )
    agents: list[SimAgent] = []
    for assignment in selected.itertuples(index=False):
        agent_id = str(assignment.agent_id)
        if agent_id not in agent_lookup.index:
            raise KeyError(f"Scheduled agent {agent_id!r} is missing from the agent registry")
        row = agent_lookup.loc[agent_id]
        assigned_skill = str(assignment.assigned_skill)
        if assigned_skill in {"None", "nan", ""}:
            raise ValueError(f"Assigned schedule row for {agent_id!r} has no assigned_skill")
        certified = set(str(row["skills"]).split("|"))
        if assigned_skill not in certified:
            raise ValueError(
                f"Scheduled agent {agent_id!r} is not certified for assigned skill "
                f"{assigned_skill!r}"
            )
        agents.append(
            SimAgent(
                agent_id=agent_id,
                skills={assigned_skill},
                proficiency=float(row["proficiency"]),
            )
        )
    return agents
