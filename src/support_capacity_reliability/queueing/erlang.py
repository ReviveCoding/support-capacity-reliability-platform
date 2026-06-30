from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np


@dataclass(frozen=True)
class QueueApproximation:
    agents: int
    utilization: float
    probability_wait: float
    average_wait_seconds: float
    abandonment_rate: float
    service_level: float


def erlang_c(
    arrival_rate_per_second: float,
    service_rate_per_second: float,
    agents: int,
    service_level_seconds: float = 120,
) -> QueueApproximation:
    if arrival_rate_per_second < 0 or service_rate_per_second <= 0 or agents <= 0:
        raise ValueError("Invalid Erlang-C parameters")
    offered_load = arrival_rate_per_second / service_rate_per_second
    utilization = offered_load / agents
    if utilization >= 1:
        return QueueApproximation(agents, utilization, 1.0, float("inf"), 0.0, 0.0)
    series = sum(offered_load**k / factorial(k) for k in range(agents))
    tail = offered_load**agents / (factorial(agents) * (1 - utilization))
    probability_wait = tail / (series + tail)
    wait_rate = agents * service_rate_per_second - arrival_rate_per_second
    average_wait = probability_wait / max(wait_rate, 1e-12)
    service_level = 1 - probability_wait * exp(-wait_rate * service_level_seconds)
    return QueueApproximation(
        agents=agents,
        utilization=utilization,
        probability_wait=probability_wait,
        average_wait_seconds=average_wait,
        abandonment_rate=0.0,
        service_level=float(np.clip(service_level, 0, 1)),
    )


def erlang_a(
    arrival_rate_per_second: float,
    service_rate_per_second: float,
    patience_rate_per_second: float,
    agents: int,
    service_level_seconds: float = 120,
    max_states: int = 500,
) -> QueueApproximation:
    if (
        arrival_rate_per_second < 0
        or service_rate_per_second <= 0
        or patience_rate_per_second <= 0
        or agents <= 0
    ):
        raise ValueError("Invalid Erlang-A parameters")
    ratios = [1.0]
    for n in range(1, max_states):
        death = service_rate_per_second * min(n, agents) + patience_rate_per_second * max(
            n - agents, 0
        )
        ratios.append(ratios[-1] * arrival_rate_per_second / max(death, 1e-12))
        if ratios[-1] < 1e-14 and n > agents + 20:
            break
    probabilities = np.asarray(ratios, dtype=float)
    probabilities /= probabilities.sum()
    states = np.arange(len(probabilities))
    queue_lengths = np.maximum(states - agents, 0)
    expected_queue = float(np.sum(probabilities * queue_lengths))
    expected_busy = float(np.sum(probabilities * np.minimum(states, agents)))
    abandonments_per_second = patience_rate_per_second * expected_queue
    abandonment_rate = abandonments_per_second / max(arrival_rate_per_second, 1e-12)
    served_rate = service_rate_per_second * expected_busy
    utilization = expected_busy / agents
    probability_wait = float(probabilities[agents:].sum()) if len(probabilities) > agents else 0.0
    average_wait = expected_queue / max(arrival_rate_per_second, 1e-12)
    effective_clearance = max(
        agents * service_rate_per_second + patience_rate_per_second - arrival_rate_per_second,
        patience_rate_per_second,
    )
    service_level = 1 - probability_wait * exp(-effective_clearance * service_level_seconds)
    if served_rate + abandonments_per_second > 0:
        abandonment_rate = abandonments_per_second / (served_rate + abandonments_per_second)
    return QueueApproximation(
        agents=agents,
        utilization=float(np.clip(utilization, 0, 10)),
        probability_wait=float(np.clip(probability_wait, 0, 1)),
        average_wait_seconds=max(0.0, average_wait),
        abandonment_rate=float(np.clip(abandonment_rate, 0, 1)),
        service_level=float(np.clip(service_level, 0, 1)),
    )


def required_agents_erlang_a(
    arrival_rate_per_second: float,
    average_handle_time_seconds: float,
    patience_mean_seconds: float,
    service_level_target: float,
    abandonment_target: float,
    service_level_seconds: float,
    max_agents: int = 200,
) -> QueueApproximation:
    service_rate = 1.0 / max(average_handle_time_seconds, 1e-6)
    patience_rate = 1.0 / max(patience_mean_seconds, 1e-6)
    best: QueueApproximation | None = None
    for agents in range(1, max_agents + 1):
        result = erlang_a(
            arrival_rate_per_second,
            service_rate,
            patience_rate,
            agents,
            service_level_seconds=service_level_seconds,
        )
        best = result
        if (
            result.service_level >= service_level_target
            and result.abandonment_rate <= abandonment_target
        ):
            return result
    assert best is not None
    return best
