from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from support_capacity_reliability.config import DataConfig


@dataclass(frozen=True)
class SyntheticBundle:
    intervals: pd.DataFrame
    contacts: pd.DataFrame
    agents: pd.DataFrame
    events: pd.DataFrame


SKILL_AHT_SECONDS = {
    "billing": 420.0,
    "technical": 720.0,
    "account": 510.0,
    "fraud": 840.0,
}

SKILL_PATIENCE_SECONDS = {
    "billing": 240.0,
    "technical": 330.0,
    "account": 270.0,
    "fraud": 210.0,
}


def _regime(day_index: int, event_days: set[int]) -> tuple[str, float]:
    if day_index in event_days:
        return "incident", 2.0
    if day_index - 1 in event_days:
        return "recovery", 1.35
    if day_index + 1 in event_days:
        return "pre_incident", 1.10
    return "normal", 1.0


def _daily_shape(hour: float) -> float:
    morning = np.exp(-0.5 * ((hour - 10.0) / 2.5) ** 2)
    afternoon = 0.8 * np.exp(-0.5 * ((hour - 15.5) / 3.0) ** 2)
    overnight = 0.12
    return float(overnight + morning + afternoon)


def generate_agents(config: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    skills = list(config.skills)
    regions = list(config.regions)
    for idx in range(config.agents):
        primary = skills[idx % len(skills)]
        secondary = skills[(idx + 1) % len(skills)] if rng.random() < 0.48 else None
        agent_skills = [primary] + ([secondary] if secondary else [])
        preferred_shift = str(rng.choice(["early", "late"]))
        available_shifts = "early|late" if rng.random() < 0.80 else preferred_shift
        overtime_eligible = bool(rng.random() < 0.75)
        rows.append(
            {
                "agent_id": f"A{idx:04d}",
                "home_region": regions[idx % len(regions)],
                "skills": "|".join(agent_skills),
                "primary_skill": primary,
                "proficiency": round(float(rng.uniform(0.82, 1.18)), 3),
                "regular_hourly_cost": round(float(rng.uniform(29, 39)), 2),
                "overtime_hourly_cost": round(float(rng.uniform(45, 58)), 2),
                "max_weekly_hours": int(rng.choice([36, 40, 40, 44])),
                "max_daily_hours": 12 if overtime_eligible else 6,
                "preferred_shift": preferred_shift,
                "available_shifts": available_shifts,
                "overtime_eligible": overtime_eligible,
                "absence_probability": round(float(rng.uniform(0.02, 0.10)), 3),
                "source_type": "synthetic_operational",
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_acd(config: DataConfig, seed: int = 42) -> SyntheticBundle:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(config.start_date)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    periods = config.days * int(24 * 60 / config.interval_minutes)
    timestamps = pd.date_range(start, periods=periods, freq=f"{config.interval_minutes}min")
    event_days = set(config.event_days)

    interval_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    carry: dict[tuple[str, str], int] = {
        (region, skill): 0 for region in config.regions for skill in config.skills
    }
    contact_id = 0

    for ts_idx, timestamp in enumerate(timestamps):
        day_index = ts_idx // int(24 * 60 / config.interval_minutes)
        hour = timestamp.hour + timestamp.minute / 60.0
        regime, regime_multiplier = _regime(day_index, event_days)
        weekend_multiplier = 0.72 if timestamp.dayofweek >= 5 else 1.0
        if regime != "normal":
            event_rows.append(
                {
                    "event_id": f"EV-{timestamp.date()}-{ts_idx}",
                    "timestamp": timestamp,
                    "event_type": regime,
                    "intensity": regime_multiplier,
                    "source_type": "synthetic_operational",
                }
            )

        for region_index, region in enumerate(config.regions):
            regional_multiplier = 1.0 + region_index * 0.12
            for skill_index, skill in enumerate(config.skills):
                base_rate = 4.2 + skill_index * 1.1 + region_index * 0.8
                skill_event_multiplier = regime_multiplier
                if regime == "incident" and skill == "technical":
                    skill_event_multiplier *= 1.55
                if regime == "incident" and region == config.regions[0]:
                    skill_event_multiplier *= 1.25
                expected = (
                    base_rate
                    * _daily_shape(hour)
                    * weekend_multiplier
                    * regional_multiplier
                    * skill_event_multiplier
                )
                returning_carry = carry[(region, skill)]
                fresh_latent = int(rng.poisson(max(expected, 0.05)))
                latent = fresh_latent + returning_carry
                offered = int(rng.binomial(latent, config.observed_reach_rate)) if latent else 0

                base_aht = SKILL_AHT_SECONDS.get(skill, 540.0)
                aht_multiplier = 1.0 + (
                    0.18 if regime == "incident" else 0.05 if regime == "recovery" else 0
                )
                average_handle_time = max(
                    60.0, float(rng.lognormal(np.log(base_aht * aht_multiplier), 0.18))
                )
                base_patience = SKILL_PATIENCE_SECONDS.get(skill, 270.0)
                patience_multiplier = (
                    0.80 if regime == "incident" else 0.92 if regime == "recovery" else 1.0
                )
                patience_mean = max(
                    45.0,
                    float(
                        rng.lognormal(
                            np.log(base_patience * patience_multiplier),
                            0.10,
                        )
                    ),
                )
                shrinkage_shift = (
                    0.06 if regime == "incident" else 0.025 if regime == "recovery" else 0.0
                )
                shrinkage = float(np.clip(rng.beta(2.5, 15.0) + shrinkage_shift, 0.02, 0.45))

                nominal_agents = max(
                    1.0, config.agents / (len(config.regions) * len(config.skills))
                )
                interval_capacity = (
                    nominal_agents * (config.interval_minutes * 60.0) / average_handle_time
                )
                load_ratio = offered / max(interval_capacity * (1 - shrinkage), 1e-6)
                service_probability = float(
                    np.clip(1.05 - 0.23 * max(load_ratio - 0.75, 0), 0.42, 0.98)
                )
                served = int(rng.binomial(offered, service_probability)) if offered else 0
                abandoned = offered - served
                redials_next = (
                    int(rng.binomial(abandoned, config.redial_probability)) if abandoned else 0
                )
                recontacts = (
                    int(rng.binomial(served, config.recontact_probability)) if served else 0
                )
                returning_contacts_next = redials_next + recontacts
                carry[(region, skill)] = returning_contacts_next

                # In the controlled generator, inverse-propensity correction reconstructs the
                # current latent offered demand from the contacts that reached the ACD. Future
                # redials and recontacts are carried to the next interval instead of being
                # incorrectly added to the current target.
                offered_load_estimate = offered / max(config.observed_reach_rate, 1e-6)
                interval_rows.append(
                    {
                        "timestamp": timestamp,
                        "region": region,
                        "skill": skill,
                        "regime": regime,
                        "latent_demand": latent,
                        "fresh_latent_demand": fresh_latent,
                        "offered_contacts": offered,
                        "served_contacts": served,
                        "abandoned_contacts": abandoned,
                        "redial_contacts_next": redials_next,
                        "recontacts": recontacts,
                        "returning_contacts_next": returning_contacts_next,
                        "observed_served": served,
                        "offered_load_estimate": float(offered_load_estimate),
                        "average_handle_time_seconds": average_handle_time,
                        "patience_mean_seconds": patience_mean,
                        "shrinkage_rate": shrinkage,
                        "source_type": "synthetic_operational",
                        "generator_seed": seed,
                    }
                )

                if config.save_event_level and offered > 0:
                    for local_idx in range(offered):
                        contact_id += 1
                        was_served = local_idx < served
                        service_time = float(rng.lognormal(np.log(base_aht * aht_multiplier), 0.35))
                        patience = float(rng.exponential(patience_mean))
                        contact_rows.append(
                            {
                                "contact_id": f"C{contact_id:09d}",
                                "interval_timestamp": timestamp,
                                "region": region,
                                "skill": skill,
                                "regime": regime,
                                "service_time_seconds": max(30.0, service_time),
                                "patience_seconds": max(5.0, patience),
                                "served_observed": was_served,
                                "abandoned_observed": not was_served,
                                "source_type": "synthetic_operational",
                                "generator_seed": seed,
                            }
                        )

    intervals = pd.DataFrame(interval_rows).sort_values(["timestamp", "region", "skill"])
    contacts = pd.DataFrame(contact_rows)
    agents = generate_agents(config, rng)
    events = pd.DataFrame(event_rows).drop_duplicates(subset=["timestamp", "event_type"])
    return SyntheticBundle(intervals=intervals, contacts=contacts, agents=agents, events=events)
