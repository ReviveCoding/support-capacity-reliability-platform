import pandas as pd

from support_capacity_reliability.config import load_config
from support_capacity_reliability.data.contracts import validate_agents, validate_intervals
from support_capacity_reliability.data.synthetic import generate_synthetic_acd


def test_synthetic_bundle_is_deterministic():
    config = load_config("configs/smoke.yaml")
    first = generate_synthetic_acd(config.data, seed=7)
    second = generate_synthetic_acd(config.data, seed=7)
    assert first.intervals.equals(second.intervals)
    assert first.agents.equals(second.agents)


def test_synthetic_flow_conservation():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=3)
    assert validate_intervals(bundle.intervals).passed
    assert validate_agents(bundle.agents).passed
    assert (
        bundle.intervals["served_contacts"] + bundle.intervals["abandoned_contacts"]
        == bundle.intervals["offered_contacts"]
    ).all()


def test_incident_increases_technical_demand_on_average():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=11)
    technical = bundle.intervals[bundle.intervals["skill"] == "technical"]
    incident = technical[technical["regime"] == "incident"]["latent_demand"].mean()
    normal = technical[technical["regime"] == "normal"]["latent_demand"].mean()
    assert incident > normal


def test_returning_contacts_are_time_aligned_and_feed_next_interval():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=17)
    intervals = bundle.intervals.sort_values(["region", "skill", "timestamp"]).copy()
    assert (
        intervals["returning_contacts_next"]
        == intervals["redial_contacts_next"] + intervals["recontacts"]
    ).all()
    for _, group in intervals.groupby(["region", "skill"]):
        group = group.reset_index(drop=True)
        next_returning_component = (
            group["latent_demand"].iloc[1:].to_numpy()
            - group["fresh_latent_demand"].iloc[1:].to_numpy()
        )
        assert (
            next_returning_component == group["returning_contacts_next"].iloc[:-1].to_numpy()
        ).all()


def test_offered_load_estimate_corrects_current_reach_without_future_double_counting():
    config = load_config("configs/smoke.yaml")
    bundle = generate_synthetic_acd(config.data, seed=19)
    expected = bundle.intervals["offered_contacts"] / config.data.observed_reach_rate
    assert (bundle.intervals["offered_load_estimate"] - expected).abs().max() < 1e-12


def test_timezone_aware_start_date_is_normalized_to_utc():
    config = load_config("configs/smoke.yaml")
    aware_data = config.data.model_copy(update={"start_date": "2026-01-01T03:00:00+03:00"})
    bundle = generate_synthetic_acd(aware_data, seed=23)
    first_timestamp = pd.to_datetime(bundle.intervals["timestamp"], utc=True).min()
    assert first_timestamp == pd.Timestamp("2026-01-01T00:00:00Z")
