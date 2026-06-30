from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GateCheck:
    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    detail: str


@dataclass
class ReleaseDecision:
    status: str
    checks: list[GateCheck]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": [asdict(check) for check in self.checks]}


_REQUIRED_FORECAST_METRICS = {
    "coverage_error",
    "worst_slice_wape",
    "peak_q90_coverage",
    "incident_q90_coverage",
    "peak_sample_count",
    "incident_sample_count",
    "fixed_origin_wape",
    "fixed_origin_coverage_error",
    "scenario_coherence_max_abs_error",
}
_REQUIRED_DECISION_METRICS = {
    "capacity_plan_success",
    "strategic_tactical_alignment",
    "schedule_feasibility",
    "abandonment_rate_ucb95",
    "service_level_lcb95",
    "p95_wait_seconds_ucb95",
    "hard_violations",
    "recourse_action_rate",
    "recourse_cost_share",
    "flow_conservation",
}
_REQUIRED_THRESHOLDS = {
    "max_interval_coverage_error",
    "max_worst_slice_wape",
    "min_peak_q90_coverage",
    "min_incident_q90_coverage",
    "min_peak_sample_count",
    "min_incident_sample_count",
    "max_fixed_origin_wape",
    "max_fixed_origin_coverage_error",
    "max_scenario_coherence_error",
    "require_capacity_plan_success",
    "min_strategic_tactical_alignment",
    "min_schedule_feasibility",
    "max_abandonment_rate",
    "min_service_level_lcb95",
    "max_p95_wait_seconds",
    "max_recourse_action_rate",
    "max_recourse_cost_share",
    "max_hard_violations",
}


def _require_keys(name: str, values: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"{name} is missing required release-gate fields: {missing}")


def evaluate_release_gate(
    forecast_metrics: dict[str, float],
    decision_metrics: dict[str, float | bool],
    thresholds: dict[str, float | bool],
    *,
    recourse_required: bool = False,
    post_recourse_eligible: bool = False,
) -> ReleaseDecision:
    """Evaluate a fail-closed release contract.

    Every blocking metric must be supplied explicitly. Missing evidence raises rather than
    receiving a favorable default, preventing partially written or schema-drifted artifacts
    from being released accidentally.
    """
    _require_keys("forecast_metrics", forecast_metrics, _REQUIRED_FORECAST_METRICS)
    _require_keys("decision_metrics", decision_metrics, _REQUIRED_DECISION_METRICS)
    _require_keys("thresholds", thresholds, _REQUIRED_THRESHOLDS)

    abandonment_ucb = float(decision_metrics["abandonment_rate_ucb95"])
    service_level_lcb = float(decision_metrics["service_level_lcb95"])
    flow_conservation = bool(decision_metrics["flow_conservation"])
    capacity_success = bool(decision_metrics["capacity_plan_success"])
    require_capacity_success = bool(thresholds["require_capacity_plan_success"])

    checks = [
        GateCheck(
            "interval_coverage_error",
            forecast_metrics["coverage_error"] <= thresholds["max_interval_coverage_error"],
            forecast_metrics["coverage_error"],
            thresholds["max_interval_coverage_error"],
            "Absolute error from the nominal 80% interval coverage.",
        ),
        GateCheck(
            "worst_slice_wape",
            forecast_metrics["worst_slice_wape"] <= thresholds["max_worst_slice_wape"],
            forecast_metrics["worst_slice_wape"],
            thresholds["max_worst_slice_wape"],
            "Worst supported regime-level WAPE.",
        ),
        GateCheck(
            "peak_sample_count",
            int(forecast_metrics["peak_sample_count"]) >= int(thresholds["min_peak_sample_count"]),
            int(forecast_metrics["peak_sample_count"]),
            int(thresholds["min_peak_sample_count"]),
            "Minimum sample count required for the peak-tail coverage check.",
        ),
        GateCheck(
            "incident_sample_count",
            int(forecast_metrics["incident_sample_count"])
            >= int(thresholds["min_incident_sample_count"]),
            int(forecast_metrics["incident_sample_count"]),
            int(thresholds["min_incident_sample_count"]),
            "Minimum incident sample count required before incident coverage can pass.",
        ),
        GateCheck(
            "peak_q90_coverage",
            forecast_metrics["peak_q90_coverage"] >= thresholds["min_peak_q90_coverage"],
            forecast_metrics["peak_q90_coverage"],
            thresholds["min_peak_q90_coverage"],
            "Q90 coverage on observations above the training-period 90th-percentile load.",
        ),
        GateCheck(
            "incident_q90_coverage",
            forecast_metrics["incident_q90_coverage"] >= thresholds["min_incident_q90_coverage"],
            forecast_metrics["incident_q90_coverage"],
            thresholds["min_incident_q90_coverage"],
            "Q90 coverage on incident-regime observations; insufficient incident evidence fails through the separate sample-count gate.",
        ),
        GateCheck(
            "fixed_origin_wape",
            forecast_metrics["fixed_origin_wape"] <= thresholds["max_fixed_origin_wape"],
            forecast_metrics["fixed_origin_wape"],
            thresholds["max_fixed_origin_wape"],
            "Frozen fixed-origin recursive forecast WAPE.",
        ),
        GateCheck(
            "fixed_origin_coverage_error",
            forecast_metrics["fixed_origin_coverage_error"]
            <= thresholds["max_fixed_origin_coverage_error"],
            forecast_metrics["fixed_origin_coverage_error"],
            thresholds["max_fixed_origin_coverage_error"],
            "Absolute coverage error for the fixed-origin recursive decision forecast.",
        ),
        GateCheck(
            "scenario_coherence",
            forecast_metrics["scenario_coherence_max_abs_error"]
            <= thresholds["max_scenario_coherence_error"],
            forecast_metrics["scenario_coherence_max_abs_error"],
            thresholds["max_scenario_coherence_error"],
            "Maximum aggregation mismatch between leaf scenarios and global totals.",
        ),
        GateCheck(
            "capacity_plan_success",
            capacity_success if require_capacity_success else True,
            str(capacity_success),
            str(require_capacity_success),
            "Strategic capacity optimization must return a successful solver result.",
        ),
        GateCheck(
            "strategic_tactical_alignment",
            float(decision_metrics["strategic_tactical_alignment"])
            >= thresholds["min_strategic_tactical_alignment"],
            float(decision_metrics["strategic_tactical_alignment"]),
            thresholds["min_strategic_tactical_alignment"],
            "Share of tactical shift-skill coverage supported by strategic capacity units.",
        ),
        GateCheck(
            "schedule_feasibility",
            float(decision_metrics["schedule_feasibility"])
            >= thresholds["min_schedule_feasibility"],
            float(decision_metrics["schedule_feasibility"]),
            thresholds["min_schedule_feasibility"],
            "Fraction of required coverage constraints satisfied without hard violations.",
        ),
        GateCheck(
            "abandonment_rate",
            abandonment_ucb <= thresholds["max_abandonment_rate"],
            abandonment_ucb,
            thresholds["max_abandonment_rate"],
            "95% upper confidence bound for simulated abandonment.",
        ),
        GateCheck(
            "service_level_lcb95",
            service_level_lcb >= thresholds["min_service_level_lcb95"],
            service_level_lcb,
            thresholds["min_service_level_lcb95"],
            "95% lower confidence bound for offered-contact service level.",
        ),
        GateCheck(
            "p95_wait_seconds_ucb95",
            float(decision_metrics["p95_wait_seconds_ucb95"]) <= thresholds["max_p95_wait_seconds"],
            float(decision_metrics["p95_wait_seconds_ucb95"]),
            thresholds["max_p95_wait_seconds"],
            "95% upper confidence bound for simulated p95 queue waiting time.",
        ),
        GateCheck(
            "hard_violations",
            int(decision_metrics["hard_violations"]) <= int(thresholds["max_hard_violations"]),
            int(decision_metrics["hard_violations"]),
            int(thresholds["max_hard_violations"]),
            "Remaining post-recourse shift-skill coverage violations.",
        ),
        GateCheck(
            "recourse_action_rate",
            float(decision_metrics["recourse_action_rate"])
            <= thresholds["max_recourse_action_rate"],
            float(decision_metrics["recourse_action_rate"]),
            thresholds["max_recourse_action_rate"],
            "Applied schedule-repair actions divided by pre-recourse assigned shifts.",
        ),
        GateCheck(
            "recourse_cost_share",
            float(decision_metrics["recourse_cost_share"]) <= thresholds["max_recourse_cost_share"],
            float(decision_metrics["recourse_cost_share"]),
            thresholds["max_recourse_cost_share"],
            "Positive recourse labor cost divided by deployed labor cost.",
        ),
        GateCheck(
            "flow_conservation",
            flow_conservation,
            str(flow_conservation),
            "True",
            "All queue replications must satisfy served + abandoned = offered.",
        ),
    ]
    failed_check_names = {check.name for check in checks if not check.passed}
    status = "PASS" if not failed_check_names else "ITERATE"

    # A strictly bounded recourse stage may recover the one explicitly labeled
    # pre-recourse strategic-to-tactical support shortfall. All other hard gates,
    # including recourse action-rate and recourse-cost-share limits, must still pass.
    recourse_recoverable_failures = {"strategic_tactical_alignment"}
    only_recourse_recoverable_failure = bool(failed_check_names) and (
        failed_check_names <= recourse_recoverable_failures
    )
    if (
        recourse_required
        and post_recourse_eligible
        and (status == "PASS" or only_recourse_recoverable_failure)
    ):
        status = "PASS_WITH_RECOURSE"

    return ReleaseDecision(status, checks)
