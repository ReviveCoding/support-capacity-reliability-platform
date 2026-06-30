import pytest

from support_capacity_reliability.reliability.release_gate import evaluate_release_gate


def _forecast_metrics() -> dict[str, float]:
    return {
        "coverage_error": 0.02,
        "worst_slice_wape": 0.2,
        "peak_q90_coverage": 0.8,
        "incident_q90_coverage": 0.9,
        "peak_sample_count": 40,
        "incident_sample_count": 12,
        "fixed_origin_wape": 0.25,
        "fixed_origin_coverage_error": 0.04,
        "scenario_coherence_max_abs_error": 0.0,
    }


def _decision_metrics() -> dict[str, float | bool]:
    return {
        "capacity_plan_success": True,
        "strategic_tactical_alignment": 1.0,
        "schedule_feasibility": 1.0,
        "abandonment_rate_ucb95": 0.05,
        "service_level_lcb95": 0.9,
        "p95_wait_seconds_ucb95": 120.0,
        "hard_violations": 0,
        "recourse_action_rate": 0.1,
        "recourse_cost_share": 0.1,
        "flow_conservation": True,
    }


def _thresholds() -> dict[str, float | bool]:
    return {
        "max_interval_coverage_error": 0.1,
        "max_worst_slice_wape": 0.5,
        "min_peak_q90_coverage": 0.65,
        "min_incident_q90_coverage": 0.8,
        "min_peak_sample_count": 10,
        "min_incident_sample_count": 5,
        "max_fixed_origin_wape": 0.5,
        "max_fixed_origin_coverage_error": 0.1,
        "max_scenario_coherence_error": 1e-8,
        "require_capacity_plan_success": True,
        "min_strategic_tactical_alignment": 1.0,
        "min_schedule_feasibility": 1.0,
        "max_abandonment_rate": 0.2,
        "min_service_level_lcb95": 0.8,
        "max_p95_wait_seconds": 600.0,
        "max_recourse_action_rate": 0.4,
        "max_recourse_cost_share": 0.4,
        "max_hard_violations": 0,
    }


def test_release_gate_passes_good_metrics():
    decision = evaluate_release_gate(_forecast_metrics(), _decision_metrics(), _thresholds())
    assert decision.status == "PASS"


def test_release_gate_rejects_incoherent_scenarios_and_failed_capacity_plan():
    forecast = _forecast_metrics()
    forecast["scenario_coherence_max_abs_error"] = 0.5
    operational = _decision_metrics()
    operational["capacity_plan_success"] = False
    decision = evaluate_release_gate(forecast, operational, _thresholds())
    assert decision.status == "ITERATE"
    failed = {check.name for check in decision.checks if not check.passed}
    assert {"scenario_coherence", "capacity_plan_success"}.issubset(failed)


def test_release_gate_fails_closed_when_blocking_evidence_is_missing():
    forecast = _forecast_metrics()
    forecast.pop("peak_q90_coverage")
    with pytest.raises(ValueError, match="peak_q90_coverage"):
        evaluate_release_gate(forecast, _decision_metrics(), _thresholds())


def test_release_gate_rejects_missing_tail_sample_support_and_wait_uncertainty():
    forecast = _forecast_metrics()
    forecast["incident_sample_count"] = 0
    operational = _decision_metrics()
    operational["p95_wait_seconds_ucb95"] = 900.0
    operational["hard_violations"] = 1
    decision = evaluate_release_gate(forecast, operational, _thresholds())
    failed = {check.name for check in decision.checks if not check.passed}
    assert decision.status == "ITERATE"
    assert {
        "incident_sample_count",
        "p95_wait_seconds_ucb95",
        "hard_violations",
    }.issubset(failed)


def test_release_gate_allows_only_alignment_shortfall_with_bounded_recourse():
    decision_metrics = _decision_metrics()
    decision_metrics["strategic_tactical_alignment"] = 0.75

    decision = evaluate_release_gate(
        _forecast_metrics(),
        decision_metrics,
        _thresholds(),
        recourse_required=True,
        post_recourse_eligible=True,
    )

    assert decision.status == "PASS_WITH_RECOURSE"
    assert [check.name for check in decision.checks if not check.passed] == [
        "strategic_tactical_alignment"
    ]


def test_release_gate_does_not_override_alignment_without_actual_recourse():
    decision_metrics = _decision_metrics()
    decision_metrics["strategic_tactical_alignment"] = 0.75

    decision = evaluate_release_gate(
        _forecast_metrics(),
        decision_metrics,
        _thresholds(),
        recourse_required=False,
        post_recourse_eligible=True,
    )

    assert decision.status == "ITERATE"


def test_release_gate_does_not_override_alignment_when_another_hard_gate_fails():
    decision_metrics = _decision_metrics()
    decision_metrics["strategic_tactical_alignment"] = 0.75
    decision_metrics["hard_violations"] = 1

    decision = evaluate_release_gate(
        _forecast_metrics(),
        decision_metrics,
        _thresholds(),
        recourse_required=True,
        post_recourse_eligible=True,
    )

    assert decision.status == "ITERATE"
