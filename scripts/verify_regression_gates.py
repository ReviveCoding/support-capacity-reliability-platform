"""Compare deterministic candidate outcomes with the frozen v1.4.0 baseline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def check(name: str, passed: bool, baseline, candidate, tolerance=None) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "baseline": baseline,
        "candidate": candidate,
        "tolerance": tolerance,
    }


def main() -> None:
    baseline = load("reports/baseline/baseline_metrics.json")
    smoke = load("outputs/smoke/run_summary.json")
    stress = load("outputs/stress_insufficient_workforce/run_summary.json")
    checks = [
        check(
            "canonical_release_status",
            smoke["release_status"] == baseline["canonical"]["release_status"],
            baseline["canonical"]["release_status"],
            smoke["release_status"],
        ),
        check(
            "stress_release_status",
            stress["release_status"] == baseline["stress"]["release_status"],
            baseline["stress"]["release_status"],
            stress["release_status"],
        ),
        check(
            "canonical_one_step_wape",
            smoke["selected_forecast_metrics"]["wape"]
            <= baseline["canonical"]["one_step_wape"] + 1e-12,
            baseline["canonical"]["one_step_wape"],
            smoke["selected_forecast_metrics"]["wape"],
            1e-12,
        ),
        check(
            "canonical_fixed_origin_wape",
            smoke["fixed_origin_forecast_metrics"]["wape"]
            <= baseline["canonical"]["fixed_origin_wape"] + 1e-12,
            baseline["canonical"]["fixed_origin_wape"],
            smoke["fixed_origin_forecast_metrics"]["wape"],
            1e-12,
        ),
        check(
            "canonical_total_cost",
            smoke["selected_decision_metrics"]["total_cost"]
            <= baseline["canonical"]["total_cost"] + 1e-8,
            baseline["canonical"]["total_cost"],
            smoke["selected_decision_metrics"]["total_cost"],
            1e-8,
        ),
        check(
            "canonical_hard_violations",
            int(smoke["selected_decision_metrics"]["hard_violations"]) == 0,
            0,
            smoke["selected_decision_metrics"]["hard_violations"],
        ),
        check(
            "stress_structural_violation_retained",
            int(stress["selected_decision_metrics"]["hard_violations"]) >= 1,
            baseline["stress"]["hard_violations"],
            stress["selected_decision_metrics"]["hard_violations"],
        ),
        check(
            "canonical_bundle_replay",
            float(smoke["model_bundle"]["verification_max_abs_error"]) <= 1e-9,
            1e-9,
            smoke["model_bundle"]["verification_max_abs_error"],
            1e-9,
        ),
        check(
            "stress_bundle_replay",
            float(stress["model_bundle"]["verification_max_abs_error"]) <= 1e-9,
            1e-9,
            stress["model_bundle"]["verification_max_abs_error"],
            1e-9,
        ),
    ]
    report = {
        "schema_version": "1.0",
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports" / "candidate" / "regression_gates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
