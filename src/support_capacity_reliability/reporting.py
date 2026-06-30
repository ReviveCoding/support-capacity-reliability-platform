from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from support_capacity_reliability.utils import ensure_dir, write_json


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


def write_reports(
    output_dir: Path,
    summary: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
    test_metrics: pd.DataFrame,
    slice_report: pd.DataFrame,
    policy_results: pd.DataFrame,
    capacity_plan: pd.DataFrame,
    release_decision: dict[str, Any],
) -> None:
    reports = ensure_dir(output_dir / "reports")
    write_json(output_dir / "run_summary.json", summary)
    write_json(reports / "release_gate_decision.json", release_decision)

    validation = f"""# Validation Report

## Run

- Project: `{summary["project_name"]}`
- Seed: `{summary["seed"]}`
- Selected forecast variant: `{summary["selected_variant"]}`
- RCWE applied to selected variant: `{summary.get("selected_rcwe_applied", False)}`
- Selected policy from tuning replay: `{summary["selected_policy_from_replay"]}`
- Deployed policy after intraday recourse: `{summary["deployed_policy"]}`
- Release status: **{release_decision["status"]}**

## Calibration leaderboard

{_markdown_table(validation_leaderboard)}

## Frozen-test metrics

{_markdown_table(test_metrics)}

## Slice diagnostics

{_markdown_table(slice_report, 50)}
"""
    (reports / "validation_report.md").write_text(validation, encoding="utf-8")

    diag = summary.get("decision_diagnostics", {})
    decision = f"""# Workforce Decision Memo

## Decision

The pipeline selected **{summary["selected_policy_from_replay"]}** on a policy-tuning replay that precedes the frozen evaluation horizon. It then evaluated two-stage frozen-scenario recourse and deployed **{summary["deployed_policy"]}** on the final frozen replay. The forecast model was not promoted on predictive accuracy alone; queue performance, schedule feasibility, service level, abandonment, and simulated cost were considered. Policies with failed hard gates are excluded from initial selection. A `PASS_WITH_RECOURSE` status means the base schedule did not pass by itself and the reported recovery depends on the explicitly labeled recourse stage.

## Decision-science diagnostics

- Policy-selection protocol: `{diag.get("policy_selection_protocol", "n/a")}`
- Tuning origin: `{diag.get("policy_tuning_origin", "n/a")}`
- Frozen evaluation origin: `{diag.get("frozen_evaluation_origin", "n/a")}`
- Hindsight-best frozen candidate: `{diag.get("hindsight_best_candidate_policy", "n/a")}`
- Selected-policy cost gap versus hindsight best: `{diag.get("selection_cost_gap_vs_hindsight_best", "n/a")}`
- Point forecast incremental cost versus fixed ratio: `{diag.get("incremental_cost_point_minus_fixed", "n/a")}`
- Probabilistic policy incremental cost versus point forecast: `{diag.get("incremental_cost_probabilistic_minus_point", "n/a")}`

Positive incremental cost means the added policy was more expensive in frozen replay; negative values mean cost reduction. The realized-offered reference is a staffing reference, not a globally optimal oracle, so EVPI and VSS are not claimed.

## Policy comparison

{_markdown_table(policy_results)}

## Strategic capacity plan

{_markdown_table(capacity_plan)}

## Claim boundary

All operational outcomes in this report are offline simulated outcomes on synthetic ACD and workforce records. They do not represent a live contact-center deployment or actual customer impact.
"""
    (reports / "decision_memo.md").write_text(decision, encoding="utf-8")

    model_selection = f"""# Model and Decision Selection Report

## Forecast selection

Selected forecast variant: `{summary["selected_variant"]}`. RCWE selected: `{summary.get("selected_rcwe_applied", False)}`.

The validation score combines WAPE, interval coverage error, and interval width so an advanced model is not promoted solely because it is newer or more complex.

## Decision selection

Selected policy from tuning replay: `{summary["selected_policy_from_replay"]}`.

Deployed policy after intraday recourse: `{summary["deployed_policy"]}`.

Policy selection uses a replay horizon that ends before the final frozen evaluation horizon. The report preserves cases where a simpler policy wins, because rejecting unnecessary complexity is part of the release criterion.

## Decision diagnostics

{_markdown_table(pd.DataFrame([diag]))}
"""
    (reports / "model_selection_report.md").write_text(model_selection, encoding="utf-8")

    model_card = f"""# Model Card

## Intended use

Offline contact-workload forecasting and workforce decision research using public-data adapters and synthetic operational data.

## Model ladder

Seasonal heuristic, negative-binomial/Poisson-style statistical regression, LightGBM quantile regression, a GPU-ready PyTorch quantile network, and an optional official Chronos-2 adapter.

## Selected variant

`{summary["selected_variant"]}`

## Reliability controls

- temporal splitting and frozen test
- quantile ordering
- calibration partition
- Reference-Conditioned Workload Envelope (RCWE)
- low-reference-support interval inflation
- regime, region, and skill slices
- decision-aware release gate

## Limitations

The smoke run uses generated data. Public NYC 311 data is a workload-pattern proxy, not an ACD contact-center log. Chronos-2 is optional and is not downloaded by smoke mode.
"""
    (reports / "model_card.md").write_text(model_card, encoding="utf-8")

    twin_card = """# Digital Twin Card

## Supported processes

- time-varying voice arrivals
- multi-skill agent matching
- heterogeneous service time and agent proficiency
- customer patience and abandonment
- shift-specific staffing
- queue wait, service level, utilization, and flow accounting

## Validation

- served + abandoned = offered
- deterministic replay under fixed seed
- Erlang-C and Erlang-A analytical baselines
- monotonicity tests for arrival rate, service time, and agent capacity

## Unsupported processes

- production telephony behavior
- chat concurrency in canonical smoke mode
- exact customer redial timing inside the queue simulator
- live routing rules or customer segmentation
"""
    (reports / "digital_twin_card.md").write_text(twin_card, encoding="utf-8")

    limitations = """# Limitations and Claim Discipline

1. Synthetic ACD and workforce records are used for canonical reproducibility.
2. NYC 311 and CFPB adapters are public workload proxies, not proprietary contact-center logs.
3. Operational metrics are simulated and must not be described as live business impact.
4. The Chronos-2 adapter requires optional dependencies and model access; smoke mode does not download weights.
5. The tactical scheduler assigns one primary skill per shift, while the digital twin permits scheduled multi-skilled agents to flex across their eligible skills.
6. Strategic and tactical decisions are simplified portfolio evidence, not a complete enterprise workforce-management product.
7. The recourse experiment observes the frozen realized offered-demand scenario and measures recoverability; it is not a causal real-time rollout.
"""
    (reports / "limitations.md").write_text(limitations, encoding="utf-8")
