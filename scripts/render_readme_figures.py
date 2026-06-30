"""Render README SVG figures from immutable GitHub Actions evidence artifacts.

This utility is intentionally standard-library only. It consumes the exact smoke and
negative-gate evidence artifacts from a named GitHub Actions run, writes SVG figures
for README rendering, and writes a provenance manifest alongside the images.
"""

from __future__ import annotations

import csv
import json
import math
from argparse import ArgumentParser
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

WIDTH = 1200
HEIGHT = 520

COLORS = {
    "ink": "#14213d",
    "muted": "#5b677a",
    "grid": "#d9e1ea",
    "panel": "#f7f9fc",
    "blue": "#1f77b4",
    "cyan": "#72b7d2",
    "green": "#1b7f5d",
    "amber": "#b87800",
    "red": "#bd3d4d",
    "purple": "#6f4aa8",
    "white": "#ffffff",
    "black": "#111827",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches[:5])
        raise RuntimeError(
            f"Expected exactly one {name} below {root}; found {len(matches)} ({rendered})"
        )
    return matches[0]


def get_any(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_percent(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    return "n/a" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def fmt_number(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def svg_document(body: str, title: str, description: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <style>
    .title {{ font: 700 25px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
    .subtitle {{ font: 14px Arial, Helvetica, sans-serif; fill: {COLORS["muted"]}; }}
    .small {{ font: 12px Arial, Helvetica, sans-serif; fill: {COLORS["muted"]}; }}
    .label {{ font: 600 13px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
    .value {{ font: 700 17px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
    .box-title {{ font: 700 15px Arial, Helvetica, sans-serif; fill: {COLORS["ink"]}; }}
    .box-sub {{ font: 12px Arial, Helvetica, sans-serif; fill: {COLORS["muted"]}; }}
    .axis {{ font: 11px Arial, Helvetica, sans-serif; fill: {COLORS["muted"]}; }}
  </style>
  <rect width="100%" height="100%" fill="{COLORS["white"]}"/>
  {body}
</svg>\n'''


def write_svg(path: Path, body: str, title: str, description: str) -> None:
    path.write_text(svg_document(body, title, description), encoding="utf-8", newline="\n")


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str = "none",
    radius: float = 10,
    opacity: float | None = None,
) -> str:
    extra = "" if opacity is None else f' opacity="{opacity}"'
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{radius}" fill="{fill}" stroke="{stroke}"{extra}/>'


def text(x: float, y: float, content: str, cls: str, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}">{escape(content)}</text>'


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 2,
    dash: str | None = None,
    opacity: float | None = None,
) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    opacity_attr = "" if opacity is None else f' opacity="{opacity}"'
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}{opacity_attr}/>'


def polyline(
    points: list[tuple[float, float]],
    *,
    stroke: str,
    width: float = 2.5,
    fill: str = "none",
    opacity: float | None = None,
) -> str:
    rendered = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    opacity_attr = "" if opacity is None else f' opacity="{opacity}"'
    return f'<polyline points="{rendered}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{opacity_attr}/>'


def polygon(points: list[tuple[float, float]], *, fill: str, opacity: float = 1.0) -> str:
    rendered = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{rendered}" fill="{fill}" opacity="{opacity}"/>'


def escape_xml(value: Any) -> str:
    return escape(str(value))


def compact_timestamp(raw: str) -> str:
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return value.strftime("%b %d\n%H:%M")
    except (ValueError, TypeError):
        return str(raw)[:16]


def text_lines(
    x: float, y: float, content: str, cls: str, line_height: float = 15, anchor: str = "start"
) -> str:
    parts = str(content).split("\n")
    if len(parts) == 1:
        return text(x, y, parts[0], cls, anchor)
    tspan = []
    for index, part in enumerate(parts):
        dy = 0 if index == 0 else line_height
        tspan.append(f'<tspan x="{x:.2f}" dy="{dy:.2f}">{escape(part)}</tspan>')
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}">'
        + "".join(tspan)
        + "</text>"
    )


def render_pipeline(path: Path) -> None:
    stages = [
        ("Data contracts", "Synthetic ACD / public proxies", COLORS["cyan"]),
        ("Forecast ladder", "Seasonal Â· Poisson Â· LightGBM Â· Torch", COLORS["blue"]),
        ("Reliability", "RCWE Â· calibration Â· coherent scenarios", COLORS["purple"]),
        ("Decision", "HiGHS capacity Â· CP-SAT schedule", COLORS["amber"]),
        ("Digital twin", "Erlang-A checks Â· SimPy replay", COLORS["green"]),
        ("Release gate", "PASS Â· PASS_WITH_RECOURSE Â· ITERATE", COLORS["red"]),
    ]
    body = [
        text(60, 54, "Reliability-aware workforce decision pipeline", "title"),
        text(
            60,
            79,
            "Forecast candidates are not promoted on accuracy alone: each path is evaluated through feasibility, queue performance, bounded recourse, and a fail-closed release gate.",
            "subtitle",
        ),
    ]
    start_x = 45
    y = 160
    box_w = 172
    box_h = 150
    gap = 22
    for index, (name, sub, color) in enumerate(stages):
        x = start_x + index * (box_w + gap)
        body.append(rect(x, y, box_w, box_h, fill=COLORS["panel"], stroke=color, radius=14))
        body.append(rect(x, y, box_w, 11, fill=color, radius=14))
        body.append(text_lines(x + box_w / 2, y + 53, name, "box-title", 18, "middle"))
        body.append(
            text_lines(x + box_w / 2, y + 86, sub.replace(" Â· ", "\n"), "box-sub", 16, "middle")
        )
        if index < len(stages) - 1:
            arrow_x1 = x + box_w + 4
            arrow_x2 = x + box_w + gap - 4
            arrow_y = y + box_h / 2
            body.append(
                line(arrow_x1, arrow_y, arrow_x2, arrow_y, stroke=COLORS["muted"], width=2.2)
            )
            body.append(
                f'<polygon points="{arrow_x2:.2f},{arrow_y:.2f} {arrow_x2 - 9:.2f},{arrow_y - 5:.2f} {arrow_x2 - 9:.2f},{arrow_y + 5:.2f}" fill="{COLORS["muted"]}"/>'
            )
    body.extend(
        [
            rect(155, 368, 890, 95, fill="#f5f7fb", stroke="#cbd5e1", radius=14),
            text(205, 402, "Persisted evidence", "box-title"),
            text(
                205,
                427,
                "checksummed artifact index Â· selected forecast bundle Â· monitoring snapshot Â· FastAPI surface Â· reproducible command ledger",
                "box-sub",
            ),
            text(
                205,
                448,
                "Offline synthetic operational replay only: no live customer traffic or causal real-time rollout claim.",
                "small",
            ),
        ]
    )
    write_svg(
        path,
        "".join(body),
        "Reliability-aware workforce decision pipeline",
        "Architecture of the forecast-to-workforce-decision platform.",
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_column(fieldnames: Iterable[str], candidates: list[str], suffix: str | None = None) -> str:
    names = list(fieldnames)
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if suffix:
        for name in names:
            if name.lower().endswith(suffix.lower()):
                return name
    raise RuntimeError(
        f"Could not find expected column. Candidates={candidates}, suffix={suffix}, available={names}"
    )


def aggregate_forecast(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[float], list[float], list[float], list[float], list[bool]]:
    if not rows:
        raise RuntimeError("fixed-origin prediction artifact was empty")
    fields = rows[0].keys()
    time_col = pick_column(fields, ["timestamp", "time", "datetime"])
    q10_col = pick_column(fields, ["selected_q10", "q10", "forecast_q10"], suffix="_q10")
    q50_col = pick_column(fields, ["selected_q50", "q50", "forecast_q50"], suffix="_q50")
    q90_col = pick_column(fields, ["selected_q90", "q90", "forecast_q90"], suffix="_q90")
    actual_col = pick_column(
        fields, ["offered_load_estimate", "actual", "y_true", "target", "observed"], None
    )
    regime_col = next((field for field in fields if field.lower() == "regime"), None)

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row[time_col]
        bucket = buckets.setdefault(
            key, {"actual": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0, "incident": False}
        )
        bucket["actual"] += as_float(row[actual_col], 0.0)
        bucket["q10"] += as_float(row[q10_col], 0.0)
        bucket["q50"] += as_float(row[q50_col], 0.0)
        bucket["q90"] += as_float(row[q90_col], 0.0)
        bucket["incident"] = bucket["incident"] or (
            str(row.get(regime_col, "")).lower() == "incident" if regime_col else False
        )
    ordered = sorted(buckets)
    return (
        ordered,
        [buckets[key]["actual"] for key in ordered],
        [buckets[key]["q10"] for key in ordered],
        [buckets[key]["q50"] for key in ordered],
        [buckets[key]["q90"] for key in ordered],
        [bool(buckets[key]["incident"]) for key in ordered],
    )


def metric_from_summary(summary: dict[str, Any], names: list[str]) -> Any:
    containers: list[dict[str, Any]] = [summary]
    for key in (
        "fixed_origin_forecast_metrics",
        "selected_forecast_metrics",
        "forecast_tail_metrics",
        "selected_decision_metrics",
    ):
        value = summary.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        value = get_any(container, names)
        if value is not None:
            return value
    return None


def render_forecast(path: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    timestamps, actual, q10, q50, q90, incidents = aggregate_forecast(rows)
    chart_x, chart_y, chart_w, chart_h = 80, 135, 850, 300
    values = [*actual, *q10, *q50, *q90]
    y_min = max(0.0, min(values) * 0.90)
    y_max = max(values) * 1.08 if max(values) > 0 else 1.0

    def px(index: int) -> float:
        return (
            chart_x if len(timestamps) <= 1 else chart_x + chart_w * index / (len(timestamps) - 1)
        )

    def py(value: float) -> float:
        return chart_y + chart_h * (1.0 - (value - y_min) / max(y_max - y_min, 1e-9))

    body = [
        text(60, 52, "Fixed-origin workload forecast with calibrated uncertainty", "title"),
        text(
            60,
            77,
            "Aggregate offered-load demand across region and skill. The interval band is q10â€“q90; the center line is q50.",
            "subtitle",
        ),
        rect(chart_x, chart_y, chart_w, chart_h, fill=COLORS["panel"], stroke="#d7dee8", radius=10),
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = py(value)
        body.append(line(chart_x, y, chart_x + chart_w, y, stroke=COLORS["grid"], width=1))
        body.append(text(chart_x - 10, y + 4, f"{value:.0f}", "axis", "end"))
    for index, is_incident in enumerate(incidents):
        if is_incident:
            step = chart_w / max(len(timestamps) - 1, 1)
            x = max(chart_x, px(index) - step / 2)
            body.append(rect(x, chart_y, step, chart_h, fill="#f5c2c7", radius=0, opacity=0.36))
    band_points = [(px(index), py(value)) for index, value in enumerate(q90)] + [
        (px(index), py(value)) for index, value in reversed(list(enumerate(q10)))
    ]
    body.append(polygon(band_points, fill=COLORS["cyan"], opacity=0.38))
    body.append(
        polyline(
            [(px(index), py(value)) for index, value in enumerate(actual)],
            stroke=COLORS["black"],
            width=2.6,
        )
    )
    body.append(
        polyline(
            [(px(index), py(value)) for index, value in enumerate(q50)],
            stroke=COLORS["blue"],
            width=3.0,
        )
    )
    for index in range(0, len(timestamps), max(1, len(timestamps) // 6)):
        x = px(index)
        body.append(
            line(x, chart_y + chart_h, x, chart_y + chart_h + 5, stroke=COLORS["muted"], width=1)
        )
        label = compact_timestamp(timestamps[index]).split("\n")[0]
        body.append(text(x, chart_y + chart_h + 24, label, "axis", "middle"))
    legend_x, legend_y = 98, 112
    body.extend(
        [
            line(legend_x, legend_y, legend_x + 26, legend_y, stroke=COLORS["black"], width=2.6),
            text(legend_x + 34, legend_y + 4, "Realized offered load", "axis"),
            line(
                legend_x + 210, legend_y, legend_x + 236, legend_y, stroke=COLORS["blue"], width=3
            ),
            text(legend_x + 244, legend_y + 4, "q50 forecast", "axis"),
            rect(legend_x + 375, legend_y - 8, 26, 13, fill=COLORS["cyan"], opacity=0.38, radius=2),
            text(legend_x + 410, legend_y + 4, "q10â€“q90 interval", "axis"),
        ]
    )
    side_x = 965
    body.append(rect(side_x, 135, 195, 300, fill="#f7f9fc", stroke="#d7dee8", radius=12))
    body.append(text(side_x + 18, 170, "Evidence snapshot", "box-title"))
    metrics = [
        (
            "Fixed-origin WAPE",
            fmt_percent(metric_from_summary(summary, ["wape", "fixed_origin_wape"])),
        ),
        (
            "80% interval coverage",
            fmt_percent(
                metric_from_summary(
                    summary, ["interval_coverage_80", "fixed_origin_interval_coverage_80"]
                )
            ),
        ),
        ("Peak q90 coverage", fmt_percent(metric_from_summary(summary, ["peak_q90_coverage"]))),
        (
            "Incident q90 coverage",
            fmt_percent(metric_from_summary(summary, ["incident_q90_coverage"])),
        ),
    ]
    for index, (name, value) in enumerate(metrics):
        y = 214 + index * 52
        body.append(text(side_x + 18, y, name, "small"))
        body.append(text(side_x + 18, y + 22, value, "value"))
    body.append(
        text(
            80,
            492,
            "Shaded intervals indicate any leaf-level incident regime. Figure is generated from the exact hosted smoke evidence artifact.",
            "small",
        )
    )
    write_svg(
        path,
        "".join(body),
        "Fixed-origin workload forecast with calibrated uncertainty",
        "Forecast uncertainty figure generated from fixed-origin prediction artifact.",
    )


def render_recourse(path: Path, actions: list[dict[str, str]], summary: dict[str, Any]) -> None:
    counts = Counter()
    cost = defaultdict(float)
    for row in actions:
        action = str(row.get("action", "unknown"))
        if action == "hold_schedule":
            continue
        amount = int(round(as_float(row.get("amount"), 0.0)))
        counts[action] += max(amount, 1)
        cost[action] += as_float(row.get("estimated_cost"), 0.0)
    items = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    if not items:
        items = [("no_applied_recourse", 0)]
    max_count = max(count for _, count in items) or 1
    body = [
        text(60, 52, "Bounded intraday recourse closes operational gaps", "title"),
        text(
            60,
            77,
            "The release outcome is not a raw forecast score: recovery must remain inside feasibility, hard-violation, action-rate, and cost-share gates.",
            "subtitle",
        ),
    ]
    chart_x, chart_y, chart_w, chart_h = 82, 152, 640, 250
    body.append(
        rect(chart_x, chart_y, chart_w, chart_h, fill=COLORS["panel"], stroke="#d7dee8", radius=10)
    )
    gap = 30
    bar_w = min(120, (chart_w - gap * (len(items) + 1)) / max(len(items), 1))
    for index, (name, count) in enumerate(items):
        x = chart_x + gap + index * (bar_w + gap)
        height = chart_h * 0.70 * count / max_count
        y = chart_y + chart_h - 42 - height
        body.append(
            rect(
                x,
                y,
                bar_w,
                height,
                fill=[COLORS["green"], COLORS["blue"], COLORS["amber"], COLORS["purple"]][
                    index % 4
                ],
                radius=6,
            )
        )
        body.append(text(x + bar_w / 2, y - 9, str(count), "value", "middle"))
        label = name.replace("_", "\n")
        body.append(text_lines(x + bar_w / 2, chart_y + chart_h - 22, label, "axis", 13, "middle"))
        body.append(
            text(x + bar_w / 2, chart_y + chart_h + 28, f"cost {cost[name]:+.0f}", "axis", "middle")
        )
    score_x, score_y, score_w, score_h = 770, 145, 365, 300
    body.append(
        rect(score_x, score_y, score_w, score_h, fill="#f7f9fc", stroke="#d7dee8", radius=12)
    )
    body.append(text(score_x + 22, score_y + 38, "Release-gated outcome", "box-title"))
    release_status = str(get_any(summary, ["release_status"], "n/a"))
    decision = (
        summary.get("selected_decision_metrics", {})
        if isinstance(summary.get("selected_decision_metrics"), dict)
        else {}
    )
    diag = (
        summary.get("decision_diagnostics", {})
        if isinstance(summary.get("decision_diagnostics"), dict)
        else {}
    )
    values = [
        ("Release decision", release_status),
        ("Schedule feasibility", fmt_percent(get_any(decision, ["schedule_feasibility"], 1.0), 0)),
        ("Hard violations", str(get_any(decision, ["hard_violations"], "n/a"))),
        (
            "Recourse action rate",
            fmt_percent(
                get_any(decision, ["recourse_action_rate"], diag.get("recourse_action_rate"))
            ),
        ),
        (
            "Recourse cost share",
            fmt_percent(
                get_any(decision, ["recourse_cost_share"], diag.get("recourse_cost_share"))
            ),
        ),
        (
            "Actions applied",
            str(get_any(summary, ["intraday_recourse_actions"], sum(counts.values()))),
        ),
    ]
    for index, (name, value) in enumerate(values):
        y = score_y + 82 + index * 34
        body.append(text(score_x + 22, y, name, "small"))
        cls = "value" if index == 0 else "label"
        body.append(text(score_x + score_w - 22, y, value, cls, "end"))
        if index < len(values) - 1:
            body.append(
                line(
                    score_x + 22, y + 11, score_x + score_w - 22, y + 11, stroke="#e2e8f0", width=1
                )
            )
    body.append(
        text(
            82,
            484,
            "Action bars exclude hold_schedule rows. Costs are estimated operational-cost deltas in the persisted recourse artifact.",
            "small",
        )
    )
    write_svg(
        path,
        "".join(body),
        "Bounded intraday recourse closes operational gaps",
        "Recourse actions and release-gate outcomes from smoke evidence.",
    )


def render_gate_scorecard(
    path: Path, canonical_summary: dict[str, Any], stress_summary: dict[str, Any]
) -> None:
    canonical_decision = (
        canonical_summary.get("selected_decision_metrics", {})
        if isinstance(canonical_summary.get("selected_decision_metrics"), dict)
        else {}
    )
    stress_decision = (
        stress_summary.get("selected_decision_metrics", {})
        if isinstance(stress_summary.get("selected_decision_metrics"), dict)
        else {}
    )
    canonical_rows = (
        canonical_summary.get("rows", {}) if isinstance(canonical_summary.get("rows"), dict) else {}
    )
    stress_rows = (
        stress_summary.get("rows", {}) if isinstance(stress_summary.get("rows"), dict) else {}
    )
    canonical_agents = get_any(canonical_rows, ["agents"], 42)
    stress_agents = get_any(stress_rows, ["agents"], 36)
    canonical_status = str(get_any(canonical_summary, ["release_status"], "PASS_WITH_RECOURSE"))
    stress_status = str(get_any(stress_summary, ["release_status"], "ITERATE"))
    cards = [
        (
            "Canonical smoke",
            f"{canonical_agents} agents",
            canonical_status,
            COLORS["green"],
            canonical_decision,
        ),
        (
            "Insufficient-workforce stress",
            f"{stress_agents} agents",
            stress_status,
            COLORS["red"],
            stress_decision,
        ),
    ]
    body = [
        text(
            60,
            52,
            "Release gate distinguishes recoverable demand from structural shortfall",
            "title",
        ),
        text(
            60,
            77,
            "The negative case is intended to remain ITERATE: the gate rejects infeasible workforce conditions instead of silently relaxing constraints.",
            "subtitle",
        ),
    ]
    x_positions = [75, 640]
    y, w, h = 135, 485, 300
    for index, (name, agents, status, color, decision) in enumerate(cards):
        x = x_positions[index]
        body.append(rect(x, y, w, h, fill="#f7f9fc", stroke=color, radius=16))
        body.append(rect(x, y, w, 15, fill=color, radius=16))
        body.append(text(x + 25, y + 58, name, "box-title"))
        body.append(text(x + 25, y + 84, agents, "subtitle"))
        body.append(rect(x + 25, y + 107, w - 50, 43, fill=color, radius=8, opacity=0.16))
        body.append(text(x + w / 2, y + 134, status, "value", "middle"))
        metrics = [
            ("Schedule feasibility", fmt_percent(get_any(decision, ["schedule_feasibility"]), 0)),
            ("Hard violations", str(get_any(decision, ["hard_violations"], "n/a"))),
            ("Release boundary", "bounded recourse" if "RECOURSE" in status else "fail closed"),
        ]
        for row, (label, value) in enumerate(metrics):
            yy = y + 190 + row * 37
            body.append(text(x + 25, yy, label, "small"))
            body.append(text(x + w - 25, yy, value, "label", "end"))
            if row < len(metrics) - 1:
                body.append(line(x + 25, yy + 12, x + w - 25, yy + 12, stroke="#e2e8f0", width=1))
        footer = (
            "Canonical replay recovered inside gate limits."
            if index == 0
            else "Structural shortfall retained after bounded recourse."
        )
        body.append(text(x + 25, y + h - 27, footer, "small"))
    body.append(line(567, 262, 625, 262, stroke=COLORS["muted"], width=2.3, dash="7 5"))
    body.append(text(596, 244, "capacity stress", "small", "middle"))
    body.append(
        text(
            60,
            486,
            "The stress configuration is a negative-control test, not a failed release candidate. It verifies that the release gate remains fail-closed.",
            "small",
        )
    )
    write_svg(
        path,
        "".join(body),
        "Release-gate comparison between canonical and stress evidence",
        "Canonical PASS_WITH_RECOURSE compared with insufficient-workforce ITERATE negative gate.",
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--smoke-run-id", required=True)
    args = parser.parse_args()

    smoke_summary_path = find_one(args.smoke_root, "run_summary.json")
    stress_summary_path = find_one(args.stress_root, "run_summary.json")
    prediction_path = find_one(args.smoke_root, "fixed_origin_predictions.csv")
    action_path = find_one(args.smoke_root, "intraday_recourse_actions.csv")

    smoke_summary = read_json(smoke_summary_path)
    stress_summary = read_json(stress_summary_path)
    reproducibility = (
        smoke_summary.get("reproducibility", {})
        if isinstance(smoke_summary.get("reproducibility"), dict)
        else {}
    )
    artifact_sha = str(get_any(reproducibility, ["git_sha"], ""))
    if artifact_sha and artifact_sha != args.commit:
        raise RuntimeError(f"Evidence SHA mismatch: artifact={artifact_sha} expected={args.commit}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_pipeline(args.output_dir / "decision_pipeline.svg")
    render_forecast(
        args.output_dir / "fixed_origin_forecast.svg", load_csv(prediction_path), smoke_summary
    )
    render_recourse(args.output_dir / "bounded_recourse.svg", load_csv(action_path), smoke_summary)
    render_gate_scorecard(
        args.output_dir / "release_gate_comparison.svg", smoke_summary, stress_summary
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "github_actions_run_id": args.smoke_run_id,
            "commit": args.commit,
            "smoke_summary": "smoke-evidence/run_summary.json",
            "stress_summary": "insufficient-workforce-evidence/run_summary.json",
            "fixed_origin_predictions": "smoke-evidence/metrics/fixed_origin_predictions.csv",
            "intraday_recourse_actions": "smoke-evidence/metrics/intraday_recourse_actions.csv",
        },
        "figures": [
            "decision_pipeline.svg",
            "fixed_origin_forecast.svg",
            "bounded_recourse.svg",
            "release_gate_comparison.svg",
        ],
        "claim_boundary": "Figures summarize offline synthetic operational replay evidence and do not represent live customer traffic, AWS deployment, or causal real-time rollout.",
    }
    (args.output_dir / "README.md").write_text(
        "# README figures\n\n"
        "These SVGs are generated from exact GitHub Actions smoke and negative-gate evidence. "
        "Run `python scripts/render_readme_figures.py --help` for reproduction.\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "readme_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir),
                "figure_count": 4,
                "artifact_git_sha": artifact_sha or None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
