"""Create and validate the release-candidate handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [
    Path(".github"),
    Path("configs"),
    Path("constraints"),
    Path("docs"),
    Path("scripts"),
    Path("src"),
    Path("tests"),
]
SOURCE_FILES = [
    Path(".dockerignore"),
    Path(".gitignore"),
    Path("Dockerfile"),
    Path("LICENSE"),
    Path("Makefile"),
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("RELEASE_NOTES.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("requirements-dev.txt"),
]
GENERATED_SOURCE_EXCLUSIONS = {Path("docs/release_qualification.md")}

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "outputs",
    "reports",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_paths() -> list[Path]:
    paths: set[Path] = set()
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if path.is_file():
            paths.add(path)
    for relative in SOURCE_ROOTS:
        base = ROOT / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if rel in GENERATED_SOURCE_EXCLUSIONS:
                continue
            if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in rel.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def build_source_manifest() -> dict[str, Any]:
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths()
    ]
    return {
        "schema_version": "1.0",
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "source_fingerprint": stable_hash(entries),
        "files": entries,
    }


def source_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_map = {
        row["path"]: row
        for row in baseline.get("files", [])
        if not any(part in EXCLUDED_PARTS for part in Path(row["path"]).parts)
    }
    candidate_map = {row["path"]: row for row in candidate["files"]}
    added = sorted(set(candidate_map) - set(baseline_map))
    deleted = sorted(set(baseline_map) - set(candidate_map))
    modified = sorted(
        path
        for path in set(candidate_map) & set(baseline_map)
        if candidate_map[path]["sha256"] != baseline_map[path]["sha256"]
    )
    payload = {"added": added, "modified": modified, "deleted": deleted}
    payload["diff_checksum"] = stable_hash(payload)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_state() -> dict[str, Any]:
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "commit": None, "dirty_tree": None}
    if inside.returncode != 0:
        return {"available": False, "commit": None, "dirty_tree": None}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"available": True, "commit": commit, "dirty_tree": bool(status.strip())}


def dependency_snapshot() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    snapshot_path = ROOT / "reports" / "candidate" / "pip_freeze.txt"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(freeze.stdout, encoding="utf-8")
    contracts = []
    for relative in [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "constraints/qualification-py313.txt",
    ]:
        path = ROOT / relative
        if path.is_file():
            contracts.append(file_record(path))
    return {
        "pip_freeze": file_record(snapshot_path),
        "dependency_contracts": contracts,
        "lockfile": file_record(ROOT / "constraints" / "qualification-py313.txt"),
        "lockfile_note": (
            "An exact Python 3.13 qualification constraints file is present; "
            "it is not a universal cross-platform lockfile."
        ),
    }


def metric_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    decision = summary["selected_decision_metrics"]
    return {
        "release_status": summary["release_status"],
        "selected_variant": summary["selected_variant"],
        "selected_policy": summary["selected_policy_from_replay"],
        "deployed_policy": summary["deployed_policy"],
        "one_step_wape": summary["selected_forecast_metrics"]["wape"],
        "fixed_origin_wape": summary["fixed_origin_forecast_metrics"]["wape"],
        "coverage": summary["selected_forecast_metrics"]["interval_coverage_80"],
        "peak_q90_coverage": summary["forecast_tail_metrics"]["peak_q90_coverage"],
        "incident_q90_coverage": summary["forecast_tail_metrics"]["incident_q90_coverage"],
        "total_cost": decision["total_cost"],
        "service_level_lcb95": decision["service_level_lcb95"],
        "abandonment_rate_ucb95": decision["abandonment_rate_ucb95"],
        "schedule_feasibility": decision["schedule_feasibility"],
        "hard_violations": decision["hard_violations"],
        "flow_conservation": decision["flow_conservation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="release_candidate_handoff.json")
    parser.add_argument("--evidence-level", default="Q3")
    args = parser.parse_args()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    baseline_manifest = load_json(ROOT / "reports" / "baseline" / "source_manifest.json")
    baseline_metrics = load_json(ROOT / "reports" / "baseline" / "baseline_metrics.json")
    test_summary = load_json(ROOT / "reports" / "candidate" / "test_summary.json")
    evidence_log = load_json(ROOT / "reports" / "candidate" / "evidence_log.json")
    smoke = load_json(ROOT / "outputs" / "smoke" / "run_summary.json")
    stress = load_json(ROOT / "outputs" / "stress_insufficient_workforce" / "run_summary.json")
    candidate_manifest = build_source_manifest()
    candidate_manifest_path = ROOT / "reports" / "candidate" / "source_manifest.json"
    candidate_manifest_path.write_text(
        json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8"
    )
    diff = source_diff(baseline_manifest, candidate_manifest)
    diff_path = ROOT / "reports" / "candidate" / "source_diff.json"
    diff_path.write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")

    wheel = ROOT / "dist" / f"support_capacity_reliability-{version}-py3-none-any.whl"
    sdist = ROOT / "dist" / f"support_capacity_reliability-{version}.tar.gz"
    artifacts = [file_record(wheel), file_record(sdist)]
    dependencies = dependency_snapshot()
    git = git_state()

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "project_name": project["name"],
        "candidate_version": version,
        "candidate_status": "RELEASE_CANDIDATE",
        "release_qualified": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": git["commit"],
        "git_available": git["available"],
        "dirty_tree": git["dirty_tree"],
        "source_fingerprint": candidate_manifest["source_fingerprint"],
        "source_manifest": file_record(candidate_manifest_path),
        "baseline_source_fingerprint": (ROOT / "reports" / "baseline" / "source_fingerprint.txt")
        .read_text()
        .strip(),
        "diff_checksum": diff["diff_checksum"],
        "source_diff": file_record(diff_path),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "executable": sys.executable,
            "cpu_only_canonical_validation": True,
            "gpu_validation_executed": False,
        },
        "dependencies": dependencies,
        "supported_environment_claims": {
            "python": ">=3.11,<3.14",
            "current_environment": f"Python {platform.python_version()} on {platform.system()}",
            "github_matrix_declared": ["ubuntu:3.11", "ubuntu:3.12", "ubuntu:3.13", "windows:3.11"],
            "github_hosted_execution_verified": False,
            "windows_execution_verified": False,
            "cpu_fallback_verified": True,
            "cuda_chronos_verified": False,
        },
        "required_entrypoints": {
            "cli": "support-capacity",
            "api": "uvicorn support_capacity_reliability.api.app:app",
            "canonical_pipeline": "make smoke",
            "negative_gate": "make stress",
            "tests": "make test",
            "coverage": "make coverage",
            "build": "make package",
            "installed_wheel_e2e": "make wheel-e2e",
        },
        "datasets_and_fixtures": {
            "canonical": "Deterministic synthetic ACD, workforce, incident, redial, recontact, and operational-parameter data.",
            "negative": "Deterministic 36-agent insufficient-workforce synthetic fixture expected to return ITERATE.",
            "public_proxy": "Optional NYC 311 request-arrival adapter; not treated as contact-center ACD data.",
            "proprietary_data_used": False,
        },
        "baseline_metrics": baseline_metrics,
        "final_metrics": {
            "canonical": metric_snapshot(smoke),
            "stress": metric_snapshot(stress),
            "tests": test_summary["tests"],
            "coverage_percent": test_summary.get("coverage_percent"),
        },
        "metric_gates_and_tolerances": {
            "functional_regression": "Canonical and stress release status must remain unchanged.",
            "one_step_wape": "No increase beyond 1e-12 relative to baseline deterministic fixture.",
            "fixed_origin_wape": "No increase beyond 1e-12 relative to baseline deterministic fixture.",
            "total_cost": "No increase beyond 1e-8 relative to baseline deterministic fixture.",
            "coverage_minimum": 85,
            "hard_violations_canonical": 0,
            "stress_expected_status": "ITERATE",
            "artifact_replay_max_abs_error": 1e-9,
        },
        "tests_and_command_evidence": evidence_log,
        "build_artifacts": artifacts,
        "known_limitations": [
            "Canonical and stress results use deterministic synthetic offline replay, not live business outcomes.",
            "Docker image build was not executed because Docker is unavailable in the current environment.",
            "Remote GitHub Actions execution on the exact candidate commit was not performed.",
            "Chronos-2 weights, CUDA training, and GPU inference were not executed.",
            "No Windows runtime execution was performed in the current environment.",
            "The joblib model bundle must only be loaded from a trusted artifact source.",
            "The exact Python 3.13 constraints file is not a universal cross-platform lockfile.",
        ],
        "unresolved_items": [
            {
                "severity": "High",
                "item": "Exact-commit GitHub-hosted qualification",
                "reason": "No remote repository run is available in this environment.",
                "next_gate": "Push exact source fingerprint and require all configured CI jobs to pass.",
            },
            {
                "severity": "High",
                "item": "Docker image runtime qualification",
                "reason": "Docker executable is unavailable.",
                "next_gate": "Build image, run as non-root, verify writable outputs, smoke, and negative gate.",
            },
            {
                "severity": "Medium",
                "item": "Windows qualification",
                "reason": "Current execution environment is Linux.",
                "next_gate": "Run setup, tests, smoke, stress, and wheel E2E on Windows Python 3.11.",
            },
            {
                "severity": "Medium",
                "item": "Chronos/CUDA full-mode qualification",
                "reason": "Optional dependency and model weights were not installed.",
                "next_gate": "Run full doctor, GPU smoke, artifact reload, and metric comparison on the declared RTX environment.",
            },
        ],
        "evidence_level": args.evidence_level,
        "evidence_level_definition": {
            "Q0": "file and configuration review",
            "Q1": "current workspace command execution",
            "Q2": "clean local copy or extracted archive execution",
            "Q3": "built-artifact-only installation and execution",
            "Q4": "exact-commit GitHub-hosted execution",
        },
        "next_qualification_required_gates": [
            "Create a Git commit from the exact source fingerprint and run remote GitHub Actions.",
            "Build and run the Docker image in a Docker-enabled environment.",
            "Run Windows qualification if Windows support is claimed externally.",
            "Run Chronos/CUDA qualification before making GPU or foundation-model performance claims.",
            "Confirm release ZIP and handoff checksums in the final delivery location.",
        ],
    }

    output = ROOT / args.output
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Re-read and validate every referenced path and checksum before reporting success.
    written = load_json(output)
    for record in [
        written["source_manifest"],
        written["source_diff"],
        written["dependencies"]["pip_freeze"],
        *written["dependencies"]["dependency_contracts"],
        *written["build_artifacts"],
    ]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Handoff checksum validation failed for {path}")
    print(output)
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
