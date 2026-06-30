"""Canonical cross-platform local qualification runner.

The runner uses the same public entrypoints as CI, records every command, and exits on
the first failed required gate. It does not push, dispatch workflows, or access secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    name: str
    command: list[str]
    cwd: str
    exit_code: int
    duration_seconds: float
    log_path: str
    status: str


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _run(
    name: str,
    command: list[str],
    log_dir: Path,
    *,
    timeout_seconds: int,
    environment: dict[str, str],
) -> CommandResult:
    log_path = log_dir / f"{name}.log"
    started = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {json.dumps(command)}\nCWD: {ROOT}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate(process)
            exit_code = 124
            log.write(f"\nTIMEOUT after {timeout_seconds} seconds\n")
    duration = round(time.monotonic() - started, 3)
    return CommandResult(
        name=name,
        command=command,
        cwd=str(ROOT),
        exit_code=exit_code,
        duration_seconds=duration,
        log_path=str(log_path.relative_to(ROOT)),
        status="PASS" if exit_code == 0 else "FAIL",
    )


def _commands(profile: str, skip_install: bool) -> list[tuple[str, list[str], int]]:
    python = sys.executable
    commands: list[tuple[str, list[str], int]] = []
    if not skip_install:
        commands.extend(
            [
                ("install", [python, "-m", "pip", "install", ".[dev]"], 900),
                ("pip_check", [python, "-m", "pip", "check"], 120),
            ]
        )
    commands.extend(
        [
            (
                "doctor",
                [
                    python,
                    "-m",
                    "support_capacity_reliability.cli",
                    "doctor",
                    "--config",
                    "configs/smoke.yaml",
                ],
                180,
            ),
            (
                "format",
                [python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
                120,
            ),
            ("lint", [python, "-m", "ruff", "check", "src", "tests", "scripts"], 120),
            (
                "tests_coverage",
                [
                    python,
                    "scripts/run_test_suite.py",
                    "--suite",
                    "all",
                    "--coverage",
                    "--coverage-fail-under",
                    "85",
                    "--summary-json",
                    "reports/candidate/test_summary.json",
                ],
                1200,
            ),
            (
                "validate_smoke",
                [
                    python,
                    "-m",
                    "support_capacity_reliability.cli",
                    "validate-config",
                    "--config",
                    "configs/smoke.yaml",
                ],
                120,
            ),
            ("build", [python, "-m", "build"], 600),
            ("verify_distribution", [python, "scripts/verify_distribution.py"], 180),
            (
                "installed_wheel_e2e",
                [
                    python,
                    "scripts/smoke_installed_wheel.py",
                    "--run-pipeline",
                    "--run-api-pipeline",
                ],
                1200,
            ),
        ]
    )
    if profile in {"STANDARD", "EXTENDED"}:
        for index in range(1, 4):
            commands.append((f"api_smoke_{index}", [python, "scripts/smoke_api.py"], 180))
        for index in range(1, 4):
            commands.extend(
                [
                    (
                        f"pipeline_smoke_{index}",
                        [
                            python,
                            "-m",
                            "support_capacity_reliability.cli",
                            "run",
                            "--config",
                            "configs/smoke.yaml",
                            "--require-release",
                        ],
                        900,
                    ),
                    (
                        f"verify_output_{index}",
                        [
                            python,
                            "-m",
                            "support_capacity_reliability.cli",
                            "verify-output",
                            "--output",
                            "outputs/smoke",
                        ],
                        180,
                    ),
                    (
                        f"verify_bundle_{index}",
                        [
                            python,
                            "-m",
                            "support_capacity_reliability.cli",
                            "verify-model-bundle",
                            "--artifact-dir",
                            "outputs/smoke/artifacts",
                        ],
                        180,
                    ),
                ]
            )
        commands.extend(
            [
                (
                    "validate_stress",
                    [
                        python,
                        "-m",
                        "support_capacity_reliability.cli",
                        "validate-config",
                        "--config",
                        "configs/stress_insufficient_workforce.yaml",
                    ],
                    120,
                ),
                (
                    "stress_gate",
                    [
                        python,
                        "-m",
                        "support_capacity_reliability.cli",
                        "run",
                        "--config",
                        "configs/stress_insufficient_workforce.yaml",
                        "--expected-status",
                        "ITERATE",
                    ],
                    900,
                ),
            ]
        )
    if profile == "EXTENDED":
        commands.extend(
            [
                ("ci_static", [python, "scripts/verify_ci.py"], 120),
                ("security_scan", [python, "scripts/security_scan.py"], 180),
                ("regression_gate", [python, "scripts/verify_regression_gates.py"], 180),
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["CORE", "STANDARD", "EXTENDED"], default="CORE")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--output", default="reports/local_qualification.json")
    args = parser.parse_args()

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "reports" / "qualification_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )

    results: list[CommandResult] = []
    final_status = "PASS"
    for name, command, timeout_seconds in _commands(args.profile, args.skip_install):
        result = _run(
            name,
            command,
            log_dir,
            timeout_seconds=timeout_seconds,
            environment=environment,
        )
        results.append(result)
        print(f"[{result.status}] {name}: rc={result.exit_code} {result.duration_seconds:.3f}s")
        if result.exit_code != 0:
            final_status = "FAIL"
            break

    payload = {
        "schema_version": "1.0",
        "profile": args.profile,
        "status": final_status,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "commands": [asdict(result) for result in results],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    raise SystemExit(0 if final_status == "PASS" else 1)


if __name__ == "__main__":
    main()
