"""Run pytest modules in isolated processes and optionally merge coverage data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support_capacity_reliability.process_utils import (  # noqa: E402
    IsolatedCommandTimeout,
    run_isolated_command,
)


def _test_files(suite: str) -> list[Path]:
    files: list[Path] = []
    if suite in {"unit", "all"}:
        files.extend(sorted((ROOT / "tests" / "unit").glob("test_*.py")))
    if suite in {"integration", "all"}:
        files.extend(sorted((ROOT / "tests" / "integration").glob("test_*.py")))
    if not files:
        raise RuntimeError(f"No tests found for suite={suite}")
    return files


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "42",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(SRC),
        }
    )
    return env


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    accepted_returncodes: set[int] | None = None,
):
    accepted = {0} if accepted_returncodes is None else accepted_returncodes
    try:
        result = run_isolated_command(
            command,
            cwd=ROOT,
            env=env,
            timeout_seconds=timeout,
            terminate_group_on_success=True,
        )
    except IsolatedCommandTimeout as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        raise SystemExit(str(exc)) from exc
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode not in accepted:
        raise SystemExit(result.returncode)
    return result


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for key in counts:
        matches = re.findall(rf"(\d+) {key}", output)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["unit", "integration", "all"], default="all")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--coverage-fail-under", type=float, default=85.0)
    parser.add_argument("--per-file-timeout", type=int, default=240)
    parser.add_argument(
        "--summary-json",
        default="reports/test_summary.json",
        help="Write machine-readable test evidence to this path",
    )
    args = parser.parse_args()

    env = _base_env()
    files = _test_files(args.suite)
    if args.coverage:
        for path in ROOT.glob(".coverage*"):
            path.unlink(missing_ok=True)

    aggregate = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    module_results: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(ROOT)
        print(f"\n[{index}/{len(files)}] {relative}", flush=True)
        if args.coverage:
            command = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--source=support_capacity_reliability",
                "-m",
                "pytest",
                str(relative),
                "-rA",
            ]
        else:
            command = [sys.executable, "-m", "pytest", str(relative), "-rA"]
        result = _run(
            command,
            env=env,
            timeout=args.per_file_timeout,
            accepted_returncodes={0, 5},
        )
        combined_output = result.stdout + "\n" + result.stderr
        if result.returncode == 5:
            normalized_output = combined_output.lower()
            if (
                "skipped" not in normalized_output
                or "failed" in normalized_output
                or "error" in normalized_output
            ):
                raise SystemExit(result.returncode)
        counts = _pytest_counts(combined_output)
        for key, value in counts.items():
            aggregate[key] += value
        module_results.append(
            {
                "path": str(relative),
                "returncode": result.returncode,
                **counts,
            }
        )

    coverage_percent: float | None = None
    if args.coverage:
        reports = ROOT / "reports"
        reports.mkdir(exist_ok=True)
        _run([sys.executable, "-m", "coverage", "combine"], env=env, timeout=60)
        coverage_result = _run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--show-missing",
                f"--fail-under={args.coverage_fail_under:g}",
            ],
            env=env,
            timeout=60,
        )
        total_matches = re.findall(
            r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", coverage_result.stdout, re.MULTILINE
        )
        if total_matches:
            coverage_percent = float(total_matches[-1])
        _run(
            [
                sys.executable,
                "-m",
                "coverage",
                "xml",
                "-o",
                "reports/coverage.xml",
            ],
            env=env,
            timeout=60,
        )

    summary = {
        "schema_version": "1.0",
        "suite": args.suite,
        "isolated_module_count": len(files),
        "tests": aggregate,
        "coverage_enabled": args.coverage,
        "coverage_percent": coverage_percent,
        "coverage_fail_under": args.coverage_fail_under if args.coverage else None,
        "modules": module_results,
    }
    summary_path = ROOT / args.summary_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nPASS: {aggregate['passed']} tests across {len(files)} isolated pytest modules "
        f"({args.suite})"
    )
    print(f"Summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
