from __future__ import annotations

import os
import platform
import sys
import tempfile
import textwrap
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from support_capacity_reliability.config import load_config
from support_capacity_reliability.process_utils import (
    IsolatedCommandTimeout,
    run_isolated_command,
)

REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "simpy": "simpy",
    "ortools": "ortools",
    "pydantic": "pydantic",
    "yaml": "PyYAML",
    "fastapi": "fastapi",
    "tabulate": "tabulate",
}


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _probe_python(code: str, timeout_seconds: int = 30) -> tuple[bool, str]:
    """Run a bounded Python runtime probe in an isolated process group.

    Some native libraries complete the requested probe but block during interpreter
    shutdown. The probe is trusted project-owned code, so flush its output and exit
    the isolated worker directly after the probe completes. This keeps the public CLI
    on normal Python shutdown semantics while bounding native-library teardown.
    """
    env = os.environ.copy()
    wrapped = (
        "import os, sys, traceback\n"
        "try:\n" + textwrap.indent(code, "    ") + "\nexcept Exception:\n"
        "    traceback.print_exc()\n"
        "    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)\n"
        "sys.stdout.flush(); sys.stderr.flush(); os._exit(0)\n"
    )
    with tempfile.TemporaryDirectory(prefix="support_capacity_doctor_") as temp_dir:
        env.setdefault("MPLBACKEND", "Agg")
        env["MPLCONFIGDIR"] = temp_dir
        try:
            completed = run_isolated_command(
                [sys.executable, "-c", wrapped],
                timeout_seconds=timeout_seconds,
                env=env,
            )
        except IsolatedCommandTimeout:
            return False, f"runtime probe timed out after {timeout_seconds}s"
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return False, output or f"runtime probe exited with code {completed.returncode}"
    return True, output or "runtime probe completed"


@cache
def _probe_import(module: str, timeout_seconds: int = 30) -> tuple[bool, str]:
    """Import a dependency in an isolated process with a hard timeout."""
    code = f"import importlib; importlib.import_module({module!r}); print('import_ok')"
    passed, detail = _probe_python(code, timeout_seconds=timeout_seconds)
    if not passed and detail.startswith("runtime probe timed out"):
        return False, f"import timed out after {timeout_seconds}s"
    return passed, detail


def _run_pip_check(timeout_seconds: int = 60) -> tuple[int, str]:
    """Run ``pip check`` without depending on pip interpreter shutdown."""
    code = (
        "import os, sys; "
        "from pip._internal.cli.main import main; "
        "returncode = int(main(['check']) or 0); "
        "sys.stdout.flush(); sys.stderr.flush(); os._exit(returncode)"
    )
    try:
        completed = run_isolated_command(
            [sys.executable, "-c", code],
            timeout_seconds=timeout_seconds,
            env=os.environ.copy(),
        )
    except IsolatedCommandTimeout:
        return 124, f"pip check timed out after {timeout_seconds}s"
    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode, output


def run_doctor(config_path: str | Path = "configs/smoke.yaml") -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    python_ok = (3, 11) <= sys.version_info[:2] < (3, 14)
    checks.append(
        {
            "name": "python_version",
            "passed": python_ok,
            "detail": platform.python_version(),
        }
    )

    for module, package in REQUIRED_MODULES.items():
        passed, import_detail = _probe_import(module)
        detail = _package_version(package) if passed else import_detail
        checks.append({"name": f"dependency:{package}", "passed": passed, "detail": detail})

    returncode, output = _run_pip_check()
    if returncode == 0:
        passed = True
        detail = output or "No broken requirements found"
    else:
        direct_names = {
            package.lower().replace("_", "-") for package in REQUIRED_MODULES.values()
        } | {"support-capacity-reliability"}
        conflict_lines = [line.strip() for line in output.splitlines() if line.strip()]
        project_conflicts = [
            line
            for line in conflict_lines
            if line.split(maxsplit=1)[0].lower().replace("_", "-") in direct_names
        ]
        passed = False if returncode == 124 else not project_conflicts
        detail = (
            output
            if returncode == 124
            else (
                "Project dependency conflicts: " + " | ".join(project_conflicts)
                if project_conflicts
                else "Unrelated environment conflicts ignored: " + " | ".join(conflict_lines)
            )
        )
    checks.append({"name": "pip_dependency_consistency", "passed": passed, "detail": detail})

    config = None
    try:
        config = load_config(config_path)
        output_parent = Path(config.project.output_dir).expanduser().resolve().parent
        output_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_parent, delete=True):
            pass
        checks.append(
            {
                "name": "config_and_output_writable",
                "passed": True,
                "detail": str(output_parent),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "config_and_output_writable",
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    if config is not None and "torch_quantile" in config.forecast.models:
        passed, import_detail = _probe_import("torch")
        detail = (
            _package_version("torch")
            if passed
            else (import_detail + "; install support-capacity-reliability[torch]")
        )
        checks.append(
            {
                "name": "optional_dependency:torch",
                "passed": passed,
                "detail": detail,
            }
        )

    if config is not None and config.forecast.chronos.enabled:
        from support_capacity_reliability.forecasting.chronos_adapter import Chronos2Adapter

        status = Chronos2Adapter.availability()
        checks.append(
            {
                "name": "optional_dependency:chronos-forecasting",
                "passed": status.available,
                "detail": status.package_version if status.available else status.reason,
            }
        )

    passed, detail = _probe_python(
        "from scipy.optimize import Bounds, LinearConstraint, milp; "
        "import numpy as np; "
        "r=milp(c=np.array([1.0]), integrality=np.array([1]), "
        "bounds=Bounds([0.0],[10.0]), "
        "constraints=LinearConstraint([[1.0]],[1.0],[np.inf])); "
        "assert r.success and r.x is not None; print(r.message)",
        timeout_seconds=30,
    )
    checks.append({"name": "scipy_highs_milp", "passed": passed, "detail": detail})

    passed, detail = _probe_python(
        "from ortools.sat.python import cp_model; "
        "m=cp_model.CpModel(); v=m.NewBoolVar('value'); m.Add(v==1); "
        "s=cp_model.CpSolver(); s.parameters.num_search_workers=1; st=s.Solve(m); "
        "assert st in (cp_model.OPTIMAL,cp_model.FEASIBLE) and s.Value(v)==1; "
        "print(s.StatusName(st))",
        timeout_seconds=30,
    )
    checks.append({"name": "ortools_cp_sat", "passed": passed, "detail": detail})

    passed, detail = _probe_python(
        "import simpy; env=simpy.Environment(); event=env.timeout(1, value=1); "
        "assert env.run(until=event)==1; print('event loop completed')",
        timeout_seconds=15,
    )
    checks.append({"name": "simpy_event_loop", "passed": passed, "detail": detail})

    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "checks": checks,
    }
