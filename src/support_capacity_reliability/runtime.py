from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from support_capacity_reliability.process_utils import (
    IsolatedCommandTimeout,
    run_isolated_command,
)


class IsolatedPipelineError(RuntimeError):
    pass


class IsolatedPipelineBusyError(IsolatedPipelineError):
    pass


def run_pipeline_isolated(
    config_path: str | Path,
    *,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    timeout = timeout_seconds or int(os.environ.get("SUPPORT_CAPACITY_RUN_TIMEOUT_SECONDS", "1800"))
    resolved = str(Path(config_path).expanduser().resolve())
    code = r"""
import json, os, sys
from support_capacity_reliability.pipeline import PipelineBusyError, PipelineError, run_pipeline
path = sys.argv[1]
try:
    summary = run_pipeline(path)
    payload = {"ok": True, "summary": summary}
    code = 0
except PipelineBusyError as exc:
    payload = {"ok": False, "kind": "busy", "message": str(exc)}
    code = 75
except PipelineError as exc:
    payload = {"ok": False, "kind": "pipeline", "message": str(exc)}
    code = 1
except Exception as exc:
    payload = {"ok": False, "kind": "unexpected", "message": f"{type(exc).__name__}: {exc}"}
    code = 1
sys.stdout.write(json.dumps(payload, default=str))
sys.stdout.flush()
os._exit(code)
"""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    try:
        completed = run_isolated_command(
            [sys.executable, "-c", code, resolved],
            timeout_seconds=timeout,
            env=environment,
            terminate_group_on_success=True,
        )
    except IsolatedCommandTimeout as exc:
        raise IsolatedPipelineError(
            f"Pipeline worker exceeded timeout of {timeout} seconds"
        ) from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker returned no output"
        raise IsolatedPipelineError(f"Pipeline worker returned invalid output: {detail}") from exc
    if completed.returncode == 0 and payload.get("ok") is True:
        return dict(payload["summary"])
    message = str(payload.get("message") or completed.stderr.strip() or "pipeline worker failed")
    if payload.get("kind") == "busy" or completed.returncode == 75:
        raise IsolatedPipelineBusyError(message)
    raise IsolatedPipelineError(message)
