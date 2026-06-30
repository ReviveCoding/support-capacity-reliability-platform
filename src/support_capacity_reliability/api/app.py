from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from support_capacity_reliability import __version__
from support_capacity_reliability.config import load_config, resolve_config_path
from support_capacity_reliability.queueing.erlang import required_agents_erlang_a
from support_capacity_reliability.runtime import (
    IsolatedPipelineBusyError,
    run_pipeline_isolated,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Support Capacity Reliability API", version=__version__)
PROJECT_ROOT = Path.cwd().resolve()
CONFIG_ROOT = (PROJECT_ROOT / "configs").resolve()
OUTPUT_ROOT = (PROJECT_ROOT / "outputs").resolve()
_BUNDLED_DEFAULTS = {
    "smoke.yaml",
    "stress_insufficient_workforce.yaml",
    "full.yaml",
}


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StaffingRequest(StrictRequest):
    contacts_per_interval: float = Field(ge=0)
    interval_minutes: int = Field(default=30, ge=1)
    average_handle_time_seconds: float = Field(gt=0)
    patience_mean_seconds: float = Field(gt=0)
    service_level_target: float = Field(default=0.8, ge=0, le=1)
    abandonment_target: float = Field(default=0.12, ge=0, le=1)
    service_level_seconds: float = Field(default=120, gt=0)


class PipelineRequest(StrictRequest):
    config_path: str = "configs/smoke.yaml"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/required-staffing")
def required_staffing(request: StaffingRequest) -> dict[str, Any]:
    result = required_agents_erlang_a(
        arrival_rate_per_second=request.contacts_per_interval / (request.interval_minutes * 60.0),
        average_handle_time_seconds=request.average_handle_time_seconds,
        patience_mean_seconds=request.patience_mean_seconds,
        service_level_target=request.service_level_target,
        abandonment_target=request.abandonment_target,
        service_level_seconds=request.service_level_seconds,
        max_agents=500,
    )
    return result.__dict__


def _validate_requested_config_path(requested: str) -> tuple[Path, Path]:
    candidate = Path(requested)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=400, detail="Config must be a YAML file")
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(CONFIG_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="Config path must remain inside the workspace configs directory",
        ) from exc
    return candidate, resolved


@contextmanager
def _resolved_api_config(requested: str) -> Iterator[Path]:
    candidate, resolved = _validate_requested_config_path(requested)
    if resolved.is_file():
        yield resolved
        return
    if candidate.name not in _BUNDLED_DEFAULTS or candidate.parent not in {
        Path("."),
        Path("configs"),
    }:
        raise HTTPException(status_code=404, detail=f"Config not found: {candidate.name}")
    with resolve_config_path(candidate) as bundled:
        yield bundled


def _validate_api_output_path(config: Any) -> Path:
    output = Path(config.project.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    try:
        output.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="API pipeline output must remain inside the workspace outputs directory",
        ) from exc
    return output


@app.post("/run-pipeline")
def execute_pipeline(request: PipelineRequest) -> dict[str, Any]:
    try:
        with _resolved_api_config(request.config_path) as path:
            config = load_config(path)
            _validate_api_output_path(config)
            return run_pipeline_isolated(path)
    except HTTPException:
        raise
    except IsolatedPipelineBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {exc}") from exc
    except Exception as exc:
        logger.exception("Pipeline execution failed for config %s", request.config_path)
        raise HTTPException(
            status_code=500,
            detail="Pipeline execution failed; inspect server logs for the internal error",
        ) from exc
