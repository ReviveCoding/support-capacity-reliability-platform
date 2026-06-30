import json

import pytest

from support_capacity_reliability import runtime
from support_capacity_reliability.process_utils import (
    IsolatedCommandResult,
    IsolatedCommandTimeout,
)


def _result(payload: dict[str, object] | str, returncode: int = 0, stderr: str = ""):
    stdout = json.dumps(payload) if isinstance(payload, dict) else payload
    return IsolatedCommandResult(("python",), returncode, stdout, stderr)


def test_isolated_pipeline_returns_success_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime,
        "run_isolated_command",
        lambda *args, **kwargs: _result({"ok": True, "summary": {"release_status": "PASS"}}),
    )

    result = runtime.run_pipeline_isolated(tmp_path / "config.yaml", timeout_seconds=9)

    assert result == {"release_status": "PASS"}


def test_isolated_pipeline_maps_busy_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime,
        "run_isolated_command",
        lambda *args, **kwargs: _result(
            {"ok": False, "kind": "busy", "message": "output locked"},
            returncode=75,
        ),
    )

    with pytest.raises(runtime.IsolatedPipelineBusyError, match="output locked"):
        runtime.run_pipeline_isolated(tmp_path / "config.yaml")


def test_isolated_pipeline_maps_worker_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime,
        "run_isolated_command",
        lambda *args, **kwargs: _result(
            {"ok": False, "kind": "pipeline", "message": "gate failed"},
            returncode=1,
        ),
    )

    with pytest.raises(runtime.IsolatedPipelineError, match="gate failed"):
        runtime.run_pipeline_isolated(tmp_path / "config.yaml")


def test_isolated_pipeline_rejects_invalid_worker_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime,
        "run_isolated_command",
        lambda *args, **kwargs: _result("not-json", returncode=1, stderr="worker traceback"),
    )

    with pytest.raises(runtime.IsolatedPipelineError, match="worker traceback"):
        runtime.run_pipeline_isolated(tmp_path / "config.yaml")


def test_isolated_pipeline_terminates_timed_out_worker(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise IsolatedCommandTimeout("timed out")

    monkeypatch.setattr(runtime, "run_isolated_command", timeout)

    with pytest.raises(runtime.IsolatedPipelineError, match="exceeded timeout of 3 seconds"):
        runtime.run_pipeline_isolated(tmp_path / "config.yaml", timeout_seconds=3)
