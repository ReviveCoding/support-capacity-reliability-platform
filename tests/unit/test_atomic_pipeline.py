from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

import support_capacity_reliability.pipeline as pipeline


def _config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["project"]["output_dir"] = str(tmp_path / "published")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_failed_run_preserves_previous_published_output(tmp_path: Path, monkeypatch):
    config_path = _config(tmp_path)
    final = tmp_path / "published"
    final.mkdir()
    (final / "marker.txt").write_text("previous", encoding="utf-8")

    def fail(config):
        staging = Path(config.project.output_dir)
        staging.mkdir(parents=True)
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(pipeline, "_run_pipeline_impl", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        pipeline.run_pipeline(config_path)
    assert (final / "marker.txt").read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".published.staging-*"))
    assert not (tmp_path / ".published.lock").exists()


def test_active_run_lock_is_rejected(tmp_path: Path, monkeypatch):
    config_path = _config(tmp_path)
    lock = tmp_path / ".published.lock"
    lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "_run_pipeline_impl", lambda config: {})
    with pytest.raises(pipeline.PipelineBusyError):
        pipeline.run_pipeline(config_path)
