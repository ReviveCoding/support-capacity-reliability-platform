import sys

import pytest

from support_capacity_reliability import cli


def test_cli_require_release_exits_on_iterate(monkeypatch, capsys):
    monkeypatch.setattr(
        "support_capacity_reliability.runtime.run_pipeline_isolated",
        lambda _: {"release_status": "ITERATE"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["support-capacity", "run", "--config", "configs/smoke.yaml", "--require-release"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "ITERATE" in capsys.readouterr().out


def test_cli_expected_iterate_accepts_negative_gate(monkeypatch):
    monkeypatch.setattr(
        "support_capacity_reliability.runtime.run_pipeline_isolated",
        lambda _: {"release_status": "ITERATE"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "support-capacity",
            "run",
            "--config",
            "configs/smoke.yaml",
            "--expected-status",
            "ITERATE",
        ],
    )
    cli.main()
