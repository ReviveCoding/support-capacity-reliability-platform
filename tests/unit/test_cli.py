import json
import sys

from support_capacity_reliability.cli import main


def test_validate_config_cli(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["support-capacity", "validate-config", "--config", "configs/smoke.yaml"],
    )
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"]["name"] == "support-capacity-reliability"


def test_doctor_cli(monkeypatch, capsys):
    import support_capacity_reliability.doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "run_doctor",
        lambda _: {"status": "PASS", "checks": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["support-capacity", "doctor", "--config", "configs/smoke.yaml"],
    )
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"


def test_missing_config_exits_cleanly_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["support-capacity", "validate-config", "--config", "configs/typo.yaml"],
    )
    import pytest

    with pytest.raises(SystemExit) as exc:
        main()
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "configuration error:" in captured.err
    assert "Traceback" not in captured.err


def test_busy_pipeline_uses_temporary_failure_exit_code(monkeypatch, capsys):
    import pytest

    from support_capacity_reliability.runtime import IsolatedPipelineBusyError

    def busy(_):
        raise IsolatedPipelineBusyError("lock owned")

    monkeypatch.setattr("support_capacity_reliability.runtime.run_pipeline_isolated", busy)
    monkeypatch.setattr(
        sys,
        "argv",
        ["support-capacity", "run", "--config", "configs/smoke.yaml"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    captured = capsys.readouterr()
    assert exc.value.code == 75
    assert "temporary failure:" in captured.err


def test_version_cli(monkeypatch, capsys):
    import pytest

    monkeypatch.setattr(sys, "argv", ["support-capacity", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "support-capacity 1.4.1rc2" in captured.out


def test_missing_output_exits_cleanly_without_traceback(monkeypatch, capsys, tmp_path):
    import pytest

    monkeypatch.setattr(
        sys,
        "argv",
        ["support-capacity", "verify-output", "--output", str(tmp_path / "missing")],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "verification error:" in captured.err
    assert "Traceback" not in captured.err


def test_debug_preserves_verification_exception(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "support-capacity",
            "--debug",
            "verify-output",
            "--output",
            str(tmp_path / "missing"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        main()
