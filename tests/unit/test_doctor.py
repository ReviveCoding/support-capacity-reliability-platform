from support_capacity_reliability.doctor import _probe_python, run_doctor


def test_doctor_aggregates_successful_checks(monkeypatch):
    monkeypatch.setattr(
        "support_capacity_reliability.doctor._probe_import",
        lambda module: (True, "import_ok"),
    )
    monkeypatch.setattr(
        "support_capacity_reliability.doctor._probe_python",
        lambda code, timeout_seconds=30: (True, "runtime probe completed"),
    )
    monkeypatch.setattr(
        "support_capacity_reliability.doctor._run_pip_check",
        lambda: (0, "No broken requirements found."),
    )
    report = run_doctor("configs/smoke.yaml")
    assert report["status"] == "PASS"
    assert all(check["passed"] for check in report["checks"])


def test_runtime_probe_times_out_cleanly():
    passed, detail = _probe_python("import time; time.sleep(10)", timeout_seconds=1)
    assert not passed
    assert detail == "runtime probe timed out after 1s"


def test_runtime_probe_bypasses_blocking_interpreter_shutdown():
    passed, detail = _probe_python(
        "import atexit, time; atexit.register(lambda: time.sleep(30)); print('probe completed')",
        timeout_seconds=3,
    )
    assert passed
    assert detail == "probe completed"


def test_pip_check_timeout_is_a_failed_doctor_check(monkeypatch):
    monkeypatch.setattr(
        "support_capacity_reliability.doctor._probe_import",
        lambda module: (True, "import_ok"),
    )
    monkeypatch.setattr(
        "support_capacity_reliability.doctor._probe_python",
        lambda code, timeout_seconds=30: (True, "runtime probe completed"),
    )
    monkeypatch.setattr(
        "support_capacity_reliability.doctor._run_pip_check",
        lambda: (124, "pip check timed out after 60s"),
    )
    report = run_doctor("configs/smoke.yaml")
    check = next(item for item in report["checks"] if item["name"] == "pip_dependency_consistency")
    assert not check["passed"]
    assert "timed out" in check["detail"]
    assert report["status"] == "FAIL"
