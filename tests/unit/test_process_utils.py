import os
import sys
import time

import pytest

from support_capacity_reliability.process_utils import (
    IsolatedCommandTimeout,
    run_isolated_command,
)


def test_isolated_command_does_not_wait_for_stdout_inheriting_grandchild():
    code = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print('parent_done', flush=True)"
    )
    started = time.monotonic()
    result = run_isolated_command(
        [sys.executable, "-c", code],
        timeout_seconds=10,
        terminate_group_on_success=True,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0
    assert result.stdout.strip() == "parent_done"
    assert elapsed < 10


def test_isolated_command_timeout_captures_partial_output():
    code = "import time; print('started', flush=True); time.sleep(30)"
    with pytest.raises(IsolatedCommandTimeout) as exc:
        run_isolated_command([sys.executable, "-c", code], timeout_seconds=3)
    assert "started" in exc.value.stdout


def test_isolated_command_propagates_environment(tmp_path):
    env = os.environ.copy()
    env["SUPPORT_CAPACITY_TEST_VALUE"] = "ok"
    result = run_isolated_command(
        [sys.executable, "-c", "import os; print(os.environ['SUPPORT_CAPACITY_TEST_VALUE'])"],
        timeout_seconds=10,
        env=env,
        cwd=tmp_path,
    )
    assert result.stdout.strip() == "ok"
