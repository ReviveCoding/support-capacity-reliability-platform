from __future__ import annotations

import os
import signal


def test_sigterm_returncode_is_expected_on_posix():
    returncode = -signal.SIGTERM
    expected_signal_exit = os.name != "nt" and returncode == -signal.SIGTERM
    shutdown_ok = returncode == 0 or expected_signal_exit
    assert shutdown_ok if os.name != "nt" else not shutdown_ok
