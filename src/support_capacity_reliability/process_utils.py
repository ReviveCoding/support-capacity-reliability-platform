"""Bounded subprocess execution without inherited-pipe shutdown deadlocks."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IsolatedCommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class IsolatedCommandTimeout(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate the direct process on Windows, or its session group on POSIX."""
    if os.name == "nt":
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_isolated_command(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    terminate_group_on_success: bool = False,
) -> IsolatedCommandResult:
    """Run a command in a fresh process group and collect output through files.

    Pipes can remain open when native libraries or grandchildren inherit their descriptors,
    making ``communicate()`` wait after the direct child has already exited. Temporary files
    avoid that lifecycle coupling. On Windows, a detached grandchild can still retain an
    inherited log-file handle after a successful leader exit; cleanup is therefore best-effort
    only for this explicitly requested success-cleanup path.
    """
    args = tuple(str(value) for value in command)
    temporary = tempfile.TemporaryDirectory(
        prefix="support_capacity_process_",
        ignore_cleanup_errors=(os.name == "nt" and terminate_group_on_success),
    )
    try:
        temporary_path = Path(temporary.name)
        stdout_path = temporary_path / "stdout.log"
        stderr_path = temporary_path / "stderr.log"
        popen_kwargs: dict[str, object] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process: subprocess.Popen[bytes] = subprocess.Popen(
                list(args),
                cwd=str(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                **popen_kwargs,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                stdout_handle.flush()
                stderr_handle.flush()
                stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                raise IsolatedCommandTimeout(
                    f"Command exceeded timeout of {timeout_seconds:g} seconds: {' '.join(args)}",
                    stdout=stdout,
                    stderr=stderr,
                ) from exc

            if terminate_group_on_success:
                _terminate_process_group(process)

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        return IsolatedCommandResult(args, returncode, stdout, stderr)
    finally:
        temporary.cleanup()
