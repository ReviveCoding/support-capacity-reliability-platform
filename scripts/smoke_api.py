"""Launch a real Uvicorn process and verify HTTP health and staffing endpoints."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
LOG_PATH = REPORTS / "api_server.log"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return int(response.status), json.loads(response.read().decode("utf-8"))


def main() -> None:
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "support_capacity_reliability.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            health_url = f"http://127.0.0.1:{port}/health"
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Uvicorn exited early with code {process.returncode}")
                try:
                    status, health = _request(health_url)
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Uvicorn did not become healthy within 30 seconds"
                        ) from exc
                    time.sleep(0.25)
            assert status == 200
            assert health["status"] == "ok"

            staffing_status, staffing = _request(
                f"http://127.0.0.1:{port}/required-staffing",
                {
                    "contacts_per_interval": 12,
                    "interval_minutes": 30,
                    "average_handle_time_seconds": 420,
                    "patience_mean_seconds": 240,
                },
            )
            assert staffing_status == 200
            assert int(staffing["agents"]) >= 1
            assert 0.0 <= float(staffing["service_level"]) <= 1.0
            assert 0.0 <= float(staffing["abandonment_rate"]) <= 1.0
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "health": health,
                        "staffing_agents": staffing["agents"],
                        "port": port,
                    },
                    indent=2,
                )
            )
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
