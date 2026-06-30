"""Benchmark the public FastAPI entrypoint through a real Uvicorn socket."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, payload: dict[str, object] | None = None) -> tuple[int, bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return int(response.status), response.read()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _linux_process_snapshot(pid: int) -> dict[str, int | None]:
    status_path = Path(f"/proc/{pid}/status")
    fd_path = Path(f"/proc/{pid}/fd")
    result: dict[str, int | None] = {"rss_kib": None, "threads": None, "fd_count": None}
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result["rss_kib"] = int(line.split()[1])
            elif line.startswith("Threads:"):
                result["threads"] = int(line.split()[1])
    if fd_path.is_dir():
        result["fd_count"] = len(list(fd_path.iterdir()))
    return result


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="reports/api_performance.json")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        raise SystemExit("requests/concurrency must be positive and warmup non-negative")

    port = _free_port()
    log_path = ROOT / "reports" / "api_benchmark_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
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
    start_new_session = os.name != "nt"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("wb") as log:
        started_at = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        try:
            health_url = f"http://127.0.0.1:{port}/health"
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited early with code {process.returncode}")
                try:
                    status, _ = _request(health_url)
                    if status == 200:
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("server did not become healthy within 30 seconds")
                time.sleep(0.1)
            startup_seconds = time.monotonic() - started_at
            endpoint = f"http://127.0.0.1:{port}/required-staffing"
            payload = {
                "contacts_per_interval": 12,
                "interval_minutes": 30,
                "average_handle_time_seconds": 420,
                "patience_mean_seconds": 240,
            }
            for _ in range(args.warmup):
                status, body = _request(endpoint, payload)
                if status != 200 or not body:
                    raise RuntimeError("warmup request failed")
            before = _linux_process_snapshot(process.pid)

            def measured_request() -> tuple[float, int]:
                request_start = time.perf_counter()
                try:
                    status, body = _request(endpoint, payload)
                    if not body:
                        return time.perf_counter() - request_start, 599
                    return time.perf_counter() - request_start, status
                except Exception:
                    return time.perf_counter() - request_start, 599

            latencies: list[float] = []
            statuses: list[int] = []
            measurement_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(measured_request) for _ in range(args.requests)]
                for future in as_completed(futures):
                    latency, status = future.result()
                    latencies.append(latency * 1000)
                    statuses.append(status)
            elapsed = time.perf_counter() - measurement_start
            after = _linux_process_snapshot(process.pid)
        finally:
            _stop(process)

    errors = sum(status != 200 for status in statuses)
    expected_signal_exit = os.name != "nt" and process.returncode == -signal.SIGTERM
    shutdown_ok = process.returncode == 0 or expected_signal_exit
    payload_out = {
        "schema_version": "1.0",
        "status": "PASS" if errors == 0 and shutdown_ok else "FAIL",
        "public_entrypoint": "uvicorn support_capacity_reliability.api.app:app",
        "requests": args.requests,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup,
        "errors": errors,
        "error_rate": errors / args.requests,
        "startup_seconds": round(startup_seconds, 6),
        "wall_clock_seconds": round(elapsed, 6),
        "throughput_requests_per_second": round(args.requests / elapsed, 3),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 6),
            "p95": round(_percentile(latencies, 0.95), 6),
            "p99": round(_percentile(latencies, 0.99), 6),
            "mean": round(statistics.fmean(latencies), 6),
            "max": round(max(latencies), 6),
        },
        "process_before": before,
        "process_after": after,
        "shutdown_returncode": process.returncode,
        "shutdown_ok": shutdown_ok,
        "expected_signal_exit": expected_signal_exit,
        "gate": {
            "predefined_latency_threshold": None,
            "required_error_rate": 0.0,
            "observational_only_for_latency_and_memory": True,
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload_out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload_out, indent=2))
    raise SystemExit(0 if payload_out["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
