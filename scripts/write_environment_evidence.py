"""Write non-secret runtime and GitHub runner metadata for qualification evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

SAFE_ENV_KEYS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITHUB_EVENT_NAME",
    "GITHUB_REF",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_NUMBER",
    "GITHUB_SHA",
    "RUNNER_ARCH",
    "RUNNER_NAME",
    "RUNNER_OS",
    "ImageOS",
    "ImageVersion",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        package_version = metadata.version("support-capacity-reliability")
    except metadata.PackageNotFoundError:
        package_version = "NOT_INSTALLED"
    payload = {
        "schema_version": "1.0",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_version": package_version,
        "runner": {key: os.environ.get(key) for key in SAFE_ENV_KEYS if os.environ.get(key)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
