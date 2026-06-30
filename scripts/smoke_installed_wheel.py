"""Install the built wheel into an isolated target and validate runtime behavior."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the bundled canonical pipeline through the installed CLI outside the repository",
    )
    parser.add_argument(
        "--run-api-pipeline",
        action="store_true",
        help="Run the bundled canonical pipeline through FastAPI TestClient outside the repository",
    )
    args = parser.parse_args()
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected exactly one wheel, found {wheels}")
    with tempfile.TemporaryDirectory(prefix="support-capacity-wheel-") as temporary:
        workspace = Path(temporary)
        target = workspace / "site"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheels[0]),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target)
        validate = _run(
            [
                sys.executable,
                "-m",
                "support_capacity_reliability.cli",
                "validate-config",
            ],
            cwd=workspace,
            env=env,
        )
        config = json.loads(validate.stdout)
        assert config["project"]["name"] == "support-capacity-reliability"

        probe = _run(
            [
                sys.executable,
                "-c",
                (
                    "import json, support_capacity_reliability as p; "
                    "from fastapi.testclient import TestClient; "
                    "from support_capacity_reliability.api.app import app; "
                    "r=TestClient(app).get('/health'); "
                    "print(json.dumps({'file':p.__file__,'version':p.__version__,"
                    "'status':r.status_code,'payload':r.json()}))"
                ),
            ],
            cwd=workspace,
            env=env,
        )
        payload = json.loads(probe.stdout)
        assert str(target) in payload["file"]
        assert payload["status"] == 200
        assert payload["payload"] == {"status": "ok", "version": payload["version"]}

        cli_release_status = None
        if args.run_pipeline:
            cli_run = _run(
                [
                    sys.executable,
                    "-m",
                    "support_capacity_reliability.cli",
                    "run",
                    "--require-release",
                ],
                cwd=workspace,
                env=env,
            )
            cli_summary = json.loads(cli_run.stdout)
            cli_release_status = cli_summary["release_status"]
            assert cli_release_status in {"PASS", "PASS_WITH_RECOURSE"}
            cli_output = workspace / "outputs" / "smoke"
            assert (cli_output / "run_manifest.json").is_file()
            assert (cli_output / "artifact_index.json").is_file()
            assert (cli_output / "artifacts" / "selected_forecast_bundle_manifest.json").is_file()
            output_verification = _run(
                [
                    sys.executable,
                    "-m",
                    "support_capacity_reliability.cli",
                    "verify-output",
                    "--output",
                    str(cli_output),
                ],
                cwd=workspace,
                env=env,
            )
            assert json.loads(output_verification.stdout)["status"] == "PASS"
            bundle_verification = _run(
                [
                    sys.executable,
                    "-m",
                    "support_capacity_reliability.cli",
                    "verify-model-bundle",
                    "--artifact-dir",
                    str(cli_output / "artifacts"),
                ],
                cwd=workspace,
                env=env,
            )
            assert json.loads(bundle_verification.stdout)["status"] in {
                "PASS",
                "EXTERNAL_MODEL_REFERENCE",
            }

        api_release_status = None
        if args.run_api_pipeline:
            api_workspace = workspace / "api-workspace"
            api_workspace.mkdir()
            api_probe = _run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "from fastapi.testclient import TestClient; "
                        "from support_capacity_reliability.api.app import app; "
                        "r=TestClient(app).post('/run-pipeline', "
                        "json={'config_path':'configs/smoke.yaml'}); "
                        "print(json.dumps({'status':r.status_code,'payload':r.json()}))"
                    ),
                ],
                cwd=api_workspace,
                env=env,
            )
            api_payload = json.loads(api_probe.stdout)
            assert api_payload["status"] == 200, api_payload
            api_release_status = api_payload["payload"]["release_status"]
            assert api_release_status in {"PASS", "PASS_WITH_RECOURSE"}
            api_output = api_workspace / "outputs" / "smoke"
            assert (api_output / "run_manifest.json").is_file()
            assert (api_output / "artifact_index.json").is_file()
            assert (api_output / "artifacts" / "selected_forecast_bundle_manifest.json").is_file()
            api_output_verification = _run(
                [
                    sys.executable,
                    "-m",
                    "support_capacity_reliability.cli",
                    "verify-output",
                    "--output",
                    str(api_output),
                ],
                cwd=api_workspace,
                env=env,
            )
            assert json.loads(api_output_verification.stdout)["status"] == "PASS"
            api_bundle_verification = _run(
                [
                    sys.executable,
                    "-m",
                    "support_capacity_reliability.cli",
                    "verify-model-bundle",
                    "--artifact-dir",
                    str(api_output / "artifacts"),
                ],
                cwd=api_workspace,
                env=env,
            )
            assert json.loads(api_bundle_verification.stdout)["status"] in {
                "PASS",
                "EXTERNAL_MODEL_REFERENCE",
            }

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "wheel": wheels[0].name,
                    "module_file": payload["file"],
                    "version": payload["version"],
                    "bundled_config": config["project"]["name"],
                    "cli_release_status": cli_release_status,
                    "api_release_status": api_release_status,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
