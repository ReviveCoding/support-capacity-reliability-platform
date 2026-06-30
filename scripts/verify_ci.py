"""Static checks for GitHub Actions and Docker release behavior."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CODEQL_WORKFLOW = ROOT / ".github" / "workflows" / "codeql.yml"
CHECKOUT = "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10"
SETUP_PYTHON = "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405"
UPLOAD_ARTIFACT = "actions/upload-artifact@4cec3d8aa04e39d1a68397de0c4cd6fb9dce8ec1"


def main() -> None:
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert "workflow_dispatch" in workflow["on"]
    jobs = workflow["jobs"]
    assert set(jobs) == {"quality", "torch-quality", "smoke", "insufficient-workforce-gate"}
    matrix = jobs["quality"]["strategy"]["matrix"]["include"]
    matrix_pairs = {(item["os"], item["python-version"]) for item in matrix}
    assert matrix_pairs == {
        ("ubuntu-latest", "3.11"),
        ("ubuntu-latest", "3.12"),
        ("ubuntu-latest", "3.13"),
        ("windows-latest", "3.11"),
    }
    for job in jobs.values():
        assert int(job["timeout-minutes"]) <= 30
        uses = [step.get("uses", "") for step in job["steps"]]
        assert CHECKOUT in uses
        assert SETUP_PYTHON in uses
        checkout_step = next(step for step in job["steps"] if step.get("uses") == CHECKOUT)
        assert checkout_step["with"]["persist-credentials"] == "false"
    assert UPLOAD_ARTIFACT in workflow_text
    assert "if: always()" in workflow_text
    assert "python -m pip install -e" not in workflow_text
    assert "python -m pip check" in workflow_text
    assert "https://download.pytorch.org/whl/cpu" in workflow_text
    assert "tests/unit/test_torch_model.py" in workflow_text
    assert "python scripts/qualify_local.py --profile CORE" in workflow_text
    assert "--require-release" in workflow_text
    assert "--expected-status ITERATE" in workflow_text
    assert "--run-pipeline --run-api-pipeline" in workflow_text
    assert "outputs/smoke/artifacts" in workflow_text
    assert "outputs/smoke/artifact_index.json" in workflow_text
    assert "dist/*.whl" in workflow_text
    assert "dist/*.tar.gz" in workflow_text
    assert "outputs/stress_insufficient_workforce/artifacts" in workflow_text
    assert (ROOT / ".github" / "dependabot.yml").exists()
    codeql_text = CODEQL_WORKFLOW.read_text(encoding="utf-8")
    assert "security-events: write" in codeql_text
    assert "github/codeql-action/init@8aad20d150bbac5944a9f9d289da16a4b0d87c1e" in codeql_text
    assert "github/codeql-action/analyze@8aad20d150bbac5944a9f9d289da16a4b0d87c1e" in codeql_text
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--require-release" in docker
    assert "pip install --no-cache-dir ." in docker
    assert "pip install --no-cache-dir -e ." not in docker
    assert "USER appuser" in docker
    assert "PYTHONUNBUFFERED=1" in docker
    print("CI and Docker static verification PASS")


if __name__ == "__main__":
    main()
