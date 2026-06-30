from __future__ import annotations

import argparse
import json
import os
import sys

from support_capacity_reliability import __version__
from support_capacity_reliability.process_utils import (
    IsolatedCommandTimeout,
    run_isolated_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support capacity reliability platform")
    parser.add_argument(
        "--version",
        action="version",
        version=f"support-capacity {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Preserve Python tracebacks for operational failures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the end-to-end pipeline")
    run.add_argument("--config", default="configs/smoke.yaml")
    run.add_argument(
        "--require-release",
        action="store_true",
        help="Exit nonzero unless release_status is PASS or PASS_WITH_RECOURSE",
    )
    run.add_argument(
        "--expected-status",
        choices=["PASS", "PASS_WITH_RECOURSE", "ITERATE"],
        default=None,
        help="Exit nonzero unless release_status exactly matches this value",
    )
    validate = subparsers.add_parser("validate-config", help="Validate a configuration")
    validate.add_argument("--config", default="configs/smoke.yaml")
    doctor = subparsers.add_parser("doctor", help="Check dependencies and local solver runtime")
    doctor.add_argument("--config", default="configs/smoke.yaml")
    verify_output = subparsers.add_parser(
        "verify-output", help="Verify a published output tree against its artifact index"
    )
    verify_output.add_argument("--output", default="outputs/smoke")
    verify_bundle = subparsers.add_parser(
        "verify-model-bundle", help="Replay a trusted persisted forecast bundle"
    )
    verify_bundle.add_argument("--artifact-dir", default="outputs/smoke/artifacts")
    return parser


def _emit_json_and_exit_if_console(payload: object, code: int) -> None:
    text = json.dumps(payload, indent=2, default=str) + "\n"
    if os.environ.get("SUPPORT_CAPACITY_CONSOLE_ENTRYPOINT") == "1":
        os.write(sys.stdout.fileno(), text.encode("utf-8", errors="replace"))
        os._exit(code)
    print(text, end="")


def _exit_if_console_invocation(code: int) -> None:
    """Exit immediately for public CLI invocations after long native workflows complete.

    Direct unit-test calls to ``main()`` keep returning normally. Console-script and
    ``python -m`` executions set this flag through ``entrypoint()`` so native solver
    libraries cannot delay process shutdown after the user-visible result is emitted.
    """
    if os.environ.get("SUPPORT_CAPACITY_CONSOLE_ENTRYPOINT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def _verify_model_bundle_in_isolated_process(artifact_dir: str) -> dict[str, object]:
    code = (
        "import json, os, sys; "
        "from support_capacity_reliability.artifacts import verify_model_bundle; "
        f"result=verify_model_bundle({artifact_dir!r}); "
        "sys.stdout.write(json.dumps(result)); sys.stdout.flush(); os._exit(0)"
    )
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        completed = run_isolated_command(
            [sys.executable, "-c", code],
            timeout_seconds=120,
            env=environment,
            terminate_group_on_success=True,
        )
    except IsolatedCommandTimeout as exc:
        raise RuntimeError("Isolated model-bundle verification timed out") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"Isolated model-bundle verification failed: {detail}")
    return json.loads(completed.stdout)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from pydantic import ValidationError
    from yaml import YAMLError

    from support_capacity_reliability.config import resolve_config_path

    try:
        if args.command == "run":
            from support_capacity_reliability.runtime import (
                IsolatedPipelineBusyError,
                IsolatedPipelineError,
                run_pipeline_isolated,
            )

            try:
                with resolve_config_path(args.config) as config_path:
                    summary = run_pipeline_isolated(config_path)
            except IsolatedPipelineBusyError as exc:
                parser.exit(75, f"temporary failure: {exc}\n")
            except IsolatedPipelineError as exc:
                parser.exit(1, f"pipeline failure: {exc}\n")
            release_status = str(summary.get("release_status", ""))
            code = 0
            if args.require_release and release_status not in {"PASS", "PASS_WITH_RECOURSE"}:
                code = 1
            if args.expected_status is not None and release_status != args.expected_status:
                code = 1
            _emit_json_and_exit_if_console(summary, code)
            if code != 0:
                sys.exit(code)
            return 0
        elif args.command == "validate-config":
            from support_capacity_reliability.config import load_config

            with resolve_config_path(args.config) as config_path:
                config = load_config(config_path)
            print(config.model_dump_json(indent=2))
            return 0
        elif args.command == "doctor":
            from support_capacity_reliability.doctor import run_doctor

            with resolve_config_path(args.config) as config_path:
                report = run_doctor(config_path)
            print(json.dumps(report, indent=2, default=str))
            if report["status"] != "PASS":
                sys.exit(1)
            return 0
        elif args.command == "verify-output":
            from support_capacity_reliability.artifacts import verify_published_artifacts

            report = verify_published_artifacts(args.output)
            print(json.dumps(report, indent=2, default=str))
            return 0
        elif args.command == "verify-model-bundle":
            report = _verify_model_bundle_in_isolated_process(args.artifact_dir)
            print(json.dumps(report, indent=2, default=str))
            return 0
        else:
            parser.print_help()
            sys.exit(2)
    except (ValidationError, YAMLError) as exc:
        if args.debug:
            raise
        parser.exit(2, f"configuration error: {exc}\n")
    except FileNotFoundError as exc:
        if args.debug:
            raise
        category = (
            "configuration error"
            if args.command in {"run", "validate-config", "doctor"}
            else "verification error"
        )
        parser.exit(2, f"{category}: {exc}\n")
    except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
        if args.debug:
            raise
        parser.exit(1, f"operation failed: {exc}\n")


def entrypoint() -> None:
    """Console-script entrypoint with hard exit to avoid native-library shutdown hangs."""
    os.environ["SUPPORT_CAPACITY_CONSOLE_ENTRYPOINT"] = "1"
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(int(code or 0))


if __name__ == "__main__":
    entrypoint()
