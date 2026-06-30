"""Conservative local secret and release-file scan without external services."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "outputs",
    "qualification_logs",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".dockerignore",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "generic_secret_assignment": re.compile(
        r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"
    ),
}


def _eligible(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in {"Dockerfile", "Makefile", "LICENSE"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 2_000_000


def main() -> None:
    findings: list[dict[str, object]] = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or not _eligible(path):
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for rule_name, pattern in RULES.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "rule": rule_name,
                            "path": str(path.relative_to(ROOT)),
                            "line": line_number,
                        }
                    )
    payload = {
        "schema_version": "1.0",
        "status": "PASS" if not findings else "FAIL",
        "files_scanned": scanned,
        "findings": findings,
        "limitations": [
            "Local regex scan only; this is not GitHub secret scanning or a vulnerability database scan.",
            "Matched secret values are never written to the report.",
        ],
    }
    output = ROOT / "reports" / "security_scan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
