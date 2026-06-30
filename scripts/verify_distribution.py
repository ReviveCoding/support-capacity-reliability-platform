"""Verify wheel and source-distribution contents after ``python -m build``."""

from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path

from support_capacity_reliability import __version__

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DEFAULT_CONFIGS = {
    "smoke.yaml",
    "smoke_torch.yaml",
    "stress_insufficient_workforce.yaml",
    "full.yaml",
}
FORBIDDEN_PARTS = {"outputs", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _assert_clean(names: list[str], archive_name: str) -> None:
    contaminated = [
        name for name in names if any(part in FORBIDDEN_PARTS for part in Path(name).parts)
    ]
    if contaminated:
        raise AssertionError(f"{archive_name} contains generated/cache files: {contaminated[:5]}")


def _verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        _assert_clean(names, wheel.name)
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        version_match = re.search(r"^Version:\s*(.+)$", metadata, flags=re.MULTILINE)
        assert version_match and version_match.group(1).strip() == __version__
        entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        assert "support-capacity = support_capacity_reliability.cli:entrypoint" in archive.read(
            entry_points
        ).decode("utf-8")
        packaged = {
            Path(name).name
            for name in names
            if "/default_configs/" in name and name.endswith(".yaml")
        }
        assert packaged == DEFAULT_CONFIGS, (packaged, DEFAULT_CONFIGS)


def _verify_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        _assert_clean(names, sdist.name)
        packaged = {
            Path(name).name
            for name in names
            if "/default_configs/" in name and name.endswith(".yaml")
        }
        assert packaged == DEFAULT_CONFIGS, (packaged, DEFAULT_CONFIGS)
        pkg_info_name = next(name for name in names if name.endswith("/PKG-INFO"))
        pkg_info = archive.extractfile(pkg_info_name)
        assert pkg_info is not None
        metadata = pkg_info.read().decode("utf-8")
        version_match = re.search(r"^Version:\s*(.+)$", metadata, flags=re.MULTILINE)
        assert version_match and version_match.group(1).strip() == __version__


def main() -> None:
    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"Expected one wheel and one sdist, found {wheels=} {sdists=}")
    _verify_wheel(wheels[0])
    _verify_sdist(sdists[0])
    print(f"distribution verification PASS: version={__version__}")


if __name__ == "__main__":
    main()
