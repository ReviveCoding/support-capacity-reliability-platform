from importlib.metadata import version

from support_capacity_reliability import __version__


def test_package_version_matches_distribution_metadata():
    assert __version__ == version("support-capacity-reliability")
