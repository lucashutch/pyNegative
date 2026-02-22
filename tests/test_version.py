from importlib.metadata import version

import pynegative


def test_version_available():
    """Verify that the version is available and not 'unknown'."""
    assert pynegative.__version__ != "unknown"
    # It should match the installed version
    assert pynegative.__version__ == version("pynegative")


def test_version_format():
    """Verify that the version string has a reasonable format."""
    # Since we are using setuptools-scm, it might have dev suffix
    # but it should at least start with numbers (e.g., 0.1.4)
    # or be a valid version string.
    v = pynegative.__version__
    assert len(v) > 0
    # Basic check for major.minor
    parts = v.split(".")
    assert len(parts) >= 2
    assert parts[0].isdigit()
