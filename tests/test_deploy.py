"""Sanity checks for the systemd deploy artifacts."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIT_FILE = REPO_ROOT / "deploy" / "systemd" / "leo-trident.service"
INSTALL_SH = REPO_ROOT / "deploy" / "install-systemd.sh"
UNINSTALL_SH = REPO_ROOT / "deploy" / "uninstall-systemd.sh"


def test_unit_file_valid():
    if shutil.which("systemd-analyze") is None:
        pytest.skip("systemd-analyze not available")
    assert UNIT_FILE.is_file(), f"missing unit file: {UNIT_FILE}"
    result = subprocess.run(
        ["systemd-analyze", "verify", "--user", str(UNIT_FILE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"systemd-analyze verify failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_installer_syntax():
    assert INSTALL_SH.is_file(), f"missing installer: {INSTALL_SH}"
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_uninstaller_syntax():
    assert UNINSTALL_SH.is_file(), f"missing uninstaller: {UNINSTALL_SH}"
    result = subprocess.run(
        ["bash", "-n", str(UNINSTALL_SH)], capture_output=True, text=True
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"
