import re
import subprocess
from unittest.mock import Mock

import pytest

from tommos_adapter import TommosRunner


def test_name():
    assert TommosRunner().package_name == "Tommos"


def test_call_dry_run():
    runner = TommosRunner()
    cmd = runner._call("hyst", dry_run=True)
    assert cmd == "tommos loop hyst"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pixi run -e cuda tommos", "pixi run -e cuda tommos loop hyst"),
        (["pixi", "run", "-e", "cuda", "tommos"], "pixi run -e cuda tommos loop hyst"),
    ],
)
def test_call_dry_run_with_command_prefix(command, expected):
    runner = TommosRunner(command=command)
    assert runner._call("hyst", dry_run=True) == expected


def test_call_dry_run_does_not_require_local_executable(monkeypatch):
    which = Mock()
    monkeypatch.setattr("tommos_adapter.tommos_runner.shutil.which", which)

    assert TommosRunner(command="missing-tommos")._call("hyst", dry_run=True) == "missing-tommos loop hyst"
    which.assert_not_called()


def test_call_missing_executable(monkeypatch):
    monkeypatch.setattr("tommos_adapter.tommos_runner.shutil.which", lambda executable: None)

    with pytest.raises(OSError, match="command executable 'missing-tommos' was not found"):
        TommosRunner(command="missing-tommos")._call("hyst")


def test_call_resolves_executable(monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr("tommos_adapter.tommos_runner.shutil.which", lambda executable: "/resolved/tommos")
    monkeypatch.setattr("tommos_adapter.tommos_runner.subprocess.run", run)

    result = TommosRunner()._call("hyst")

    assert result.returncode == 0
    run.assert_called_once_with(
        ["/resolved/tommos", "loop", "hyst"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.mark.parametrize("command", ["", [], ["tommos", ""]])
def test_invalid_command(command):
    with pytest.raises(ValueError, match="at least one non-empty string"):
        TommosRunner(command=command)


def test_call():
    """Test calculator call without needed input files.

    The simulation will fail and we can check return code and reason for the error.
    """
    runner = TommosRunner()
    res = runner._call("nonexistent-system")
    assert res.returncode == 1
    assert "FileNotFoundError" in res.stderr.decode("utf-8")
    assert re.search(
        "No such file or directory: 'nonexistent-system.npz'",
        res.stderr.decode("utf-8"),
    )
