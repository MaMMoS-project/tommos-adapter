"""Run the `tommos` micromagnetic calculator."""

import shlex
import shutil
import subprocess
from collections.abc import Sequence

from micromagneticmodel import adapter_base


class TommosRunner(adapter_base.ExternalRunner):
    """Runner to execute `tommos`.

    Args:
        command: Executable or command prefix used to invoke TOMMOS. For example,
            ``"tommos"``, ``"/path/to/tommos"``, or
            ``["pixi", "run", "-e", "cuda", "tommos"]``.
    """

    def __init__(self, command: str | Sequence[str] = "tommos") -> None:
        """Initialise the runner."""
        command_parts = shlex.split(command) if isinstance(command, str) else list(command)

        if not command_parts or any(not isinstance(part, str) or not part for part in command_parts):
            msg = "command must contain at least one non-empty string."
            raise ValueError(msg)

        self.command = tuple(command_parts)

    @property
    def package_name(self) -> str:
        """Name of the calculator."""
        return "Tommos"

    def _call(self, argstr: str, need_stderr: bool = False, dry_run: bool = False) -> str | subprocess.CompletedProcess:
        """Call the external calculator in a subprocess.

        Args:
            argstr: String argument to add to the process call.
            need_stderr: Whether the stderr is needed.
            dry_run: Whether to only return the command to be executed.

        Returns:
            Command string if run with `dry_run=True` without executing it, otherwise the command is executed and the
            result of the subprocess call is returned.
        """
        command = [*self.command, "loop", *shlex.split(argstr)]

        # `need_stderr` is ignored in this implementation, and stdout and stderr are always captured.
        stdout = stderr = subprocess.PIPE

        if dry_run:
            return shlex.join(command)

        executable = shutil.which(command[0])
        if executable is None:
            msg = (
                f"Cannot run TOMMOS: command executable {command[0]!r} was not found. "
                "Install it, add it to PATH, or configure TommosRunner(command=...)."
            )
            raise OSError(msg)

        command[0] = executable
        return subprocess.run(command, stdout=stdout, stderr=stderr)
