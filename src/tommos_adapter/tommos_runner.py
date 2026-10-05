"""Run the `tommos` micromagnetic calculator."""

import shlex
import subprocess

from micromagneticmodel import adapter_base


class TommosRunner(adapter_base.ExternalRunner):
    """Runner to execute `tommos`."""

    @property
    def package_name(self):
        """Name of the calculator."""
        return "Tommos Calculator"

    def _call(self, argstr, need_stderr=False, dry_run=False):
        """Call the external calcualtor in a subprocess."""
        command = shlex.split(f"tommos loop {argstr}")

        # `need_stderr` is ignored in this implementation, and stdout and stderr are always captured.
        stdout = stderr = subprocess.PIPE

        if dry_run:
            return shlex.join(command)
        return subprocess.run(command, stdout=stdout, stderr=stderr)
