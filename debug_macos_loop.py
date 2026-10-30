"""Time tommos hysteresis drives from examples/example.ipynb with per-step output.

Run from the repository root with `pixi run python debug_macos_loop.py [timeout_seconds]`.
Writes logs to `macos-debug/`.
"""

import importlib.metadata
import io
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

import tommos_adapter as ta

OUT = Path(__file__).resolve().parent / "macos-debug"
TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0

CASES = {
    # example.ipynb Cell 13: single sweep, passes on macOS CI in about 3-5 minutes.
    "single_2T": [[(-2 / mm.consts.mu0, 0, 0), (1 / mm.consts.mu0, 0, 0), 16]],
    # First half of Cell 18 alone, without `--loop`.
    "sweep_4T": [[(4 / mm.consts.mu0, 0, 0), (-4 / mm.consts.mu0, 0, 0), 31]],
    # example.ipynb Cell 18: full loop, times out on macOS CI after 2000 s.
    "loop_4T": [
        [(4 / mm.consts.mu0, 0, 0), (-4 / mm.consts.mu0, 0, 0), 31],
        [(-4 / mm.consts.mu0, 0, 0), (4 / mm.consts.mu0, 0, 0), 31],
    ],
}


class StreamingRunner(ta.tommos_runner.TommosRunner):
    """Runner that streams tommos output with elapsed times to a log file and enforces a timeout."""

    def __init__(self, log_path: Path, timeout: float):
        """Store log path and timeout."""
        self.log_path = log_path
        self.timeout = timeout

    def _call(self, argstr: str, need_stderr: bool = False, dry_run: bool = False):
        """Run tommos like the parent runner, but stream its output instead of capturing it."""
        command = shlex.split(super()._call(argstr, dry_run=True))
        if dry_run:
            return shlex.join(command)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        start = time.monotonic()
        with self.log_path.open("w") as log:
            log.write(f"command: {shlex.join(command)}\n")
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            timer = threading.Timer(self.timeout, proc.kill)
            timer.start()
            for line in proc.stdout:
                log.write(f"[{time.monotonic() - start:8.1f}s] {line}")
                log.flush()
            proc.wait()
            timer.cancel()
            log.write(f"returncode: {proc.returncode} after {time.monotonic() - start:.1f}s\n")
        return subprocess.CompletedProcess(command, proc.returncode, stdout=b"", stderr=b"")


def environment_report() -> str:
    """Describe the machine and the numerical stack."""
    lines = [
        f"platform: {platform.platform()} {platform.machine()}",
        f"python: {sys.version}",
        f"cpu_count: {os.cpu_count()}",
    ]
    for name in ("tommos", "tommos-adapter", "numpy", "scipy", "jax", "jaxlib", "pyamg", "micromagneticmodel"):
        try:
            lines.append(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{name}: not installed")
    lines += [f"{k}={v}" for k, v in sorted(os.environ.items()) if "THREAD" in k or k.startswith(("MKL", "JAX", "XLA"))]
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return "\n".join(lines) + "\n\nnumpy config:\n" + buf.getvalue()


def make_system(m0: np.ndarray, mesh: pv.UnstructuredGrid) -> mm.System:
    """Build the system from example.ipynb with initial magnetisation `m0`."""
    system = mm.System(name="hysteresis")
    mesh = mesh.copy()
    mesh.point_data["m"] = m0
    mesh.cell_data["Ms"] = (1.76 / mm.consts.mu0) * np.ones(mesh.n_cells)
    system.m = mesh
    system.energy = mm.Exchange(A=1e-12) + mm.UniaxialAnisotropy(K=4e5, u=(0, 0, 1)) + mm.Demag()
    return system


def main() -> None:
    """Run all cases and write a summary."""
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    os.chdir(OUT)
    (OUT / "environment.txt").write_text(environment_report())

    subprocess.run(shlex.split("tommos mesh --geom box --extent 20,20,20 --h 2 --out-name cube"), check=True)
    mesh = pv.read("cube.vtu")
    mesh.points *= ta.scripts.DEFAULT_MESH_UNIT
    mesh.cell_data["region_id"] = mesh.get_array("mat_id")
    mesh.cell_data.remove("mat_id")
    m0 = np.zeros((mesh.n_points, 3), dtype=np.float32)
    m0[:, [1, 2]] = 1 / np.sqrt(2)

    summary = []
    for name, hsteps in CASES.items():
        log_path = OUT / f"{name}.log"
        print(f"running {name} (timeout {TIMEOUT:.0f}s), log: {log_path}", flush=True)
        start = time.monotonic()
        try:
            ta.HysteresisDriver().drive(
                make_system(m0, mesh), Hsteps=hsteps, runner=StreamingRunner(log_path, TIMEOUT), dirname=name, verbose=0
            )
            status = "ok"
        except Exception as exc:
            status = f"failed: {type(exc).__name__}"
        steps = sum(line.startswith("step ", 12) for line in log_path.read_text().splitlines())
        summary.append(f"{name}: {status}, {time.monotonic() - start:.1f}s, {steps} steps logged")
        print(summary[-1], flush=True)
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
