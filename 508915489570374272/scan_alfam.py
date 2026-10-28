#!/usr/bin/env python3
"""Scan RSP_alfam (eddy viscosity) and report the resulting Gaia G amplitude.

Eddy viscosity damps the nonlinear limit cycle, so lower alfam gives a larger
amplitude and a more strongly nonlinear (sawtooth) light-curve shape. This runs
the full two-stage pipeline once per value, archives each LOGS_colors, and
prints the amplitude against the observed one.

Each value costs a full settle plus Colors run, roughly 9 minutes.

    ./scan_alfam.py 0.05 0.08 0.10 0.12 0.15 0.18 0.20 0.22
    ./scan_alfam.py --timeout 900 0.05 0.10 0.15 0.20

Safe to leave unattended. Every run is bounded by --timeout (default 900 s; a
converged settle plus Colors takes about 9 minutes, so this only fires on a
model that will not settle). A crash, timeout or non-convergence is recorded
and the scan moves on. target.toml is restored on exit, including on Ctrl-C.
Results are also written to scan_alfam/results.csv as they complete.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "target.toml"
ARCHIVE = ROOT / "scan_alfam"


def set_alfam(value: float) -> None:
    text = CONFIG.read_text()
    new, n = re.subn(r"^rsp_alfam = .*$", f"rsp_alfam = {value}", text,
                     flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"expected one rsp_alfam line in target.toml, found {n}")
    CONFIG.write_text(new)


def read_history(path: Path):
    lines = path.read_text().splitlines()
    idx = next(i for i, l in enumerate(lines) if "model_number" in l.split())
    names = lines[idx].split()
    data = np.genfromtxt(path, skip_header=idx + 1)
    return {name: data[:, i] for i, name in enumerate(names)}


def observed_amplitude() -> float | None:
    obs = ROOT / "observations" / "gaia_epoch_photometry.csv"
    if not obs.is_file():
        return None
    rows = [l.split(",") for l in obs.read_text().splitlines()[1:] if l.strip()]
    mags = [float(r[2]) for r in rows if r[1].strip().upper() == "G"]
    return max(mags) - min(mags) if mags else None


def clear_outputs() -> None:
    """Remove previous run products so a failure cannot be read as a success."""
    for name in ("LOGS_colors", "LOGS_settle", "LOGS"):
        path = ROOT / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("settled.mod", "colors_final.mod", "SETTLED_ACCEPTED",
                 "LINA_REJECTED", "SETTLING_REJECTED", "MODE_REJECTED",
                 "ENERGY_REJECTED"):
        path = ROOT / name
        if path.is_file():
            path.unlink()


def write_results(rows, a_obs) -> None:
    out = ARCHIVE / "results.csv"
    header = ("rsp_alfam,g_amplitude_model,g_amplitude_observed,"
              "period_days,runtime_s,status")
    obs_str = "" if a_obs is None else f"{a_obs:.6f}"
    with out.open("w") as fh:
        fh.write(header + "\n")
        for value, amp, period, elapsed, status in rows:
            amp_str = "" if amp is None else f"{amp:.6f}"
            per_str = "" if period is None else f"{period:.6f}"
            fh.write(f"{value},{amp_str},{obs_str},{per_str},"
                     f"{elapsed:.1f},{status}\n")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("values", nargs="+", type=float)
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="seconds allowed per run before it is killed")
    args = ap.parse_args()
    values = args.values

    with CONFIG.open("rb") as fh:
        original_alfam = tomllib.load(fh)["run"]["rsp_alfam"]
    backup = CONFIG.read_text()

    a_obs = observed_amplitude()
    ARCHIVE.mkdir(exist_ok=True)
    rows: list[tuple] = []

    print(f"{'alfam':<8} {'A_G(mod)':<12} {'A_G(obs)':<12} {'P(mod)':<12} status")
    print(f"{'-----':<8} {'--------':<12} {'--------':<12} {'------':<12} ------")

    try:
        for value in values:
            set_alfam(value)
            clear_outputs()
            log = ARCHIVE / f"alfam{value}.log"
            started = time.time()
            try:
                with log.open("w") as fh:
                    rc = subprocess.run(["./rn"], cwd=ROOT, stdout=fh,
                                        stderr=subprocess.STDOUT,
                                        timeout=args.timeout).returncode
                note = f"rc={rc}"
            except subprocess.TimeoutExpired:
                rc, note = -1, f"TIMEOUT after {args.timeout:.0f}s"
            except OSError as exc:
                rc, note = -1, f"could not run ./rn: {exc}"
            elapsed = time.time() - started

            hist = ROOT / "LOGS_colors" / "history.data"
            if rc != 0 or not hist.is_file():
                print(f"{value:<8} {'--':<12} {'--':<12} {'--':<12} "
                      f"FAILED ({note}, {elapsed/60:.1f} min, see {log.name})")
                rows.append((value, None, None, elapsed, note))
                write_results(rows, a_obs)
                continue

            try:
                h = read_history(hist)
            except Exception as exc:
                print(f"{value:<8} {'--':<12} {'--':<12} {'--':<12} "
                      f"unreadable history: {exc}")
                rows.append((value, None, None, elapsed, "bad history"))
                write_results(rows, a_obs)
                continue

            good = np.isfinite(h["G"])
            amp = float(np.nanmax(h["G"][good]) - np.nanmin(h["G"][good]))
            period = float(np.nanmedian(h["rsp_period_in_days"][good]))
            obs_str = f"{a_obs:.4f}" if a_obs else "--"
            print(f"{value:<8} {amp:<12.4f} {obs_str:<12} {period:<12.5f} "
                  f"ok ({elapsed/60:.1f} min)")
            rows.append((value, amp, period, elapsed, "ok"))
            write_results(rows, a_obs)

            dest = ARCHIVE / f"LOGS_colors_alfam{value}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ROOT / "LOGS_colors", dest)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        CONFIG.write_text(backup)
        print(f"\ntarget.toml restored (rsp_alfam = {original_alfam})")
        print(f"archived runs in {ARCHIVE}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
