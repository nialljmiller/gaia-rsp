#!/usr/bin/env python3
"""Run the pipeline once at a given RSP_alfam and measure what it actually costs.

The eight-value scan died because every run was assumed to cost about what the
alfam = 0.25 baseline did. It does not: at alfam = 0.15 the amplitude grows
fast, the surface reaches Mach 3.8, and the shock collapses the timestep, so the
run managed 4 cycles in 15 minutes against 87 cycles in 6.5 minutes at 0.25.

This runs ONE value with no timeout, reports progress live, and afterwards
prints the cost per cycle so the next scan can be planned from a measurement
rather than a guess. Ctrl-C stops it cleanly and still reports.

    ./time_alfam.py 0.20
    ./time_alfam.py 0.20 --max-minutes 45

target.toml is restored on exit.
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

CYCLE_RE = re.compile(
    r"^\s*period\s+(\d+)\s+([\d.]+)\s+delta R\s+([\d.]+)\s+"
    r"max vsurf/cs\s+([\d.]+)")


def set_alfam(value: float) -> None:
    text = CONFIG.read_text()
    new, n = re.subn(r"^rsp_alfam = .*$", f"rsp_alfam = {value}", text,
                     flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"expected one rsp_alfam line in target.toml, found {n}")
    CONFIG.write_text(new)


def clear_outputs() -> None:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("alfam", type=float)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop after this long; omit to run to completion")
    args = ap.parse_args()

    with CONFIG.open("rb") as fh:
        original = tomllib.load(fh)["run"]["rsp_alfam"]
    backup = CONFIG.read_text()

    ARCHIVE.mkdir(exist_ok=True)
    log_path = ARCHIVE / f"timed_alfam{args.alfam}.log"

    cycles: list[tuple[float, int, float, float, float]] = []
    started = time.time()
    outcome = "completed"

    print(f"alfam = {args.alfam}   (baseline 0.25 settled in 87 cycles / 6.6 min)")
    print(f"logging to {log_path}")
    print(f"{'elapsed':>9} {'cycle':>6} {'P(d)':>9} {'DeltaR':>8} "
          f"{'vsurf/cs':>9} {'s/cycle':>9}")

    try:
        set_alfam(args.alfam)
        clear_outputs()

        with log_path.open("w") as log:
            proc = subprocess.Popen(
                ["./rn"], cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    log.write(line)
                    m = CYCLE_RE.match(line)
                    if m:
                        elapsed = time.time() - started
                        n = int(m.group(1))
                        rec = (elapsed, n, float(m.group(2)),
                               float(m.group(3)), float(m.group(4)))
                        per = (elapsed - cycles[-1][0]) if cycles else elapsed
                        cycles.append(rec)
                        print(f"{elapsed/60:8.1f}m {n:6d} {rec[2]:9.4f} "
                              f"{rec[3]:8.3f} {rec[4]:9.3f} {per:9.1f}")
                        sys.stdout.flush()
                    if args.max_minutes and \
                            (time.time() - started) > args.max_minutes * 60:
                        outcome = f"stopped at --max-minutes {args.max_minutes}"
                        proc.terminate()
                        break
                proc.wait(timeout=60)
            except KeyboardInterrupt:
                outcome = "interrupted"
                proc.terminate()
                proc.wait(timeout=60)
            finally:
                if proc.poll() is None:
                    proc.kill()
    finally:
        CONFIG.write_text(backup)

    total = time.time() - started
    print(f"\n--- {outcome} after {total/60:.1f} min ---")
    print(f"target.toml restored (rsp_alfam = {original})")

    if not cycles:
        print("no completed cycles -- check the log")
        return 1

    last = cycles[-1]
    print(f"reached cycle {last[1]} in {last[0]/60:.1f} min "
          f"({last[0]/max(last[1],1):.1f} s/cycle)")
    print(f"final: P = {last[2]:.4f} d, DeltaR = {last[3]:.3f} Rsun, "
          f"max vsurf/cs = {last[4]:.3f}")

    if len(cycles) >= 4:
        recent = np.diff([c[0] for c in cycles[-4:]])
        print(f"recent cost {np.mean(recent):.1f} s/cycle "
              f"-> 87 cycles would take {87*np.mean(recent)/60:.0f} min")

    dr = [c[3] for c in cycles]
    if len(dr) >= 3:
        trend = "growing" if dr[-1] > dr[-3] else "settling or shrinking"
        print(f"DeltaR {dr[0]:.2f} -> {dr[-1]:.2f} Rsun ({trend})")

    hist = ROOT / "LOGS_colors" / "history.data"
    if hist.is_file():
        print("\nColors stage completed; replot with plot_gaia_fit.py")
    elif (ROOT / "SETTLED_ACCEPTED").is_file():
        print("\nsettle accepted but Colors did not finish")
    else:
        print("\nsettle did not reach acceptance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
