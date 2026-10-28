#!/usr/bin/env python3
"""Sweep the Smolec convection sets and rank them against the Gaia light curve.

Set A (no radiative cooling, no turbulent pressure, no turbulent flux) produced
a light curve that was too sinusoidal: the observed rise takes ~0.15 in phase
and the model's takes ~0.25, and the observed bump near phase 1.0 is absent.
Those are the terms the other sets add, so this is the parameter that should
move the shape -- alfam only scales the amplitude.

For each set this runs the full pipeline and reports:
    A_G        amplitude, against the observed 0.946
    rise       fraction of the cycle spent rising (observed ~0.15)
    R21, phi21 Fourier parameters, the standard light-curve shape metrics
    RMS        residual against the Gaia G curve after optimal shift and offset

RMS is the ranking number: it is the only one that uses the whole curve.

    ./scan_convection.py                 # all four sets
    ./scan_convection.py A B             # just these
    ./scan_convection.py --timeout 3600 D

Each run took 18 min at alfam = 0.20. Sets with turbulent pressure may differ.
target.toml is restored on exit, including on Ctrl-C.

VERIFY THE SET VALUES BELOW against $MESA_DIR/star/defaults/controls.defaults
and the RSP test-suite inlists before trusting the results. They are from
Paxton et al. 2019 Table 4 as I recall it, which is not the same as checked.
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
ARCHIVE = ROOT / "scan_convection"

SETS = {
    "A": dict(rsp_gammar=0.0, rsp_alfap=0.0, rsp_alfat=0.0),
    "B": dict(rsp_gammar=1.0, rsp_alfap=0.0, rsp_alfat=0.0),
    "C": dict(rsp_gammar=0.0, rsp_alfap=0.6666666667, rsp_alfat=0.01),
    "D": dict(rsp_gammar=1.0, rsp_alfap=0.6666666667, rsp_alfat=0.01),
}


def set_values(values: dict) -> None:
    text = CONFIG.read_text()
    for key, val in values.items():
        text, n = re.subn(rf"^{key} = .*$", f"{key} = {val}", text,
                          flags=re.MULTILINE)
        if n != 1:
            raise SystemExit(f"expected one {key} line in target.toml, found {n}")
    CONFIG.write_text(text)


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


def read_history(path: Path) -> dict:
    lines = path.read_text().splitlines()
    idx = next(i for i, l in enumerate(lines) if "model_number" in l.split())
    names = lines[idx].split()
    data = np.genfromtxt(path, skip_header=idx + 1)
    return {name: data[:, i] for i, name in enumerate(names)}


def model_curve(h: dict, n_bins: int = 300):
    """Median-combine every complete cycle onto a phase grid."""
    ph = h["rsp_phase"] % 1.0
    mg = h["G"]
    good = np.isfinite(ph) & np.isfinite(mg) & (h["rsp_num_periods"] >= 1)
    ph, mg = ph[good], mg[good]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
    centres, medians = [], []
    for b in range(n_bins):
        sel = mg[idx == b]
        if sel.size >= 3:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            medians.append(np.median(sel))
    return np.asarray(centres), np.asarray(medians)


def rise_fraction(phase, mag) -> float:
    """Fraction of the cycle spent brightening (magnitude decreasing)."""
    i_max = int(np.argmin(mag))      # brightest
    i_min = int(np.argmax(mag))      # faintest
    return float((phase[i_max] - phase[i_min]) % 1.0)


def fourier(phase, mag):
    """R21 and phi21 from a 4-term fit; the standard shape parameters."""
    cols = [np.ones_like(phase)]
    for k in range(1, 5):
        cols += [np.cos(2 * np.pi * k * phase), np.sin(2 * np.pi * k * phase)]
    coef, *_ = np.linalg.lstsq(np.vstack(cols).T, mag, rcond=None)
    amps, phis = [], []
    for k in range(1, 5):
        a, b = coef[2 * k - 1], coef[2 * k]
        amps.append(np.hypot(a, b))
        phis.append(np.arctan2(-b, a))
    r21 = amps[1] / amps[0] if amps[0] else np.nan
    phi21 = (phis[1] - 2 * phis[0]) % (2 * np.pi)
    return float(r21), float(phi21)


def observed_g():
    obs = ROOT / "observations" / "gaia_epoch_photometry.csv"
    if not obs.is_file():
        return None, None
    rows = [l.split(",") for l in obs.read_text().splitlines()[1:] if l.strip()]
    rows = [r for r in rows if r[1].strip().upper() == "G"]
    if not rows:
        return None, None
    t = np.array([float(r[0]) for r in rows])
    m = np.array([float(r[2]) for r in rows])
    with CONFIG.open("rb") as fh:
        p_obs = float(tomllib.load(fh)["observations"]["period_days"])
    return ((t - t.min()) / p_obs) % 1.0, m


def best_rms(mp, mm, op, om) -> float:
    x = np.concatenate([mp - 1, mp, mp + 1])
    y = np.concatenate([mm, mm, mm])
    best = np.inf
    for shift in np.linspace(0, 1, 500, endpoint=False):
        pred = np.interp((op - shift) % 1.0, x, y)
        pred = pred + np.median(om - pred)
        best = min(best, float(np.sqrt(np.mean((om - pred) ** 2))))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=None,
                    help="which sets to run (default: A B C D)")
    ap.add_argument("--timeout", type=float, default=2700.0,
                    help="seconds per run (default 2700; a run took 18 min)")
    args = ap.parse_args()

    names = [s.upper() for s in (args.sets or ["A", "B", "C", "D"])]
    for name in names:
        if name not in SETS:
            raise SystemExit(f"unknown set {name!r}; choose from A B C D")

    backup = CONFIG.read_text()
    ARCHIVE.mkdir(exist_ok=True)
    op, om = observed_g()
    obs_rise = rise_fraction(*(lambda o: (np.sort(o[0]), o[1][np.argsort(o[0])]))
                             ((op, om))) if op is not None else None
    obs_amp = float(om.max() - om.min()) if om is not None else None

    if obs_amp:
        print(f"observed: A_G = {obs_amp:.3f}, rise fraction = {obs_rise:.3f}")
    print(f"\n{'set':<5} {'A_G':<8} {'rise':<7} {'R21':<7} {'phi21':<7} "
          f"{'RMS':<8} {'P(d)':<10} {'min':<6} status")
    print(f"{'---':<5} {'---':<8} {'----':<7} {'---':<7} {'-----':<7} "
          f"{'---':<8} {'----':<10} {'---':<6} ------")

    rows = []
    try:
        for name in names:
            set_values(SETS[name])
            clear_outputs()
            log = ARCHIVE / f"set{name}.log"
            started = time.time()
            try:
                with log.open("w") as fh:
                    rc = subprocess.run(["./rn"], cwd=ROOT, stdout=fh,
                                        stderr=subprocess.STDOUT,
                                        timeout=args.timeout).returncode
                note = "ok" if rc == 0 else f"rc={rc}"
            except subprocess.TimeoutExpired:
                rc, note = -1, "TIMEOUT"
            except OSError as exc:
                rc, note = -1, f"error: {exc}"
            mins = (time.time() - started) / 60

            hist = ROOT / "LOGS_colors" / "history.data"
            if rc != 0 or not hist.is_file():
                print(f"{name:<5} {'--':<8} {'--':<7} {'--':<7} {'--':<7} "
                      f"{'--':<8} {'--':<10} {mins:<6.1f} {note} (see {log.name})")
                rows.append((name, None, None, None, None, None, None, mins, note))
                continue

            try:
                h = read_history(hist)
                mp, mm = model_curve(h)
                amp = float(mm.max() - mm.min())
                period = float(np.nanmedian(h["rsp_period_in_days"]))
                rise = rise_fraction(mp, mm)
                r21, phi21 = fourier(mp, mm)
                rms = best_rms(mp, mm, op, om) if op is not None else np.nan
            except Exception as exc:
                print(f"{name:<5} analysis failed: {exc}")
                rows.append((name, None, None, None, None, None, None, mins,
                             "analysis failed"))
                continue

            print(f"{name:<5} {amp:<8.3f} {rise:<7.3f} {r21:<7.3f} "
                  f"{phi21:<7.3f} {rms:<8.4f} {period:<10.5f} {mins:<6.1f} ok")
            rows.append((name, amp, rise, r21, phi21, rms, period, mins, "ok"))

            dest = ARCHIVE / f"LOGS_colors_set{name}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ROOT / "LOGS_colors", dest)

            with (ARCHIVE / "results.csv").open("w") as fh:
                fh.write("set,g_amplitude,rise_fraction,R21,phi21,rms_vs_gaia,"
                         "period_days,runtime_min,status\n")
                for r in rows:
                    fh.write(",".join("" if v is None else
                                      (f"{v:.6f}" if isinstance(v, float) else str(v))
                                      for v in r) + "\n")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        CONFIG.write_text(backup)
        print(f"\ntarget.toml restored; runs archived in {ARCHIVE}/")

    ok = [r for r in rows if r[5] is not None and np.isfinite(r[5])]
    if ok:
        best = min(ok, key=lambda r: r[5])
        print(f"\nlowest RMS: set {best[0]} ({best[5]:.4f})")
        print(f"set rsp_gammar/rsp_alfap/rsp_alfat in target.toml to "
              f"{SETS[best[0]]} and replot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
