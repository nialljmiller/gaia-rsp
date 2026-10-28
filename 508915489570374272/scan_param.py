#!/usr/bin/env python3
"""Scan any [run] parameter in target.toml and rank by fit to the Gaia curve.

Generalises scan_alfam.py and scan_convection.py: same metrics, any key.

    ./scan_param.py rsp_alfa 1.3 1.5 1.7 1.9
    ./scan_param.py rsp_alfam 0.20 0.22 0.24
    ./scan_param.py --timeout 3600 rsp_alfa 1.9

Reports per value:
    A_G        G amplitude, against the observed 0.946
    rise       fraction of the cycle spent brightening (observed 0.287)
    R21, phi21 Fourier shape parameters
    RMS        residual against the Gaia G curve after optimal shift and offset

RMS is the ranking number. Note that it mixes amplitude and shape error, so a
value that improves shape while shifting amplitude can score worse -- check the
rise fraction separately before concluding a parameter did nothing.

target.toml is restored on exit, including on Ctrl-C. Each run's LOGS_colors is
archived. Runs take 10-20 min, so plan the timeout from a measured run.
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


def set_value(key: str, value: float) -> None:
    text = CONFIG.read_text()
    new, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", text,
                     flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"expected one {key!r} line in target.toml, found {n}")
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


def read_history(path: Path) -> dict:
    lines = path.read_text().splitlines()
    idx = next(i for i, l in enumerate(lines) if "model_number" in l.split())
    names = lines[idx].split()
    data = np.genfromtxt(path, skip_header=idx + 1)
    return {name: data[:, i] for i, name in enumerate(names)}


def model_curve(h: dict, n_bins: int = 300):
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
    return float((phase[int(np.argmin(mag))] - phase[int(np.argmax(mag))]) % 1.0)


def fourier(phase, mag):
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
    return float(r21), float((phis[1] - 2 * phis[0]) % (2 * np.pi))


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
    ph = ((t - t.min()) / p_obs) % 1.0
    order = np.argsort(ph)
    return ph[order], m[order]


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
    ap.add_argument("key", help="a key under [run] in target.toml")
    ap.add_argument("values", nargs="+", type=float)
    ap.add_argument("--timeout", type=float, default=2700.0)
    args = ap.parse_args()

    with CONFIG.open("rb") as fh:
        run_section = tomllib.load(fh)["run"]
    if args.key not in run_section:
        raise SystemExit(f"{args.key!r} is not in [run]; have: "
                         f"{', '.join(sorted(run_section))}")

    archive = ROOT / f"scan_{args.key}"
    archive.mkdir(exist_ok=True)
    backup = CONFIG.read_text()
    op, om = observed_g()

    if om is not None:
        print(f"observed: A_G = {om.max()-om.min():.3f}, "
              f"rise fraction = {rise_fraction(op, om):.3f}")
    print(f"\n{args.key:<10} {'A_G':<8} {'rise':<7} {'R21':<7} {'phi21':<7} "
          f"{'RMS':<8} {'P(d)':<10} {'min':<6} status")
    print(f"{'-'*10:<10} {'---':<8} {'----':<7} {'---':<7} {'-----':<7} "
          f"{'---':<8} {'----':<10} {'---':<6} ------")

    rows = []
    try:
        for value in args.values:
            set_value(args.key, value)
            clear_outputs()
            log = archive / f"{args.key}{value}.log"
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
                print(f"{value:<10} {'--':<8} {'--':<7} {'--':<7} {'--':<7} "
                      f"{'--':<8} {'--':<10} {mins:<6.1f} {note} ({log.name})")
                rows.append((value, None, None, None, None, None, None, mins, note))
                continue

            try:
                h = read_history(hist)
                mp, mm = model_curve(h)
                amp = float(mm.max() - mm.min())
                period = float(np.nanmedian(h["rsp_period_in_days"]))
                rise = rise_fraction(mp, mm)
                r21, phi21 = fourier(mp, mm)
                rms = best_rms(mp, mm, op, om) if op is not None else float("nan")
            except Exception as exc:
                print(f"{value:<10} analysis failed: {exc}")
                rows.append((value, None, None, None, None, None, None, mins,
                             "analysis failed"))
                continue

            print(f"{value:<10} {amp:<8.3f} {rise:<7.3f} {r21:<7.3f} "
                  f"{phi21:<7.3f} {rms:<8.4f} {period:<10.5f} {mins:<6.1f} ok")
            rows.append((value, amp, rise, r21, phi21, rms, period, mins, "ok"))

            dest = archive / f"LOGS_colors_{args.key}{value}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ROOT / "LOGS_colors", dest)

            with (archive / "results.csv").open("w") as fh:
                fh.write(f"{args.key},g_amplitude,rise_fraction,R21,phi21,"
                         f"rms_vs_gaia,period_days,runtime_min,status\n")
                for r in rows:
                    fh.write(",".join("" if v is None else str(v) for v in r) + "\n")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        CONFIG.write_text(backup)
        print(f"\ntarget.toml restored; runs archived in {archive.name}/")

    ok = [r for r in rows if r[5] is not None and np.isfinite(r[5])]
    if ok:
        best = min(ok, key=lambda r: r[5])
        print(f"lowest RMS: {args.key} = {best[0]} (RMS {best[5]:.4f}, "
              f"rise {best[2]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
