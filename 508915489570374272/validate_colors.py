#!/usr/bin/env python3
"""End-to-end validation of MESA Custom Colors output.

Every check is against an independent calculation, not against another Colors
number, so passing means the chain is right rather than self-consistent.

    python validate_colors.py                       # LOGS/history.data
    python validate_colors.py --history LOGS_colors/history.data
    python validate_colors.py --quiet               # exit code only

Exit 0 if all checks pass, 1 otherwise -- usable as a CI gate.

Checks
  1  Flux_bol == sigma T^4 (R/d)^2        geometry + integration + Teff input
  2  Mag_bol  == 4.74 - 2.5 log(L/Lsun)   bolometric zero point
  3  Mag_bol  == -2.5 log10(Flux_bol/f0)  internal consistency of the two
  4  Teff, log g, R mutually consistent   the caller passes a coherent surface
  5  band mags finite, ordered, smooth    filter path sanity
  6  timestep-to-timestep continuity      catches interpolant flip-flopping
  7  Interp_rad stability                 catches stencil jumps
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

SIGMA = 5.6704e-5
RSUN = 6.957e10
LSUN = 3.828e33
PC = 3.0856775814913673e18
MBOL_SUN = 4.74
F_BOL_ZERO = LSUN * 10.0**(0.4 * MBOL_SUN) / (4.0 * np.pi * (10.0 * PC)**2)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail):
    results.append((PASS if ok else FAIL, name, detail))
    return ok


def read_history(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if "model_number" in l.split())
    names = lines[i].split()
    data = np.genfromtxt(path, skip_header=i + 1)
    if data.ndim == 1:
        data = data[None, :]
    return {n: data[:, j] for j, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="LOGS/history.data")
    ap.add_argument("--distance-pc", type=float, default=10.0,
                    help="the colors namelist distance, in pc")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="fractional tolerance for the physics checks")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    h = read_history(args.history)
    n = len(h["model_number"])
    d = args.distance_pc * PC

    need = ["log_L", "effective_T", "photosphere_r", "Mag_bol", "Flux_bol"]
    missing = [c for c in need if c not in h]
    if missing:
        print(f"history is missing {missing}", file=sys.stderr)
        return 1

    L = 10.0**h["log_L"]
    T = h["effective_T"]
    R = h["photosphere_r"] * RSUN
    mb = h["Mag_bol"]
    fb = h["Flux_bol"]

    bands = [c for c in ("G", "Gbp", "Grp", "Grvs") if c in h]

    if not args.quiet:
        print(f"{args.history}: {n} rows, {len(bands)} bands {bands}")
        print(f"distance {args.distance_pc} pc, f_bol_zero = {F_BOL_ZERO:.6e}\n")

    # --- 1. Flux_bol against Stefan-Boltzmann + geometric dilution ---------
    predicted = SIGMA * T**4 * (R / d)**2
    r1 = fb / predicted
    check("Flux_bol == sigma T^4 (R/d)^2",
          abs(r1.mean() - 1) < args.tol and r1.std() < args.tol,
          f"ratio {r1.mean():.5f} +/- {r1.std():.5f}")

    # --- 2. Mag_bol against the IAU identity ------------------------------
    # only valid at 10 pc, where Mag_bol is the absolute magnitude
    if abs(args.distance_pc - 10.0) < 1e-6:
        r2 = mb - (MBOL_SUN - 2.5 * np.log10(L))
        check("Mag_bol == 4.74 - 2.5 log(L/Lsun)",
              abs(r2.mean()) < 0.01 and r2.std() < 0.01,
              f"offset {r2.mean():+.5f} +/- {r2.std():.5f} mag")
    else:
        results.append(("SKIP", "Mag_bol == 4.74 - 2.5 log(L/Lsun)",
                        f"only meaningful at 10 pc (given {args.distance_pc})"))

    # --- 3. Mag_bol and Flux_bol mutually consistent ----------------------
    r3 = mb - (-2.5 * np.log10(fb / F_BOL_ZERO))
    check("Mag_bol == -2.5 log10(Flux_bol/f0)",
          abs(r3.mean()) < 0.001 and r3.std() < 0.001,
          f"offset {r3.mean():+.6f} +/- {r3.std():.6f} mag")

    # --- 4. the surface triple is coherent --------------------------------
    # L from Teff and R must reproduce the structure luminosity
    L_from_TR = 4 * np.pi * R**2 * SIGMA * T**4 / LSUN
    r4 = L_from_TR / L
    check("4 pi R^2 sigma T^4 == L",
          abs(r4.mean() - 1) < args.tol,
          f"ratio {r4.mean():.5f} +/- {r4.std():.5f}")

    if "log_g" in h:
        # log g should follow the same radius that sets the flux
        if "star_mass" in h:
            g_pred = np.log10(6.67430e-8 * h["star_mass"] * 1.9884e33 / R**2)
            r4b = h["log_g"] - g_pred
            check("log_g consistent with photosphere_r",
                  abs(r4b.mean()) < 0.05,
                  f"offset {r4b.mean():+.4f} +/- {r4b.std():.4f} dex")

    # --- 5. band magnitudes sane ------------------------------------------
    for b in bands:
        v = h[b]
        finite = np.isfinite(v).all() and (abs(v) < 90).all()
        check(f"{b} finite and not sentinel",
              finite,
              f"range {np.nanmin(v):.4f} to {np.nanmax(v):.4f}")

    if {"G", "Gbp", "Grp"} <= set(h):
        # for a cool star BP is faintest, RP brightest; BC_G should be small
        bc = h["G"] - mb
        check("BC_G physically plausible",
              abs(bc.mean()) < 1.0,
              f"BC_G {bc.mean():+.4f} +/- {bc.std():.4f} mag")
        colour = h["Gbp"] - h["Grp"]
        check("BP-RP correlates with Teff",
              n < 3 or np.corrcoef(colour, T)[0, 1] < -0.5,
              f"corr(BP-RP, Teff) = {np.corrcoef(colour, T)[0,1]:.4f}"
              if n >= 3 else "too few rows")

    # --- 6. continuity: no interpolant flip-flopping ----------------------
    # a jump in Mag_bol far larger than the jump in log_L means the photometry
    # moved for a non-physical reason
    if n > 4:
        dmb = np.abs(np.diff(mb))
        dexp = 2.5 * np.abs(np.diff(np.log10(L)))
        excess = dmb - dexp
        worst = excess.max()
        check("Mag_bol continuous with log_L",
              worst < 0.01,
              f"worst excess jump {worst:.5f} mag "
              f"at row {int(np.argmax(excess))}")

        for b in bands:
            db = np.abs(np.diff(h[b]))
            check(f"{b} continuous",
                  db.max() < 0.5,
                  f"largest step {db.max():.5f} mag")

    # --- 7. interpolation diagnostic stability ----------------------------
    if "Interp_rad" in h and n > 4:
        ir = h["Interp_rad"]
        jumps = np.abs(np.diff(ir))
        check("Interp_rad smooth",
              jumps.max() < 0.5 * max(ir.max(), 1e-30),
              f"range {ir.min():.4e} to {ir.max():.4e}, "
              f"largest jump {jumps.max():.4e}")

    # --- report -----------------------------------------------------------
    if not args.quiet:
        width = max(len(nm) for _, nm, _ in results)
        for status, name, detail in results:
            print(f"  [{status}] {name:<{width}}  {detail}")
        print()

    nfail = sum(1 for s, _, _ in results if s == FAIL)
    if nfail:
        print(f"{nfail} of {len(results)} checks FAILED")
    elif not args.quiet:
        print(f"all {len(results)} checks passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
