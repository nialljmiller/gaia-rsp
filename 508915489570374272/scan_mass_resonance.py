#!/usr/bin/env python3
"""Scan mass at fixed pulsation period and report the P2/P0 resonance ratio.

The remaining light-curve failures -- peak phase late by ~0.08 and the missing
bump near phase 1.0 -- are the Hertzsprung progression, which is driven by the
P2/P0 ratio passing through 0.5. That ratio depends on mass, and mass was never
fitted: M = 8 was taken from the PMLT relation at the start and the period was
then matched with luminosity alone. Many (M, L) pairs give the same period and
different P2/P0.

For each mass this root-finds the luminosity that reproduces the target linear
period, then reports P0, P1, P2, the ratios, and the growth rate. Each LINA
evaluation is ~15 s and each mass needs 2-4 of them.

    ./scan_mass_resonance.py 6.0 6.5 7.0 7.5 8.0 8.5 9.0
    ./scan_mass_resonance.py --target-p0 14.74 7.0 8.0

The target period is the LINEAR one. The nonlinear period ran 0.36% longer, so
for an observed 14.789 d the linear target is about 14.736 d.

inlist_rsp_LINA is read but never modified.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINA = ROOT / "inlist_rsp_LINA"
LOGDIR = ROOT / "scan_mass"

# P ~ L^0.919 at fixed M and Teff (De Somma+2022 PMLT), used to step L.
L_EXPONENT = 0.919


def patch(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf"^(\s*){key} = .*$", rf"\g<1>{key} = {value}", text,
                     flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"expected one {key} line in {LINA.name}, found {n}")
    return new


def run_lina(mass: float, lum: float, base: str, timeout: float):
    text = patch(base, "RSP_mass", f"{mass:.6f}d0")
    text = patch(text, "RSP_L", f"{lum:.6f}d0")
    (ROOT / "inlist").write_text(text)

    result = ROOT / "LINA_RESULT"
    if result.exists():
        result.unlink()

    log = LOGDIR / f"m{mass}_l{lum:.0f}.log"
    try:
        with log.open("w") as fh:
            subprocess.run(["./rn1"], cwd=ROOT, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None

    if not (result.is_file() and result.stat().st_size):
        return None
    fields = {}
    for line in result.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()

    def get(key):
        try:
            return float(fields[key])
        except (KeyError, ValueError):
            return float("nan")

    return dict(p0=get("mode_0_period_days"), p1=get("mode_1_period_days"),
                p2=get("mode_2_period_days"), growth=get("mode_0_growth"),
                status=fields.get("status", "?"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("masses", nargs="+", type=float)
    ap.add_argument("--target-p0", type=float, default=14.736,
                    help="linear period to match (default 14.736)")
    ap.add_argument("--l-start", type=float, default=6560.0)
    ap.add_argument("--tol", type=float, default=0.002,
                    help="fractional period tolerance (default 0.2%%)")
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    if not LINA.is_file():
        raise SystemExit(f"missing {LINA}")
    LOGDIR.mkdir(exist_ok=True)
    base = LINA.read_text()

    print(f"matching linear P0 = {args.target_p0:.4f} d by adjusting L at each mass")
    print("resonance for the Hertzsprung bump is at P2/P0 = 0.5\n")
    print(f"{'M':<7} {'L':<9} {'P0(d)':<9} {'P1/P0':<8} {'P2/P0':<8} "
          f"{'growth':<12} {'n':<3} status")
    print(f"{'-':<7} {'-':<9} {'-----':<9} {'-----':<8} {'-----':<8} "
          f"{'------':<12} {'-':<3} ------")

    rows = []
    started = time.time()
    for mass in args.masses:
        lum = args.l_start
        best = None
        for i in range(args.max_iter):
            res = run_lina(mass, lum, base, args.timeout)
            if res is None:
                print(f"{mass:<7.2f} {lum:<9.0f} {'--':<9} {'--':<8} {'--':<8} "
                      f"{'--':<12} {i+1:<3} BUILD-FAILED")
                best = None
                break
            best = (lum, res)
            err = res["p0"] / args.target_p0 - 1.0
            if abs(err) < args.tol:
                break
            # P ~ L^0.919, so scale L to close the gap.
            lum *= (args.target_p0 / res["p0"]) ** (1.0 / L_EXPONENT)

        if best is None:
            continue
        lum, res = best
        r10 = res["p1"] / res["p0"] if res["p0"] else float("nan")
        r20 = res["p2"] / res["p0"] if res["p0"] else float("nan")
        flag = "  <-- near resonance" if abs(r20 - 0.5) < 0.01 else ""
        print(f"{mass:<7.2f} {lum:<9.0f} {res['p0']:<9.4f} {r10:<8.4f} "
              f"{r20:<8.4f} {res['growth']:<12.4e} {i+1:<3} {res['status']}{flag}")
        rows.append((mass, lum, res["p0"], r10, r20, res["growth"], res["status"]))

    if rows:
        with (LOGDIR / "results.csv").open("w") as fh:
            fh.write("mass_msun,luminosity_lsun,p0_days,p1_over_p0,"
                     "p2_over_p0,growth,status\n")
            for r in rows:
                fh.write(",".join(str(v) for v in r) + "\n")

        excited = [r for r in rows if r[5] > 0]
        if excited:
            closest = min(excited, key=lambda r: abs(r[4] - 0.5))
            print(f"\nclosest to the P2/P0 = 0.5 resonance among excited models:")
            print(f"  M = {closest[0]}, L = {closest[1]:.0f}, "
                  f"P2/P0 = {closest[4]:.4f}, growth = {closest[5]:.3e}")
        else:
            print("\nno excited models in this mass range at this period")

    print(f"\n{(time.time()-started)/60:.1f} min; logs in {LOGDIR.name}/; "
          f"{LINA.name} was not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
