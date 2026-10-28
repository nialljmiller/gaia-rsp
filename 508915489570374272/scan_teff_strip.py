#!/usr/bin/env python3
"""Map the instability strip in Teff at fixed period, re-solving L at each step.

Everything so far has been done at Teff = 5400 K, which is the first temperature
that pulsates at all -- 5500 K was damped. That means the adopted model sits
right against the blue edge, and blue-edge models are characteristically
low-amplitude and sinusoidal. The sawtooth asymmetry strengthens toward the
strip interior, so cooler models are the untested direction.

For each Teff this root-finds the luminosity giving the target linear period,
then reports the growth rate and the mode ratios. Growth rising as Teff falls
means there is strip room below 5400; growth falling means 5400 was already past
the middle and the red edge is close.

    ./scan_teff_strip.py 5150 5200 5250 5300 5350 5400
    ./scan_teff_strip.py --target-p0 14.74 --mass 8.0 5250 5300

Roughly 15 s per LINA evaluation, 2-4 evaluations per temperature.
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
LOGDIR = ROOT / "scan_teff"

L_EXPONENT = 0.919      # P ~ L^0.919 at fixed M, Teff
TEFF_EXPONENT = -3.255  # P ~ Teff^-3.255 at fixed M, L


def patch(text: str, key: str, value: str) -> str:
    new, n = re.subn(rf"^(\s*){key} = .*$", rf"\g<1>{key} = {value}", text,
                     flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"expected one {key} line in {LINA.name}, found {n}")
    return new


def run_lina(mass: float, lum: float, teff: float, base: str, timeout: float):
    text = patch(base, "RSP_mass", f"{mass:.6f}d0")
    text = patch(text, "RSP_L", f"{lum:.6f}d0")
    text = patch(text, "RSP_Teff", f"{teff:.6f}d0")
    (ROOT / "inlist").write_text(text)

    result = ROOT / "LINA_RESULT"
    if result.exists():
        result.unlink()

    log = LOGDIR / f"t{teff:.0f}_l{lum:.0f}.log"
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
    ap.add_argument("teffs", nargs="+", type=float)
    ap.add_argument("--target-p0", type=float, default=14.736)
    ap.add_argument("--mass", type=float, default=8.0)
    ap.add_argument("--l-start", type=float, default=6560.0)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    if not LINA.is_file():
        raise SystemExit(f"missing {LINA}")
    LOGDIR.mkdir(exist_ok=True)
    base = LINA.read_text()

    print(f"M = {args.mass}, matching linear P0 = {args.target_p0:.4f} d")
    print("growth rising as Teff falls => strip room below the current 5400 K\n")
    print(f"{'Teff':<8} {'L':<9} {'P0(d)':<9} {'growth':<13} {'P2/P0':<8} "
          f"{'n':<3} status")
    print(f"{'----':<8} {'-':<9} {'-----':<9} {'------':<13} {'-----':<8} "
          f"{'-':<3} ------")

    rows = []
    started = time.time()
    for teff in args.teffs:
        # First guess for L: hold the period as Teff changes, P ~ Teff^-3.255.
        lum = args.l_start * (teff / 5400.0) ** (-TEFF_EXPONENT / L_EXPONENT)
        best = None
        for i in range(args.max_iter):
            res = run_lina(args.mass, lum, teff, base, args.timeout)
            if res is None:
                print(f"{teff:<8.0f} {lum:<9.0f} {'--':<9} {'--':<13} "
                      f"{'--':<8} {i+1:<3} BUILD-FAILED")
                best = None
                break
            best = (lum, res)
            if abs(res["p0"] / args.target_p0 - 1.0) < args.tol:
                break
            lum *= (args.target_p0 / res["p0"]) ** (1.0 / L_EXPONENT)

        if best is None:
            continue
        lum, res = best
        r20 = res["p2"] / res["p0"] if res["p0"] else float("nan")
        print(f"{teff:<8.0f} {lum:<9.0f} {res['p0']:<9.4f} "
              f"{res['growth']:<13.4e} {r20:<8.4f} {i+1:<3} {res['status']}")
        rows.append((teff, lum, res["p0"], res["growth"], r20, res["status"]))

    if rows:
        with (LOGDIR / "results.csv").open("w") as fh:
            fh.write("teff_k,luminosity_lsun,p0_days,growth,p2_over_p0,status\n")
            for r in rows:
                fh.write(",".join(str(v) for v in r) + "\n")

        excited = [r for r in rows if r[3] > 0]
        if excited:
            best = max(excited, key=lambda r: r[3])
            print(f"\nstrongest growth: Teff = {best[0]:.0f}, L = {best[1]:.0f}, "
                  f"growth = {best[3]:.3e}")
            print("that is the deepest point in the strip tested; run a full")
            print("settle there and compare the rise fraction against 0.41")
        else:
            print("\nnothing excited in this range")

    print(f"\n{(time.time()-started)/60:.1f} min; logs in {LOGDIR.name}/; "
          f"{LINA.name} was not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
