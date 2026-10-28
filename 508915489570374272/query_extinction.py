#!/usr/bin/env python3
"""Query a 3D dust map for the extinction toward the target.

GSP-Phot returned null for this star, so there is no Gaia extinction estimate.
The alternative to deriving it from the model/observed offset -- which forces
the match by construction -- is a 3D dust map at the Bailer-Jones distance.

Bayestar19 (Green et al. 2019) covers dec > -30 and is the right choice here
(dec = +57.8). The first run downloads ~500 MB.

    ./query_extinction.py                    # uses target.toml coords/distance
    ./query_extinction.py --ra 24.3084 --dec 57.7592 --distance 2990
    ./query_extinction.py --download         # fetch the map first

CONVERSION CAVEAT: Bayestar19 reports reddening in its own units, not E(B-V)
directly. Green et al. give A_V = 3.32 * E(Bayestar19). The Gaia band ratios
A_G/A_V = 0.789, A_BP/A_V = 1.002, A_RP/A_V = 0.589 are mean values and are
formally colour-dependent, which matters at the few-percent level for a star
that changes colour by 0.44 mag over its cycle. Treat the output as good to
~10%, not better.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

A_V_PER_BAYESTAR = 3.32     # Green et al. 2019
GAIA_OVER_A_V = {"G": 0.789, "BP": 1.002, "RP": 0.589}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, default=None)
    ap.add_argument("--dec", type=float, default=None)
    ap.add_argument("--distance", type=float, default=None, help="pc")
    ap.add_argument("--download", action="store_true",
                    help="download the Bayestar19 map (~500 MB) and exit")
    args = ap.parse_args()

    if args.download:
        from dustmaps.config import config
        import dustmaps.bayestar
        print(f"data dir: {config['data_dir']}")
        dustmaps.bayestar.fetch(version="bayestar2019")
        print("done; rerun without --download")
        return 0

    try:
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from dustmaps.bayestar import BayestarQuery
    except ImportError as exc:
        print(f"missing dependency: {exc}\n  pip install dustmaps astropy",
              file=sys.stderr)
        return 1

    ra, dec, dist = args.ra, args.dec, args.distance
    if dist is None:
        with (ROOT / "target.toml").open("rb") as fh:
            cfg = tomllib.load(fh)
        dist = float(cfg["observations"]["distance_pc"])
    if ra is None or dec is None:
        print("--ra and --dec required (from fetch_gaia_target.py output)",
              file=sys.stderr)
        return 1

    coords = SkyCoord(ra * u.deg, dec * u.deg,
                      distance=dist * u.pc, frame="icrs")
    print(f"ra={ra:.4f} dec={dec:.4f} d={dist:.0f} pc")
    print(f"galactic l={coords.galactic.l.deg:.3f} b={coords.galactic.b.deg:.3f}")

    try:
        bq = BayestarQuery(version="bayestar2019")
    except Exception as exc:
        print(f"\ncould not load Bayestar19: {exc}", file=sys.stderr)
        print("run with --download first", file=sys.stderr)
        return 2

    samples = bq(coords, mode="samples")
    median = float(np.median(samples))
    lo, hi = (float(np.percentile(samples, p)) for p in (16, 84))

    a_v = median * A_V_PER_BAYESTAR
    print(f"\nBayestar19 reddening = {median:.4f} "
          f"(16th-84th: {lo:.4f}-{hi:.4f})")
    print(f"A_V = {a_v:.4f}   (A_V = {A_V_PER_BAYESTAR} x Bayestar)")

    print("\nfor target.toml:")
    for band, key in (("G", "extinction_G"), ("BP", "extinction_Gbp"),
                      ("RP", "extinction_Grp")):
        print(f"    {key:<16} = {a_v * GAIA_OVER_A_V[band]:.4f}")

    e_bprp = a_v * (GAIA_OVER_A_V["BP"] - GAIA_OVER_A_V["RP"])
    print(f"\nimplied E(BP-RP) = {e_bprp:.4f}, "
          f"A_G/E(BP-RP) = {GAIA_OVER_A_V['G']/(GAIA_OVER_A_V['BP']-GAIA_OVER_A_V['RP']):.2f}")
    print("compare with the offset-derived values from plot_cmd_loop.py.")
    print("Agreement supports the model's absolute magnitude scale;")
    print("disagreement means part of the offset is model bias, not dust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
