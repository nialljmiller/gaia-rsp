#!/usr/bin/env python3
"""Pull everything needed for the CMD comparison from the Gaia archive.

Reads source_id from target.toml, then fetches:
  - vari_cepheid: mode, fundamental/overtone periods, mean mags, amplitudes
  - astrophysical_parameters: Teff, [M/H], A_G, A_0, FLAME L/R/M
  - external.gaiaedr3_distance: Bailer-Jones geometric/photogeometric distance
  - DataLink epoch photometry -> observations/gaia_epoch_photometry.csv

Requires: pip install astroquery

    python fetch_gaia_target.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OBS = ROOT / "observations"

QUERY = """
SELECT s.source_id, s.ra, s.dec, s.parallax, s.parallax_error,
       s.phot_g_mean_mag, s.phot_bp_mean_mag, s.phot_rp_mean_mag, s.bp_rp,
       s.phot_variable_flag, s.has_epoch_photometry,
       v.num_selected_g_fov, v.in_vari_cepheid,
       c.type_best_classification, c.mode_best_classification,
       c.pf, c.pf_error, c.p1_o, c.p1_o_error,
       c.int_average_g, c.peak_to_peak_g,
       c.int_average_bp, c.peak_to_peak_bp,
       c.int_average_rp, c.peak_to_peak_rp,
       a.teff_gspphot, a.mh_gspphot, a.ag_gspphot, a.azero_gspphot,
       a.ebpminrp_gspphot, a.lum_flame, a.radius_flame, a.mass_flame,
       d.r_med_geo, d.r_med_photogeo
FROM gaiadr3.gaia_source AS s
LEFT JOIN gaiadr3.vari_summary AS v USING (source_id)
LEFT JOIN gaiadr3.vari_cepheid AS c USING (source_id)
LEFT JOIN gaiadr3.astrophysical_parameters AS a USING (source_id)
LEFT JOIN external.gaiaedr3_distance AS d USING (source_id)
WHERE s.source_id = {source_id}
"""


def main() -> int:
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        print("need astroquery:  pip install astroquery", file=sys.stderr)
        return 1

    with (ROOT / "target.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    source_id = int(cfg["target"]["source_id"])

    print(f"querying Gaia DR3 for {source_id} ...")
    job = Gaia.launch_job(QUERY.format(source_id=source_id))
    tab = job.get_results()
    if len(tab) == 0:
        print("no row returned; check the source_id", file=sys.stderr)
        return 2

    row = tab[0]
    print()
    for name in tab.colnames:
        value = row[name]
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode()
        print(f"  {name:32s} {value}")

    # --- the numbers that matter for target.toml -------------------------
    print("\n--- for target.toml ---")

    ag = row["ag_gspphot"]
    if ag is not None and np.isfinite(ag):
        # A_BP and A_RP from A_G using mean Gaia band ratios. These are
        # colour-dependent in general; for a single Cepheid at fixed mean
        # colour the constant ratios are adequate. A_G/A_V = 0.789,
        # A_BP/A_V = 1.002, A_RP/A_V = 0.589 (Gaia DR3 extinction law).
        a_v = ag / 0.789
        print(f"  extinction_G   = {ag:.4f}")
        print(f"  extinction_Gbp = {a_v * 1.002:.4f}")
        print(f"  extinction_Grp = {a_v * 0.589:.4f}")
    else:
        print("  ag_gspphot is null -- use a 3D dust map at the distance below")

    dist = row["r_med_photogeo"]
    if dist is not None and np.isfinite(dist):
        print(f"  distance_pc    = {dist:.1f}   (Bailer-Jones photogeometric)")

    for key, label in [("pf", "fundamental period"), ("p1_o", "first overtone")]:
        val = row[key]
        if val is not None and np.isfinite(val):
            print(f"  {label:22s} {val:.6f} d")

    lum = row["lum_flame"]
    if lum is not None and np.isfinite(lum):
        print(f"  FLAME luminosity        {lum:.1f} Lsun   "
              f"(currently using {cfg['stellar']['luminosity_lsun']})")

    # --- epoch photometry -------------------------------------------------
    if not row["has_epoch_photometry"]:
        print("\nno epoch photometry for this source")
        return 0

    print("\nretrieving epoch photometry ...")
    products = Gaia.load_data(
        ids=[source_id],
        data_release="Gaia DR3",
        retrieval_type="EPOCH_PHOTOMETRY",
        data_structure="INDIVIDUAL",
        valid_data=True,
    )
    key = next(iter(products))
    ep = products[key]
    # astroquery returns this differently across versions: a bare Table, a
    # one-element list holding one, a VOTableFile, or a votable TableElement.
    # Unwrap and convert until the object actually has columns.
    for _ in range(4):
        if hasattr(ep, "colnames"):
            break
        if hasattr(ep, "to_table"):          # votable TableElement
            ep = ep.to_table()
            continue
        if hasattr(ep, "get_first_table"):   # VOTableFile
            ep = ep.get_first_table()
            continue
        try:
            ep = ep[0]
        except (TypeError, IndexError, KeyError):
            break
    if not hasattr(ep, "colnames"):
        print(f"unexpected epoch product type: {type(ep)}", file=sys.stderr)
        return 3

    cols = set(ep.colnames)

    # Two layouts exist. Long: one row per band with time/band/mag/flux columns.
    # Wide: one row per transit with g_transit_*, bp_*, rp_* columns and
    # per-band reject flags. Normalise both to (time, band, mag, mag_err).
    if {"time", "band", "mag", "flux", "flux_error"} <= cols:
        layout = "long"
    elif {"g_transit_time", "g_transit_mag"} <= cols:
        layout = "wide"
    else:
        print(f"unrecognised epoch table layout: {sorted(cols)}", file=sys.stderr)
        return 3

    def finite(value):
        if value is None or value is np.ma.masked:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def mag_err(flux, ferr):
        if flux is None or ferr is None or flux <= 0:
            return 0.0
        return 2.5 / np.log(10) * ferr / flux

    records = []
    if layout == "long":
        has_reject = "rejected_by_photometry" in cols
        for r in ep:
            if has_reject and bool(r["rejected_by_photometry"]):
                continue
            t, m = finite(r["time"]), finite(r["mag"])
            if t is None or m is None:
                continue
            band = r["band"]
            if isinstance(band, (bytes, np.bytes_)):
                band = band.decode()
            records.append((t, str(band).strip().upper(), m,
                            mag_err(finite(r["flux"]), finite(r["flux_error"]))))
    else:
        spec = [
            ("G", "g_transit_time", "g_transit_mag",
             "g_transit_flux", "g_transit_flux_error", "variability_flag_g_reject"),
            ("BP", "bp_obs_time", "bp_mag",
             "bp_flux", "bp_flux_error", "variability_flag_bp_reject"),
            ("RP", "rp_obs_time", "rp_mag",
             "rp_flux", "rp_flux_error", "variability_flag_rp_reject"),
        ]
        global_reject = "rejected_by_photometry" in cols
        for r in ep:
            if global_reject and bool(r["rejected_by_photometry"]):
                continue
            for band, tcol, mcol, fcol, ecol, rcol in spec:
                if tcol not in cols or mcol not in cols:
                    continue
                if rcol in cols:
                    flag = r[rcol]
                    if flag is not np.ma.masked and bool(flag):
                        continue
                t, m = finite(r[tcol]), finite(r[mcol])
                if t is None or m is None:
                    continue
                flux = finite(r[fcol]) if fcol in cols else None
                ferr = finite(r[ecol]) if ecol in cols else None
                records.append((t, band, m, mag_err(flux, ferr)))

    if not records:
        print("no surviving epochs after quality cuts", file=sys.stderr)
        return 3

    records.sort()
    OBS.mkdir(exist_ok=True)
    out = OBS / "gaia_epoch_photometry.csv"
    with out.open("w") as fh:
        fh.write("time,band,mag,mag_err\n")
        for t, band, m, e in records:
            fh.write(f"{t:.8f},{band},{m:.6f},{e:.6f}\n")

    counts = {b: sum(1 for r in records if r[1] == b) for b in ("G", "BP", "RP")}
    print(f"wrote {len(records)} epochs to {out}")
    print(f"  G={counts['G']}  BP={counts['BP']}  RP={counts['RP']}")
    print("times are barycentric JD (TCB) - 2455197.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
