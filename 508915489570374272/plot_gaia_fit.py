#!/usr/bin/env python3
"""
ApJ-quality Gaia light-curve comparison figure, in the style of the FN Lyr one.

Stacked panels sharing the x-axis with no vertical gap:
  - top (2 height units): Gaia epoch photometry + the MESA RSP + MESA Custom
    Colors model, both over two pulsation phases
  - bottom (1 height unit): residuals (observed - model)

With --bands G,BP,RP the figure becomes a column per band, same layout.

Two deliberate differences from the FN Lyr script:

  1. The model curve is the median of ALL complete cycles in the Colors run,
     not the single late cycle that best matches the data. Picking the best
     cycle rewards cycle-to-cycle numerical noise; the limit cycle is the
     physical object and the median is the honest estimate of it.

  2. The vertical offset is fitted but reported, and --no-offset disables it.
     Gaia gives real apparent magnitudes, so once the distance modulus and
     extinction are in target.toml the offset should be zero by construction.
     A large residual offset then means the extinction or distance is wrong,
     which is information rather than a nuisance to be absorbed.

The phase shift is always free: the epoch of the model is arbitrary.

Usage:
    python plot_gaia_fit.py
    python plot_gaia_fit.py --bands G,BP,RP --out figures/fig_gaia_fit.pdf
    python plot_gaia_fit.py --no-offset
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent

# Gaia band -> MESA Custom Colors history column
BAND_COL = {"G": "G", "BP": "Gbp", "RP": "Grp"}

N_SHIFTS = 2500
N_PHASE_BINS = 300
MIN_POINTS_PER_BIN = 3


def read_history(path: Path) -> pd.DataFrame:
    lines = Path(path).read_text().splitlines()
    header_idx = next(i for i, l in enumerate(lines) if "model_number" in l.split())
    return pd.read_csv(path, sep=r"\s+", skiprows=header_idx)


def periodic_interp(phase, mag, phase_new):
    phase = np.asarray(phase, float) % 1.0
    mag = np.asarray(mag, float)
    phase_new = np.asarray(phase_new, float) % 1.0
    good = np.isfinite(phase) & np.isfinite(mag)
    phase, mag = phase[good], mag[good]
    order = np.argsort(phase)
    phase, mag = phase[order], mag[order]
    _, uniq = np.unique(np.round(phase, 8), return_index=True)
    phase, mag = phase[np.sort(uniq)], mag[np.sort(uniq)]
    x = np.concatenate([phase - 1.0, phase, phase + 1.0])
    y = np.concatenate([mag, mag, mag])
    return np.interp(phase_new, x, y)


def representative_cycle(h: pd.DataFrame, col: str):
    """Median-combine every complete cycle onto a common phase grid."""
    ph = np.asarray(h["rsp_phase"], float) % 1.0
    mg = np.asarray(h[col], float)
    ncyc = np.asarray(h["rsp_num_periods"], float)

    good = np.isfinite(ph) & np.isfinite(mg) & (ncyc >= 1)
    ph, mg = ph[good], mg[good]
    if ph.size == 0:
        raise RuntimeError(f"no usable {col} points with rsp_num_periods >= 1")

    edges = np.linspace(0.0, 1.0, N_PHASE_BINS + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, N_PHASE_BINS - 1)

    centres, medians = [], []
    for b in range(N_PHASE_BINS):
        sel = mg[idx == b]
        if sel.size >= MIN_POINTS_PER_BIN:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            medians.append(np.median(sel))
    if len(centres) < 20:
        raise RuntimeError(f"only {len(centres)} populated phase bins for {col}")
    return np.asarray(centres), np.asarray(medians)


def best_shift(model_phase, model_mag, obs_phase, obs_mag, fit_offset: bool):
    shifts = np.linspace(0, 1, N_SHIFTS, endpoint=False)
    best = None
    for shift in shifts:
        m_eval = periodic_interp(model_phase, model_mag, (obs_phase - shift) % 1.0)
        offset = float(np.nanmedian(obs_mag - m_eval)) if fit_offset else 0.0
        resid = obs_mag - (m_eval + offset)
        rms = float(np.sqrt(np.nanmean(resid ** 2)))
        if best is None or rms < best["rms"]:
            best = dict(shift=float(shift), offset=offset, rms=rms)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="LOGS_colors/history.data")
    ap.add_argument("--obs", default=None,
                    help="default: [fit] observations_file in target.toml")
    ap.add_argument("--bands", default="G", help="e.g. G  or  G,BP,RP")
    ap.add_argument("--no-offset", action="store_true",
                    help="do not fit a vertical magnitude offset")
    ap.add_argument("--out", default="figures/fig_gaia_fit.pdf")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    with (ROOT / "target.toml").open("rb") as fh:
        cfg = tomllib.load(fh)

    p_obs = float(cfg["observations"]["period_days"])
    dist = float(cfg["fit"].get("distance_pc", 0.0)) or float(
        cfg["observations"]["distance_pc"])
    mu = 5.0 * np.log10(dist) - 5.0
    ext = {"G": float(cfg["fit"].get("extinction_G", 0.0)),
           "BP": float(cfg["fit"].get("extinction_Gbp", 0.0)),
           "RP": float(cfg["fit"].get("extinction_Grp", 0.0))}

    bands = [b.strip().upper() for b in args.bands.split(",") if b.strip()]
    for b in bands:
        if b not in BAND_COL:
            raise SystemExit(f"unknown band {b!r}; choose from G, BP, RP")

    h = read_history(ROOT / args.history)
    model_period = float(np.nanmedian(h["rsp_period_in_days"]))

    obs_path = Path(args.obs) if args.obs else ROOT / cfg["fit"]["observations_file"]
    if not obs_path.is_file():
        raise SystemExit(f"missing observations: {obs_path}\n"
                         f"run fetch_gaia_target.py first")
    obs_all = pd.read_csv(obs_path)
    # Gaia epoch tables can carry masked entries that survive as strings, so the
    # numeric columns come back as object dtype. Coerce and drop what is not a
    # real measurement.
    for col in ("time", "mag", "mag_err"):
        if col in obs_all.columns:
            obs_all[col] = pd.to_numeric(obs_all[col], errors="coerce")
    obs_all["band"] = obs_all["band"].astype(str).str.strip().str.upper()
    obs_all = obs_all[np.isfinite(obs_all["time"]) & np.isfinite(obs_all["mag"])]
    if obs_all.empty:
        raise SystemExit(f"no usable epochs in {obs_path}")
    t0 = float(obs_all["time"].min())

    print(f"observed period  {p_obs:.6f} d")
    print(f"model period     {model_period:.6f} d  "
          f"({100*(model_period/p_obs - 1):+.2f}%)")
    print(f"distance modulus {mu:.4f}  (d = {dist:.1f} pc)")
    if not any(ext.values()):
        print("WARNING: extinction is zero; the fitted offset absorbs it")

    fig = plt.figure(figsize=(7.0 if len(bands) == 1 else 4.2 * len(bands), 5.0))
    gs = GridSpec(2, len(bands), height_ratios=[2, 1], hspace=0.0,
                  wspace=0.28, figure=fig)

    for k, band in enumerate(bands):
        obs = obs_all[obs_all["band"] == band]
        if len(obs) == 0:
            raise SystemExit(f"no {band} epochs in {obs_path}")
        obs_phase = (((obs["time"].to_numpy(float) - t0) / p_obs) % 1.0)
        obs_mag = obs["mag"].to_numpy(float)
        obs_err = None
        if "mag_err" in obs:
            obs_err = obs["mag_err"].to_numpy(float)
            if not np.all(np.isfinite(obs_err)):
                obs_err = None

        mp, mm = representative_cycle(h, BAND_COL[band])
        mm = mm + mu + ext[band]

        al = best_shift(mp, mm, obs_phase, obs_mag, not args.no_offset)

        mp_shift = (mp + al["shift"]) % 1.0
        order = np.argsort(mp_shift)
        mp1, mm1 = mp_shift[order], mm[order] + al["offset"]
        mp2 = np.concatenate([mp1, mp1 + 1.0])
        mm2 = np.concatenate([mm1, mm1])

        op2 = np.concatenate([obs_phase, obs_phase + 1.0])
        om2 = np.concatenate([obs_mag, obs_mag])
        oe2 = np.concatenate([obs_err, obs_err]) if obs_err is not None else None

        m_at_obs = periodic_interp(mp, mm, (obs_phase - al["shift"]) % 1.0)
        resid = obs_mag - (m_at_obs + al["offset"])
        rr2 = np.concatenate([resid, resid])

        model_amp = float(np.nanmax(mm1) - np.nanmin(mm1))
        obs_amp = float(np.nanmax(obs_mag) - np.nanmin(obs_mag))
        pred = m_at_obs + al["offset"]
        corr = float(np.corrcoef(obs_mag - np.nanmean(obs_mag),
                                 pred - np.nanmean(pred))[0, 1])

        print(f"\n[{band}]  shift={al['shift']:.4f}  offset={al['offset']:+.4f}  "
              f"RMS={al['rms']:.4f}")
        print(f"      A_obs={obs_amp:.3f}  A_mod={model_amp:.3f}  r={corr:.4f}")

        ax_lc = fig.add_subplot(gs[0, k])
        ax_re = fig.add_subplot(gs[1, k], sharex=ax_lc)

        if oe2 is not None:
            ax_lc.errorbar(op2, om2, yerr=oe2, fmt=".", ms=4.0, elinewidth=0.6,
                           color="#3b6ea5", alpha=0.85,
                           label=f"Gaia DR3 {band}", rasterized=True)
        else:
            ax_lc.plot(op2, om2, ".", ms=4.0, color="#3b6ea5", alpha=0.85,
                       label=f"Gaia DR3 {band}", rasterized=True)
        ax_lc.plot(mp2, mm2, "-", lw=1.6, color="#d1622b",
                   label="MESA RSP + MESA Custom Colors")
        ax_lc.invert_yaxis()
        ax_lc.set_ylabel(f"Gaia {band} (mag)")
        ax_lc.legend(loc="upper center", fontsize=8.5, handlelength=1.6)
        plt.setp(ax_lc.get_xticklabels(), visible=False)

        txt = (f"$P_{{\\rm obs}} = {p_obs:.6f}$, "
               f"$P_{{\\rm mod}} = {model_period:.6f}$ d\n"
               f"$A_{{\\rm obs}} = {obs_amp:.3f}$, "
               f"$A_{{\\rm mod}} = {model_amp:.3f}$ mag")
        ax_lc.text(0.015, 0.04, txt, transform=ax_lc.transAxes, fontsize=7.8,
                   va="bottom", ha="left",
                   bbox=dict(boxstyle="round,pad=0.35", fc="white",
                             ec="0.7", lw=0.6))

        ax_re.axhline(0.0, lw=0.8, color="0.4")
        ax_re.plot(np.concatenate([obs_phase, obs_phase + 1.0]), rr2, ".",
                   ms=4.0, color="#3b6ea5", alpha=0.85, rasterized=True)
        ax_re.set_xlabel("Pulsation phase")
        ax_re.set_ylabel("O $-$ C")
        rmax = np.nanmax(np.abs(rr2)) * 1.15
        if np.isfinite(rmax) and rmax > 0:
            ax_re.set_ylim(rmax, -rmax)
        ax_re.set_xlim(0, 2)

    if args.title:
        fig.suptitle(args.title, y=0.98, fontsize=11)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=250)
    print(f"\nwrote {out} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
