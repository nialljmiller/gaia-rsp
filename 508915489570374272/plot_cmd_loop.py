#!/usr/bin/env python3
"""Plot the MESA Colors CMD loop, optionally over Gaia epoch photometry.

Reads LOGS_colors/history.data (MESA Colors writes G, Gbp, Grp as absolute
magnitudes at the 10 pc model distance), converts to apparent magnitudes with
the distance modulus and band extinctions from target.toml, and plots the
G vs BP-RP loop plus the phased light curves.

    python plot_cmd_loop.py
    python plot_cmd_loop.py --cycles 3 --no-observed

Extinction and distance come from [fit] and [observations] in target.toml. With
extinction left at zero the plot is intrinsic and will not sit on top of the
observed loop -- run fetch_gaia_target.py first.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load_history(path: Path) -> np.ndarray:
    """MESA history.data: 5 header lines, column names on line 6, data after."""
    return np.genfromtxt(path, names=True, skip_header=5)


def periodic_interp(phase: np.ndarray, mag: np.ndarray,
                     phase_new: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, float) % 1.0
    mag = np.asarray(mag, float)
    phase_new = np.asarray(phase_new, float) % 1.0
    order = np.argsort(phase)
    phase, mag = phase[order], mag[order]
    _, uniq = np.unique(np.round(phase, 8), return_index=True)
    phase, mag = phase[np.sort(uniq)], mag[np.sort(uniq)]
    x = np.concatenate([phase - 1.0, phase, phase + 1.0])
    y = np.concatenate([mag, mag, mag])
    return np.interp(phase_new, x, y)


def best_phase_shift(model_phase: np.ndarray, model_mag: np.ndarray,
                      obs_grid: np.ndarray, obs_mag: np.ndarray,
                      n: int = 720) -> tuple[float, float, float]:
    """The model's epoch is arbitrary; find the shift that lines it up with
    the observed phase grid (free additive offset, since only shape matters
    here -- the mean-magnitude offset is a separate, already-reported check).
    """
    best_rms, best_shift, best_off = np.inf, 0.0, 0.0
    for s in np.linspace(0, 1, n, endpoint=False):
        pred = periodic_interp(model_phase, model_mag, (obs_grid - s) % 1.0)
        off = float(np.median(obs_mag - pred))
        rms = float(np.sqrt(np.mean((obs_mag - (pred + off)) ** 2)))
        if rms < best_rms:
            best_rms, best_shift, best_off = rms, s, off
    return best_shift, best_off, best_rms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="LOGS_colors/history.data")
    ap.add_argument("--cycles", type=float, default=None,
                    help="how many of the final periods to plot (default: all)")
    ap.add_argument("--no-observed", action="store_true")
    ap.add_argument("--out", default="cmd_loop.pdf")
    args = ap.parse_args()

    with (ROOT / "target.toml").open("rb") as fh:
        cfg = tomllib.load(fh)

    hist_path = ROOT / args.history
    if not hist_path.is_file():
        raise SystemExit(f"missing {hist_path}")
    h = load_history(hist_path)

    for col in ("G", "Gbp", "Grp"):
        if col not in h.dtype.names:
            raise SystemExit(f"{hist_path} has no {col} column -- was Colors on?")

    day = h["star_age_day"]
    G, BP, RP = h["G"], h["Gbp"], h["Grp"]

    good = np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP)
    day, G, BP, RP = day[good], G[good], BP[good], RP[good]

    period = float(np.nanmedian(h["rsp_period_in_days"][good]))
    if args.cycles:
        keep = day >= day.max() - args.cycles * period
        day, G, BP, RP = day[keep], G[keep], BP[keep], RP[keep]

    # --- distance and extinction -----------------------------------------
    dist = float(cfg["fit"].get("distance_pc", 0.0))
    if dist <= 0:
        dist = float(cfg["observations"]["distance_pc"])
    mu = 5.0 * np.log10(dist) - 5.0

    A_G = float(cfg["fit"].get("extinction_G", 0.0))
    A_BP = float(cfg["fit"].get("extinction_Gbp", 0.0))
    A_RP = float(cfg["fit"].get("extinction_Grp", 0.0))
    reddened = any(a > 0 for a in (A_G, A_BP, A_RP))

    g_app = G + mu + A_G
    bp_app = BP + mu + A_BP
    rp_app = RP + mu + A_RP
    colour = bp_app - rp_app

    phase = ((day - day.min()) / period) % 1.0

    print(f"model period      {period:.5f} d")
    print(f"distance modulus  {mu:.4f}  (d = {dist:.1f} pc)")
    print(f"extinction        A_G={A_G:.3f} A_BP={A_BP:.3f} A_RP={A_RP:.3f}")
    print(f"mean G            {g_app.mean():.4f}")
    print(f"G amplitude       {g_app.max() - g_app.min():.4f}")
    print(f"BP-RP range       {colour.min():.4f} to {colour.max():.4f}")
    if not reddened:
        print("WARNING: extinction is zero -- this is an intrinsic loop")

    # --- loop geometry, for comparison with a straight-line fit ----------
    slope, intercept = np.polyfit(colour, g_app, 1)
    extent = np.hypot(np.ptp(colour), np.ptp(g_app))
    print(f"loop major-axis slope  {slope:.3f} mag/mag")
    print(f"loop extent            {extent:.3f} mag")

    # --- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    obs = ROOT / cfg["fit"]["observations_file"]
    if obs.is_file() and not args.no_observed:
        o = np.genfromtxt(obs, delimiter=",", names=True, dtype=None, encoding=None)
        o = o[np.isfinite(o["time"]) & np.isfinite(o["mag"])]
        bandcol = np.char.upper(np.char.strip(o["band"].astype(str)))
        bands = {b: o[bandcol == b] for b in np.unique(bandcol)}

        # CRITICAL: fold the observations on the OBSERVED period, not the model
        # period. Gaia's baseline is ~1000 d, so folding 68 cycles on a period
        # that is wrong by 0.09 d accumulates ~0.4 in phase and smears the
        # curve into noise. Only the model is folded on the model period.
        p_fold = float(cfg["observations"]["period_days"])
        t0 = min(b["time"].min() for b in bands.values())

        if {"G", "BP", "RP"} <= set(bands):
            grid = np.linspace(0, 1, 200)
            interp = {}
            for b in ("G", "BP", "RP"):
                ph = ((bands[b]["time"] - t0) / p_fold) % 1.0
                order = np.argsort(ph)
                interp[b] = np.interp(grid, ph[order], bands[b]["mag"][order],
                                      period=1.0)
            obs_colour = interp["BP"] - interp["RP"]
            obs_g = interp["G"]
            ax1.plot(obs_colour, obs_g, "o", ms=3, color="0.5",
                     label="Gaia DR3", zorder=1)

            # The model's epoch (day.min()) has no relation to the observed
            # epoch (t0) -- align them the same way plot_gaia_fit.py does, or
            # the phased light-curve panel compares unrelated phases.
            shift, g_off, shift_rms = best_phase_shift(phase, g_app, grid, obs_g)
            print(f"\nphase alignment (G)  shift={shift:.4f}  "
                  f"offset={g_off:+.4f} mag  RMS={shift_rms:.4f}")
            phase = (phase + shift) % 1.0

            # The model is intrinsic; the observations are reddened. The offset
            # between them measures the extinction directly.
            d_colour = float(np.mean(obs_colour) - np.mean(colour))
            d_g = float(np.mean(obs_g) - np.mean(g_app))
            print("\n--- reddening implied by the model/observed offset ---")
            print(f"  E(BP-RP) = {d_colour:+.4f}")
            print(f"  A_G      = {d_g:+.4f}")
            if abs(d_colour) > 1e-3:
                print(f"  A_G/E(BP-RP) = {d_g/d_colour:.2f}   "
                      f"(expect about 2.0 for Gaia)")
            print("  for target.toml, if you adopt these:")
            print(f"    extinction_G   = {d_g:.4f}")
            print(f"    extinction_Gbp = {d_g/0.789*1.002:.4f}")
            print(f"    extinction_Grp = {d_g/0.789*0.589:.4f}")
            print("  NOTE: this forces the model onto the data by construction.")
            print("  A 3D dust map value is the independent check.")

        for b, style in (("G", "o"), ("BP", "s"), ("RP", "^")):
            if b in bands:
                ph = ((bands[b]["time"] - t0) / p_fold) % 1.0
                ax2.plot(ph, bands[b]["mag"], style, ms=3, color="0.6")

    sc = ax1.scatter(colour, g_app, c=phase, cmap="twilight", s=6, zorder=2)
    ax1.plot(colour, slope * colour + intercept, "k--", lw=0.8, alpha=0.6)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$")
    ax1.set_ylabel(r"$G$" + ("" if reddened else "  (intrinsic)"))
    ax1.set_title(f"CMD loop, P = {period:.4f} d")
    fig.colorbar(sc, ax=ax1, label="phase")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="best", fontsize=8)

    for arr, lab in ((g_app, "G"), (bp_app, "BP"), (rp_app, "RP")):
        ax2.plot(phase, arr, ".", ms=3, label=lab)
    ax2.invert_yaxis()
    ax2.set_xlabel("phase")
    ax2.set_ylabel("magnitude")
    ax2.set_title("phased light curves")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = ROOT / args.out
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
