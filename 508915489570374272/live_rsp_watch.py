#!/usr/bin/env python3
"""
Live monitor for a two-stage MESA RSP radial-pulsator Colors run.

Usage from the work directory, e.g.

    cd ~/MESA/MESA_Colors_Tests/rsp_RR_Lyrae
    python live_rsp_watch.py

It auto-picks the newest readable history file among:
    LOGS_colors/history.data
    LOGS_settle/history.data
    LOGS/history.data

The display refreshes in-place and shows current run status, stability checks,
and simple ASCII plots of recent quantities.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print("ERROR: this monitor needs numpy and pandas.", file=sys.stderr)
    print(f"Import failure: {exc}", file=sys.stderr)
    sys.exit(1)

DEFAULT_HISTORY_CANDIDATES = [
    Path("LOGS_colors/history.data"),
    Path("LOGS_settle/history.data"),
    Path("LOGS/history.data"),
]

COLOR_COLUMNS = [
    "u", "g", "r", "i", "z", "y",
    "U", "B", "V", "R", "I",
    "Gbp", "G", "Grp", "Grvs",
    "TESS", "Kepler",
    "F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213",
]


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")


def pick_history_file(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    existing = [p for p in DEFAULT_HISTORY_CANDIDATES if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def read_mesa_history(path: Path) -> pd.DataFrame:
    # MESA may be writing this file while we read it. Read text first so a
    # partial write only corrupts this refresh, not the next one.
    text = path.read_text(errors="ignore")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "model_number" in line.split():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"could not find model_number header in {path}")

    from io import StringIO

    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(body), sep=r"\s+", engine="python")

    # Drop malformed repeated header rows if they appear after restarts/appends.
    if "model_number" in df.columns:
        df = df[pd.to_numeric(df["model_number"], errors="coerce").notna()].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df.reset_index(drop=True)


def first_present(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def format_float(x: float | int | None, precision: int = 4) -> str:
    if x is None:
        return "--"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "--"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.{precision}e}"
    return f"{x:.{precision}f}"


def ascii_plot(
    values: Sequence[float],
    label: str,
    width: int,
    height: int = 8,
    invert: bool = False,
    log_abs: bool = False,
) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{label}: no finite data"

    if arr.size > width:
        arr = arr[-width:]

    if log_abs:
        arr = np.log10(np.maximum(np.abs(arr), 1e-99))
        plot_label = f"log10|{label}|"
    else:
        plot_label = label

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return f"{label}: no finite data"
    if vmax == vmin:
        vmax = vmin + 1.0

    canvas = [[" " for _ in range(arr.size)] for _ in range(height)]
    for i, val in enumerate(arr):
        frac = (val - vmin) / (vmax - vmin)
        if invert:
            frac = 1.0 - frac
        row = int(round((height - 1) * (1.0 - frac)))
        row = max(0, min(height - 1, row))
        canvas[row][i] = "*"

    lines = []
    lines.append(f"{plot_label}  min={format_float(vmin)}  max={format_float(vmax)}  last={format_float(arr[-1])}")
    for r, row in enumerate(canvas):
        if r == 0:
            tick = format_float(vmax if not invert else vmin, 3).rjust(10)
        elif r == height - 1:
            tick = format_float(vmin if not invert else vmax, 3).rjust(10)
        else:
            tick = " " * 10
        lines.append(f"{tick} |{''.join(row)}|")
    lines.append(" " * 11 + "+" + "-" * arr.size + "+")
    return "\n".join(lines)


def stability_from_history(df: pd.DataFrame, min_periods: int, tol: float, needed: int) -> tuple[int, int, float | None]:
    if "rsp_num_periods" not in df.columns or "rsp_GREKM" not in df.columns:
        return 0, 0, None

    sub = df[["rsp_num_periods", "rsp_GREKM"]].dropna().copy()
    if sub.empty:
        return 0, 0, None

    # Use the last row reported for each completed period.
    sub["period_int"] = np.floor(pd.to_numeric(sub["rsp_num_periods"], errors="coerce")).astype("Int64")
    sub = sub.dropna(subset=["period_int"])
    if sub.empty:
        return 0, 0, None

    last_by_period = sub.groupby("period_int", as_index=False).tail(1)
    current_period = int(last_by_period["period_int"].iloc[-1])
    current_grekm = float(last_by_period["rsp_GREKM"].iloc[-1])

    stable_count = 0
    for _, row in last_by_period.iloc[::-1].iterrows():
        p = int(row["period_int"])
        g = float(row["rsp_GREKM"])
        if p < min_periods:
            break
        if abs(g) <= tol:
            stable_count += 1
        else:
            break
    return current_period, min(stable_count, needed), current_grekm


def progress_bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def summarize(df: pd.DataFrame, path: Path, args: argparse.Namespace) -> str:
    cols = df.columns
    last = df.iloc[-1]
    term_width = shutil.get_terminal_size((120, 30)).columns
    plot_width = max(30, min(args.plot_width, term_width - 16))

    period_col = "rsp_num_periods" if "rsp_num_periods" in cols else None
    grekm_col = "rsp_GREKM" if "rsp_GREKM" in cols else None
    teff_col = first_present(df, ["Teff", "effective_T", "log_Teff"])
    logl_col = "log_L" if "log_L" in cols else None
    logg_col = "log_g" if "log_g" in cols else None
    r_col = first_present(df, ["radius", "R", "photosphere_r", "log_R"])
    model_col = "model_number" if "model_number" in cols else None
    age_col = "star_age" if "star_age" in cols else None
    color_cols = [c for c in COLOR_COLUMNS if c in cols]

    current_period, stable_count, current_grekm = stability_from_history(
        df, args.min_periods, args.grekm_tol, args.stable_needed
    )

    stage = "unknown"
    if "colors" in str(path).lower() or color_cols:
        stage = "colors"
    elif "settle" in str(path).lower():
        stage = "settling"

    lines: list[str] = []
    lines.append("RSP radial-pulsator live monitor")
    lines.append(f"history: {path}   rows: {len(df)}   stage: {stage}   refreshed: {time.strftime('%H:%M:%S')}")
    lines.append("-" * min(term_width, 120))

    model = int(last[model_col]) if model_col else None
    age = float(last[age_col]) if age_col else None
    period = float(last[period_col]) if period_col else None
    teff_val = float(last[teff_col]) if teff_col else None
    if teff_col == "log_Teff" and teff_val is not None:
        teff_val = 10.0 ** teff_val
    logl = float(last[logl_col]) if logl_col else None
    logg = float(last[logg_col]) if logg_col else None
    rad = float(last[r_col]) if r_col else None

    lines.append(
        "model={model}  period={period}  age={age} yr  Teff={teff} K  logL={logl}  logg={logg}  R/logR={rad}".format(
            model=model if model is not None else "--",
            period=format_float(period, 3),
            age=format_float(age, 4),
            teff=format_float(teff_val, 2),
            logl=format_float(logl, 4),
            logg=format_float(logg, 4),
            rad=format_float(rad, 4),
        )
    )

    if grekm_col:
        stable_frac = stable_count / max(1, args.stable_needed)
        lines.append(
            f"rsp_GREKM={format_float(current_grekm, 5)}  "
            f"stability: {progress_bar(stable_frac)} {stable_count}/{args.stable_needed} "
            f"after min_periods={args.min_periods}, tol={args.grekm_tol:g}"
        )
        if current_period < args.min_periods:
            lines.append(f"settling gate: period {current_period}/{args.min_periods} before stability counting begins")
        elif stable_count >= args.stable_needed:
            lines.append("stability gate: satisfied; stage 1 should save settled.mod and stop soon")
        else:
            lines.append("stability gate: not yet satisfied")

    if color_cols:
        shown = color_cols[:8]
        vals = "  ".join(f"{c}={format_float(float(last[c]), 4)}" for c in shown)
        lines.append(f"colors: {vals}")
        if "g" in cols and "r" in cols:
            lines.append(f"g-r={format_float(float(last['g']) - float(last['r']), 5)}")
        elif "Gbp" in cols and "Grp" in cols:
            lines.append(f"Gbp-Grp={format_float(float(last['Gbp']) - float(last['Grp']), 5)}")
    else:
        lines.append("colors: not active in this history yet")

    lines.append("-" * min(term_width, 120))

    window_df = df.tail(args.window)
    if teff_col:
        vals = window_df[teff_col].astype(float).to_numpy()
        if teff_col == "log_Teff":
            vals = 10.0 ** vals
        lines.append(ascii_plot(vals, "Teff", plot_width, height=args.height))
        lines.append("")

    if grekm_col:
        lines.append(ascii_plot(window_df[grekm_col].astype(float).to_numpy(), "rsp_GREKM", plot_width, height=args.height, log_abs=True))
        lines.append("")

    mag_col = None
    for candidate in ["g", "G", "F146", "V", "Kepler", "TESS", "Mag_bol"]:
        if candidate in cols:
            mag_col = candidate
            break
    if mag_col:
        # Magnitudes are inverted physically: smaller is brighter. Plot as observed mag values with note.
        lines.append(ascii_plot(window_df[mag_col].astype(float).to_numpy(), f"{mag_col} mag", plot_width, height=args.height, invert=False))
        lines.append("")

    if period_col and mag_col:
        # Phase-ish recent cycle display for the last completed/current period, when possible.
        pvals = window_df[period_col].astype(float).to_numpy()
        mvals = window_df[mag_col].astype(float).to_numpy()
        if pvals.size > 0:
            pnow = math.floor(float(pvals[-1]))
            mask = np.floor(pvals) == pnow
            if mask.sum() >= 5:
                lines.append(ascii_plot(mvals[mask], f"current-cycle {mag_col} mag", plot_width, height=args.height, invert=False))
                lines.append("")

    lines.append("Ctrl-C to quit. Use --path LOGS_settle/history.data or --path LOGS_colors/history.data to force a file.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live ASCII monitor for MESA RSP radial-pulsator runs.")
    parser.add_argument("--path", default=None, help="Explicit history.data path. Default: newest of LOGS_colors, LOGS_settle, LOGS.")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--window", type=int, default=180, help="Number of recent rows used for plots.")
    parser.add_argument("--plot-width", type=int, default=88, help="Maximum ASCII plot width in characters.")
    parser.add_argument("--height", type=int, default=8, help="ASCII plot height in rows.")
    parser.add_argument("--min-periods", type=int, default=80, help="Minimum periods before stability counting.")
    parser.add_argument("--grekm-tol", type=float, default=1e-3, help="Stability tolerance for abs(rsp_GREKM).")
    parser.add_argument("--stable-needed", type=int, default=8, help="Consecutive stable periods needed.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between refreshes.")
    args = parser.parse_args()

    while True:
        path = pick_history_file(args.path)
        if path is None:
            if not args.no_clear:
                clear_screen()
            print("Waiting for a history file...")
            print("Looked for: " + ", ".join(str(p) for p in DEFAULT_HISTORY_CANDIDATES))
            time.sleep(args.interval)
            continue

        try:
            df = read_mesa_history(path)
            if len(df) == 0:
                raise ValueError("history file has zero data rows")
            output = summarize(df, path, args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            output = f"Waiting for readable history data in {path}\n{type(exc).__name__}: {exc}"

        if not args.no_clear:
            clear_screen()
        print(output, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nmonitor stopped")
        raise SystemExit(0)
