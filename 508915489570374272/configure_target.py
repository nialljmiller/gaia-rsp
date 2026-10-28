#!/usr/bin/env python3
"""Render the two-stage MESA inlists from target.toml."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable

PC_CM = 3.0856775814913673e18
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "target.toml"
SETTLE = ROOT / "inlist_rsp_RR_Lyrae"
COLORS = ROOT / "inlist_rsp_RR_Lyrae_colors"
AUTO_VALUES = {"", "auto", "__SET_IN_TARGET_TOML__"}


def value(cfg: dict[str, Any], section: str, key: str) -> Any:
    try:
        return cfg[section][key]
    except KeyError as exc:
        raise KeyError(f"missing [{section}] {key} in {CONFIG}") from exc


def optional(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    """Like value(), but tolerates target.toml files predating this key."""
    return cfg.get(section, {}).get(key, default)


def require_mesa_dir() -> Path:
    raw = os.environ.get("MESA_DIR", "").strip()
    if not raw:
        raise RuntimeError("MESA_DIR is not set")
    mesa_dir = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    required = mesa_dir / "star" / "work" / "Makefile"
    if not required.is_file():
        raise RuntimeError(
            f"MESA_DIR does not point to a usable MESA installation: {mesa_dir}\n"
            f"missing required file: {required}"
        )
    return mesa_dir


def fort_float(x: float) -> str:
    if not math.isfinite(float(x)):
        raise ValueError(f"non-finite inlist value: {x}")
    rendered = f"{float(x):.12g}"
    if "e" in rendered.lower():
        mantissa, exponent = re.split(r"[eE]", rendered)
        return f"{mantissa}d{exponent}"
    return rendered + "d0"


def quote(s: str) -> str:
    if "'" in s:
        raise ValueError("single quotes are not supported in inlist paths")
    return f"'{s}'"


def replace_assignment(text: str, key: str, rendered: str) -> str:
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*=.*$", re.MULTILINE)
    new, count = pattern.subn(lambda m: f"{m.group(1)}{key} = {rendered}", text)
    if count != 1:
        raise RuntimeError(f"expected one assignment for {key}, found {count}")
    return new


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def expand_path(raw: str, mesa_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()

    mesa_relative = mesa_dir / path
    if mesa_relative.exists():
        return mesa_relative.resolve()
    return (ROOT / path).resolve()


def parse_inlist_path(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
    match = pattern.search(path.read_text(errors="ignore"))
    return match.group(1) if match else None


def existing_rrlyrae_inlists(mesa_dir: Path) -> list[Path]:
    candidates = [
        mesa_dir.parent / "MESA_Colors_Tests" / "rsp_RR_Lyrae" / "inlist_rsp_RR_Lyrae_colors",
        Path.home() / "MESA" / "MESA_Colors_Tests" / "rsp_RR_Lyrae" / "inlist_rsp_RR_Lyrae_colors",
        ROOT.parent.parent / "MESA_Colors_Tests" / "rsp_RR_Lyrae" / "inlist_rsp_RR_Lyrae_colors",
    ]
    return [p for p in unique_paths(candidates) if p.is_file()]


def sed_tools_roots(mesa_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("SED_TOOLS_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            Path.home() / "SED_Tools",
            mesa_dir.parent / "SED_Tools",
            Path.home() / "MESA" / "SED_Tools",
            Path("/media/data3/MESA/SED_Tools"),
        ]
    )

    for inlist in existing_rrlyrae_inlists(mesa_dir):
        for key in ("stellar_atm", "instrument"):
            raw = parse_inlist_path(inlist, key)
            if not raw:
                continue
            path = Path(os.path.expandvars(os.path.expanduser(raw)))
            parts = path.parts
            if "data" in parts:
                idx = parts.index("data")
                candidates.append(Path(*parts[:idx]))

    return unique_paths([p.resolve() for p in candidates])


def gaia_candidates(mesa_dir: Path) -> list[Path]:
    # The active MESA installation is intentionally first.
    candidates: list[Path] = [
        mesa_dir / "data" / "colors_data" / "filters" / "GAIA" / "GAIA",
        mesa_dir / "data" / "colors_data" / "filters" / "GAIA",
    ]
    for root in sed_tools_roots(mesa_dir):
        candidates.extend(
            [
                root / "data" / "filters" / "GAIA" / "GAIA",
                root / "filters" / "GAIA" / "GAIA",
            ]
        )
    return unique_paths(candidates)


def atmosphere_candidates(mesa_dir: Path) -> list[Path]:
    candidates: list[Path] = [
        mesa_dir / "data" / "colors_data" / "stellar_models" / "Kurucz2003all"
    ]
    for inlist in existing_rrlyrae_inlists(mesa_dir):
        raw = parse_inlist_path(inlist, "stellar_atm")
        if raw:
            candidates.append(Path(os.path.expandvars(os.path.expanduser(raw))))
    for root in sed_tools_roots(mesa_dir):
        candidates.extend(
            [
                root / "data" / "stellar_models" / "Kurucz2003all",
                root / "stellar_models" / "Kurucz2003all",
            ]
        )
    return unique_paths(candidates)


def vega_candidates(mesa_dir: Path) -> list[Path]:
    candidates: list[Path] = [
        mesa_dir / "data" / "colors_data" / "stellar_models" / "vega_flam.csv",
        mesa_dir / "data" / "colors_data" / "stellar_models" / "vega" / "vega_flam.csv",
        mesa_dir / "data" / "colors_data" / "stellar_models" / "vega" / "vegaflam.csv",
    ]
    for root in sed_tools_roots(mesa_dir):
        candidates.extend(
            [
                root / "data" / "stellar_models" / "vega_flam.csv",
                root / "data" / "stellar_models" / "vega" / "vega_flam.csv",
            ]
        )
    return unique_paths(candidates)


def resolve_required_path(
    raw: str, label: str, candidates: list[Path], mesa_dir: Path
) -> tuple[Path | None, list[Path]]:
    if raw.strip().lower() not in AUTO_VALUES:
        path = expand_path(raw, mesa_dir)
        return (path if path.exists() else None), [path]

    for path in candidates:
        if path.exists():
            return path.resolve(), candidates
    return None, candidates


def format_missing_path(label: str, raw: str, tried: list[Path]) -> str:
    lines = [f"{label} path could not be resolved from {raw!r}.", "Tried:"]
    lines.extend(f"  - {path}" for path in tried)
    return "\n".join(lines)


def render(cfg: dict[str, Any], check_only: bool = False) -> dict[str, Any]:
    mesa_dir = require_mesa_dir()

    mass = float(value(cfg, "stellar", "mass_msun"))
    lum = float(value(cfg, "stellar", "luminosity_lsun"))
    teff = float(value(cfg, "stellar", "teff_k"))
    x = float(value(cfg, "stellar", "x"))
    z = float(value(cfg, "stellar", "z"))

    if not (mass > 0 and lum > 0 and teff > 0):
        raise ValueError("mass, luminosity, and Teff must be positive")
    if not (0 < x < 1 and 0 < z < 1 and x + z < 1):
        raise ValueError("require 0 < X,Z < 1 and X + Z < 1")

    period = float(value(cfg, "observations", "period_days"))
    settle_steps = int(value(cfg, "run", "settle_steps_per_cycle"))
    colors_steps = int(value(cfg, "run", "colors_steps_per_cycle"))
    fallback_dt = float(value(cfg, "run", "fallback_max_dt_seconds"))
    target_mode = int(value(cfg, "run", "target_mode"))
    minimum_growth = float(value(cfg, "run", "minimum_linear_growth"))
    min_delta_r = float(value(cfg, "run", "minimum_delta_r_rsun"))
    min_delta_mag = float(value(cfg, "run", "minimum_delta_mag"))
    amplitude_rel_tol = float(value(cfg, "run", "amplitude_relative_tolerance"))
    kick = float(value(cfg, "run", "kick_km_s"))

    # Convection parameters. alfam (eddy viscosity) sets the limit-cycle amplitude
    # and is the knob for matching observed Gaia peak-to-peak amplitudes. Both
    # stages must agree or the Colors restart is inconsistent with settled.mod.
    rsp_alfa = float(optional(cfg, "run", "rsp_alfa", 1.5))
    rsp_alfam = float(optional(cfg, "run", "rsp_alfam", 0.25))
    max_energy_error = float(optional(cfg, "run", "max_rel_run_energy_error", 1.0e-4))
    rsp_gammar = float(optional(cfg, "run", "rsp_gammar", 0.0))
    rsp_alfap = float(optional(cfg, "run", "rsp_alfap", 0.0))
    rsp_alfat = float(optional(cfg, "run", "rsp_alfat", 0.0))

    if target_mode < 0:
        raise ValueError("[run] target_mode must be zero or greater")
    if not math.isfinite(minimum_growth):
        raise ValueError("[run] minimum_linear_growth must be finite")
    if min_delta_r <= 0 and min_delta_mag <= 0:
        raise ValueError("at least one amplitude floor must be positive")
    if amplitude_rel_tol <= 0:
        raise ValueError("amplitude_relative_tolerance must be positive")
    if kick == 0:
        raise ValueError("kick_km_s cannot be zero because MESA skips LINA when the kick is zero")
    if rsp_alfa <= 0:
        raise ValueError("[run] rsp_alfa must be positive")
    if rsp_alfam < 0:
        raise ValueError("[run] rsp_alfam cannot be negative")
    if max_energy_error <= 0:
        raise ValueError("[run] max_rel_run_energy_error must be positive")
    for name, val in (("rsp_gammar", rsp_gammar), ("rsp_alfap", rsp_alfap),
                      ("rsp_alfat", rsp_alfat)):
        if val < 0:
            raise ValueError(f"[run] {name} cannot be negative")

    # RSP_max_dt overrides RSP_target_steps_per_cycle whenever it is smaller, so it
    # must be derived from the period rather than fixed. A hard-coded value tuned
    # for RR Lyrae silently multiplies the step count for a long-period Cepheid.
    max_dt = fallback_dt
    settle_max_dt = fallback_dt
    if period > 0:
        max_dt = min(fallback_dt, period * 86400.0 / max(colors_steps, 1))
        settle_max_dt = min(fallback_dt, period * 86400.0 / max(settle_steps, 1))

    instrument_raw = str(value(cfg, "paths", "gaia_instrument"))
    atmosphere_raw = str(value(cfg, "paths", "stellar_atmosphere"))
    vega_raw = str(value(cfg, "paths", "vega_sed"))

    instrument, instrument_tried = resolve_required_path(
        instrument_raw, "Gaia instrument", gaia_candidates(mesa_dir), mesa_dir
    )
    stellar_atm, atmosphere_tried = resolve_required_path(
        atmosphere_raw, "stellar atmosphere", atmosphere_candidates(mesa_dir), mesa_dir
    )
    vega_sed, vega_tried = resolve_required_path(
        vega_raw, "Vega SED", vega_candidates(mesa_dir), mesa_dir
    )

    errors: list[str] = []
    if instrument is None:
        errors.append(format_missing_path("Gaia instrument", instrument_raw, instrument_tried))
    if stellar_atm is None:
        errors.append(format_missing_path("stellar atmosphere", atmosphere_raw, atmosphere_tried))
    if vega_sed is None:
        errors.append(format_missing_path("Vega SED", vega_raw, vega_tried))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

    assert instrument is not None
    assert stellar_atm is not None
    assert vega_sed is not None

    mag_system = str(value(cfg, "photometry", "magnitude_system"))
    model_distance_pc = float(value(cfg, "photometry", "model_distance_pc"))

    common = {
        "Zbase": fort_float(z),
        "RSP_mass": fort_float(mass),
        "RSP_Teff": fort_float(teff),
        "RSP_L": fort_float(lum),
        "RSP_X": fort_float(x),
        "RSP_Z": fort_float(z),
        "RSP_alfa": fort_float(rsp_alfa),
        "RSP_alfam": fort_float(rsp_alfam),
        "RSP_gammar": fort_float(rsp_gammar),
        "RSP_alfap": fort_float(rsp_alfap),
        "RSP_alfat": fort_float(rsp_alfat),
    }

    settle_text = SETTLE.read_text()
    colors_text = COLORS.read_text()
    for key, rendered in common.items():
        settle_text = replace_assignment(settle_text, key, rendered)
        colors_text = replace_assignment(colors_text, key, rendered)

    settle_values = {
        "x_integer_ctrl(2)": str(int(value(cfg, "run", "settle_min_periods"))),
        "x_integer_ctrl(3)": str(int(value(cfg, "run", "settle_stable_periods"))),
        "x_integer_ctrl(4)": str(int(value(cfg, "run", "settle_max_periods"))),
        "x_integer_ctrl(6)": str(target_mode),
        "x_ctrl(2)": fort_float(float(value(cfg, "run", "grekm_tolerance"))),
        "x_ctrl(3)": fort_float(minimum_growth),
        "x_ctrl(4)": fort_float(min_delta_r),
        "x_ctrl(5)": fort_float(min_delta_mag),
        "x_ctrl(6)": fort_float(amplitude_rel_tol),
        "x_ctrl(7)": fort_float(max_energy_error),
        "RSP_target_steps_per_cycle": str(settle_steps),
        "RSP_max_dt": fort_float(settle_max_dt),
        "RSP_kick_vsurf_km_per_sec": fort_float(kick),
    }
    for key, rendered in settle_values.items():
        settle_text = replace_assignment(settle_text, key, rendered)

    colors_values = {
        "x_integer_ctrl(5)": str(int(value(cfg, "run", "colors_periods"))),
        "RSP_target_steps_per_cycle": str(colors_steps),
        "RSP_max_dt": fort_float(max_dt),
        "instrument": quote(str(instrument)),
        "mag_system": quote(mag_system),
        "vega_sed": quote(str(vega_sed)),
        "stellar_atm": quote(str(stellar_atm)),
        "distance": fort_float(model_distance_pc * PC_CM),
        "make_csv": ".true." if bool(value(cfg, "photometry", "write_sed_csv")) else ".false.",
        "sed_per_model": ".true."
        if bool(value(cfg, "photometry", "write_sed_per_model"))
        else ".false.",
    }
    for key, rendered in colors_values.items():
        colors_text = replace_assignment(colors_text, key, rendered)

    resolved = {
        "mesa_dir": str(mesa_dir),
        "source_id": str(value(cfg, "target", "source_id")),
        "name": str(value(cfg, "target", "name")),
        "stellar": {
            "mass_msun": mass,
            "luminosity_lsun": lum,
            "teff_k": teff,
            "x": x,
            "z": z,
        },
        "observations": {
            "period_days": period,
            "catalogue_distance_pc": float(value(cfg, "observations", "distance_pc")),
            "feh": float(value(cfg, "observations", "feh")),
        },
        "paths": {
            "gaia_instrument": str(instrument),
            "stellar_atmosphere": str(stellar_atm),
            "vega_sed": str(vega_sed),
        },
        "photometry": {
            "model_distance_pc": model_distance_pc,
            "magnitude_system": mag_system,
            "bands": ["Gbp", "G", "Grp"],
        },
        "run": {
            "target_mode": target_mode,
            "minimum_linear_growth": minimum_growth,
            "minimum_delta_r_rsun": min_delta_r,
            "minimum_delta_mag": min_delta_mag,
            "amplitude_relative_tolerance": amplitude_rel_tol,
            "colors_max_dt_seconds": max_dt,
            "settle_max_dt_seconds": settle_max_dt,
            "rsp_alfa": rsp_alfa,
            "rsp_alfam": rsp_alfam,
            "rsp_gammar": rsp_gammar,
            "rsp_alfap": rsp_alfap,
            "rsp_alfat": rsp_alfat,
            "max_rel_run_energy_error": max_energy_error,
        },
    }

    if check_only:
        print("target.toml, MESA_DIR, and required Colors paths are valid")
        print(f"MESA_DIR: {mesa_dir}")
        print(f"Gaia instrument: {instrument}")
        print(f"stellar atmosphere: {stellar_atm}")
        print(f"Vega SED: {vega_sed}")
        print(
            f"LINA target mode: {target_mode}; minimum growth: {minimum_growth:g}; "
            f"amplitude floors: DeltaR={min_delta_r:g}, DeltaMag={min_delta_mag:g}"
        )
        print(
            f"convection: RSP_alfa={rsp_alfa:g}, RSP_alfam={rsp_alfam:g}, "
            f"gammar={rsp_gammar:g}, alfap={rsp_alfap:g}, alfat={rsp_alfat:g}; "
            f"max rel energy error={max_energy_error:g}"
        )
        print(
            f"max_dt: settle={settle_max_dt:g} s, colors={max_dt:g} s "
            f"(from period={period:g} d)"
        )
        return resolved

    SETTLE.write_text(settle_text)
    COLORS.write_text(colors_text)
    (ROOT / "target_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")

    print(f"configured {ROOT}")
    print(f"MESA_DIR={mesa_dir}")
    print(
        f"source_id={resolved['source_id']}  M={mass:g}  L={lum:g}  "
        f"Teff={teff:g}  X={x:g}  Z={z:g}"
    )
    print(f"LINA target mode={target_mode}; minimum growth={minimum_growth:g}")
    print(f"Gaia bands: Gbp, G, Grp; model distance={model_distance_pc:g} pc")
    print(f"Gaia instrument: {instrument}")
    print(f"stellar atmosphere: {stellar_atm}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate target.toml, MESA_DIR, and required paths without rewriting inlists.",
    )
    args = parser.parse_args()
    with CONFIG.open("rb") as fh:
        cfg = tomllib.load(fh)
    render(cfg, check_only=args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
