#!/usr/bin/env python3
"""Automated grid + simplex + nonlinear Gaia RSP fitter.

This driver follows the successful FN Lyr pattern (coarse basin search followed
by Nelder-Mead refinement), but inserts a true LINA-only stage before expensive
nonlinear RSP evolution.

Stages
------
  doctor         validate workdir, $MESA_DIR, binary, configuration
  smoke          one LINA-only evaluation at the current [stellar] seed
  grid           coarse M-L-Teff LINA grid (parallel, resumable)
  simplex-lina   Nelder-Mead refinement of the best LINA grid basins
  nonlinear      full settle + Colors for the best LINA candidates
  fit            full nonlinear Nelder-Mead against Gaia G/Gbp/Grp data
  all            grid -> simplex-lina -> nonlinear; optionally full fit
  summary        print the best cached results

Every evaluation runs in an isolated deterministic cache directory under
fit/runs. The compiled executable from ROOT/build/bin/star is reused; MESA is
not rebuilt for individual evaluations.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import threading
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "target.toml"
FIT_DIR = ROOT / "fit"
RUNS_DIR = FIT_DIR / "runs"
LINA_RUNS = RUNS_DIR / "lina"
NONLINEAR_RUNS = RUNS_DIR / "nonlinear"
GRID_CSV = FIT_DIR / "grid_results.csv"
GRID_RANKED_CSV = FIT_DIR / "grid_ranked.csv"
SIMPLEX_CSV = FIT_DIR / "simplex_lina_results.csv"
SIMPLEX_JSON = FIT_DIR / "simplex_lina_summary.json"
NONLINEAR_CSV = FIT_DIR / "nonlinear_results.csv"
FULL_SIMPLEX_CSV = FIT_DIR / "simplex_full_results.csv"
FULL_SIMPLEX_JSON = FIT_DIR / "simplex_full_summary.json"
LINA_TEMPLATE = ROOT / "inlist_rsp_LINA"
SETTLE_TEMPLATE = ROOT / "inlist_rsp_RR_Lyrae"
COLORS_TEMPLATE = ROOT / "inlist_rsp_RR_Lyrae_colors"
BINARY = ROOT / "build" / "bin" / "star"

SUPPORT_FILES = [
    "history_columns_settle.list",
    "history_columns_colors.list",
    "profile_columns.list",
]
BANDS = ("G", "Gbp", "Grp")
_CONFIGURE_LOCK = threading.Lock()
_ROOT_CONFIGURED = False


@dataclass(frozen=True)
class Params:
    mass: float
    luminosity: float
    teff: float
    x: float
    z: float

    def rounded(self) -> "Params":
        return Params(
            mass=round(float(self.mass), 6),
            luminosity=round(float(self.luminosity), 5),
            teff=round(float(self.teff), 4),
            x=round(float(self.x), 8),
            z=round(float(self.z), 8),
        )


# -----------------------------------------------------------------------------
# Configuration and generic helpers
# -----------------------------------------------------------------------------

def load_cfg() -> dict[str, Any]:
    with TARGET.open("rb") as fh:
        return tomllib.load(fh)


def cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return cfg.get(section, {}).get(key, default)


def require_mesa_dir() -> Path:
    raw = os.environ.get("MESA_DIR", "").strip()
    if not raw:
        raise RuntimeError("MESA_DIR is not set")
    p = Path(raw).expanduser().resolve()
    required = p / "star" / "work" / "Makefile"
    if not required.is_file():
        raise RuntimeError(f"invalid MESA_DIR={p}; missing {required}")
    return p


def require_binary() -> Path:
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        raise RuntimeError(
            f"compiled MESA work binary is missing: {BINARY}\n"
            "Run: make clean && make"
        )
    return BINARY.resolve()


def require_period(cfg: dict[str, Any]) -> float:
    p = float(cfg_get(cfg, "observations", "period_days", 0.0))
    if not math.isfinite(p) or p <= 0:
        raise RuntimeError(
            "[observations] period_days must be set to the observed pulsation "
            "period before grid/simplex fitting"
        )
    return p


def current_seed(cfg: dict[str, Any]) -> Params:
    s = cfg["stellar"]
    return Params(
        mass=float(s["mass_msun"]),
        luminosity=float(s["luminosity_lsun"]),
        teff=float(s["teff_k"]),
        x=float(s["x"]),
        z=float(s["z"]),
    )


def target_mode(cfg: dict[str, Any]) -> int:
    return int(cfg_get(cfg, "run", "target_mode", 0))


def minimum_growth(cfg: dict[str, Any]) -> float:
    return float(cfg_get(cfg, "run", "minimum_linear_growth", 0.0))


def fort_float(x: float) -> str:
    s = f"{float(x):.12g}"
    if "e" in s.lower():
        m, e = re.split(r"[eE]", s)
        return f"{m}d{e}"
    return s + "d0"


def replace_assignment(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^(\s*){re.escape(key)}\s*=.*$", re.MULTILINE)
    out, n = pat.subn(lambda m: f"{m.group(1)}{key} = {value}", text)
    if n != 1:
        raise RuntimeError(f"expected one {key} assignment, found {n}")
    return out


def patch_rsp_inlist(text: str, p: Params, mode: int | None = None) -> str:
    values = {
        "Zbase": fort_float(p.z),
        "RSP_mass": fort_float(p.mass),
        "RSP_Teff": fort_float(p.teff),
        "RSP_L": fort_float(p.luminosity),
        "RSP_X": fort_float(p.x),
        "RSP_Z": fort_float(p.z),
    }
    if mode is not None:
        values["x_integer_ctrl(1)"] = str(mode)
    for key, value in values.items():
        text = replace_assignment(text, key, value)
    return text


def patch_lina_inlist(text: str, p: Params, cfg: dict[str, Any]) -> str:
    text = patch_rsp_inlist(text, p, mode=3)
    text = replace_assignment(text, "x_integer_ctrl(6)", str(target_mode(cfg)))
    text = replace_assignment(text, "x_ctrl(3)", fort_float(minimum_growth(cfg)))
    text = replace_assignment(
        text,
        "RSP_kick_vsurf_km_per_sec",
        fort_float(float(cfg_get(cfg, "run", "kick_km_s", 20.0))),
    )
    text = replace_assignment(
        text,
        "RSP_target_steps_per_cycle",
        str(int(cfg_get(cfg, "run", "settle_steps_per_cycle", 150))),
    )
    text = replace_assignment(
        text,
        "RSP_max_dt",
        fort_float(float(cfg_get(cfg, "run", "fallback_max_dt_seconds", 600.0))),
    )
    return text


def deterministic_id(kind: str, p: Params, cfg: dict[str, Any]) -> str:
    payload = {
        "kind": kind,
        "params": asdict(p.rounded()),
        "target_mode": target_mode(cfg),
        "minimum_growth": minimum_growth(cfg),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def prepare_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in SUPPORT_FILES:
        src = ROOT / name
        if not src.is_file():
            raise RuntimeError(f"required workdir file missing: {src}")
        safe_link(src, run_dir / name)


def run_binary(
    run_dir: Path,
    inlist_text: str,
    log_name: str,
    cfg: dict[str, Any],
    timeout: float,
) -> tuple[int, float]:
    require_binary()
    (run_dir / "inlist").write_text(inlist_text)
    env = os.environ.copy()
    env["MESA_DIR"] = str(require_mesa_dir())
    env["OMP_NUM_THREADS"] = str(int(cfg_get(cfg, "search", "threads_per_run", 1)))
    env.setdefault("OMP_PROC_BIND", "false")
    start = time.monotonic()
    with (run_dir / log_name).open("w") as log:
        try:
            proc = subprocess.run(
                [str(BINARY.resolve())],
                cwd=run_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout} seconds\n")
            rc = 124
    return rc, time.monotonic() - start


def parse_key_value(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        if "=" not in raw:
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            if re.fullmatch(r"[-+]?\d+", val):
                out[key] = int(val)
            else:
                out[key] = float(val.replace("D", "E").replace("d", "e"))
        except ValueError:
            out[key] = val
    return out


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    p = result.get("params", {})
    row = {
        "run_id": result.get("run_id", ""),
        "status": result.get("status", ""),
        "mass": p.get("mass", np.nan),
        "luminosity": p.get("luminosity", np.nan),
        "teff": p.get("teff", np.nan),
        "x": p.get("x", np.nan),
        "z": p.get("z", np.nan),
        "target_mode": result.get("target_mode", np.nan),
        "period_days": result.get("period_days", np.nan),
        "growth": result.get("growth", np.nan),
        "period_frac_error": result.get("period_frac_error", np.nan),
        "accepted_lina": result.get("accepted_lina", False),
        "objective": result.get("objective", np.nan),
        "runtime_s": result.get("runtime_s", np.nan),
        "cached": result.get("cached", False),
    }
    for key in (
        "settled",
        "colors_complete",
        "phase_shift",
        "rms_G",
        "rms_Gbp",
        "rms_Grp",
        "photometry_objective",
        "full_objective",
    ):
        if key in result:
            row[key] = result[key]
    return row


def write_csv(path: Path, results: Iterable[dict[str, Any]]) -> None:
    rows = [flatten_result(r) for r in results]
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# -----------------------------------------------------------------------------
# LINA evaluation, grid, and simplex
# -----------------------------------------------------------------------------

def lina_objective(period_days: float, growth: float, cfg: dict[str, Any]) -> tuple[float, float, bool]:
    obs = require_period(cfg)
    frac = abs(period_days - obs) / obs if math.isfinite(period_days) else math.inf
    min_g = minimum_growth(cfg)
    margin = float(cfg_get(cfg, "search", "growth_margin", 0.0))
    growth_scale = float(cfg_get(cfg, "search", "growth_scale", 0.01))
    growth_weight = float(cfg_get(cfg, "search", "growth_penalty_weight", 1.0))
    damped_base = float(cfg_get(cfg, "search", "damped_base_penalty", 1.0))
    accepted = math.isfinite(growth) and growth > min_g
    deficit = max(0.0, min_g + margin - growth) if math.isfinite(growth) else 1.0
    penalty = growth_weight * (deficit / max(growth_scale, 1e-12)) ** 2
    if not accepted:
        penalty += damped_base
    return frac * frac + penalty, frac, accepted


def evaluate_lina(p: Params, cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    p = p.rounded()
    run_id = deterministic_id("lina", p, cfg)
    run_dir = LINA_RUNS / run_id
    result_file = run_dir / "result.json"
    if result_file.is_file() and not force:
        result = json.loads(result_file.read_text())
        period = float(result.get("period_days", math.nan))
        growth = float(result.get("growth", math.nan))
        objective, frac, accepted = lina_objective(period, growth, cfg)
        result["period_frac_error"] = frac
        result["accepted_lina"] = accepted
        result["objective"] = objective
        result["cached"] = True
        return result

    prepare_run_dir(run_dir)
    for name in ("LINA_RESULT", "LINA_REJECTED", "SETTLING_REJECTED", "SETTLED_ACCEPTED"):
        (run_dir / name).unlink(missing_ok=True)
    shutil.rmtree(run_dir / "LOGS", ignore_errors=True)

    text = patch_lina_inlist(LINA_TEMPLATE.read_text(), p, cfg)
    timeout = float(cfg_get(cfg, "search", "lina_timeout_seconds", 180.0))
    rc, runtime = run_binary(run_dir, text, "run.log", cfg, timeout)

    result: dict[str, Any] = {
        "kind": "lina",
        "run_id": run_id,
        "params": asdict(p),
        "target_mode": target_mode(cfg),
        "runtime_s": runtime,
        "returncode": rc,
        "run_dir": str(run_dir),
        "cached": False,
    }

    lina_path = run_dir / "LINA_RESULT"
    if rc != 0 or not lina_path.is_file():
        result.update(
            status="crash" if rc != 124 else "timeout",
            period_days=math.nan,
            growth=math.nan,
            period_frac_error=math.inf,
            accepted_lina=False,
            objective=float(cfg_get(cfg, "search", "crash_penalty", 1e6)),
        )
    else:
        kv = parse_key_value(lina_path)
        period = float(kv.get("selected_period_days", math.nan))
        growth = float(kv.get("selected_growth", math.nan))
        objective, frac, accepted = lina_objective(period, growth, cfg)
        result.update(
            status=str(kv.get("status", "unknown")),
            reason=str(kv.get("reason", "")),
            period_days=period,
            growth=growth,
            period_frac_error=frac,
            accepted_lina=accepted,
            objective=objective,
            lina=kv,
        )

    write_json_atomic(result_file, result)
    return result


def search_bounds(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    s = cfg.get("search", {})
    return {
        "mass": (float(s.get("mass_min", 6.0)), float(s.get("mass_max", 10.0))),
        "luminosity": (
            float(s.get("luminosity_min", 3000.0)),
            float(s.get("luminosity_max", 8000.0)),
        ),
        "teff": (float(s.get("teff_min", 5000.0)), float(s.get("teff_max", 6500.0))),
    }


def grid_points(cfg: dict[str, Any]) -> list[Params]:
    bounds = search_bounds(cfg)
    s = cfg.get("search", {})
    masses = np.linspace(*bounds["mass"], int(s.get("mass_points", 5)))
    lums = np.linspace(*bounds["luminosity"], int(s.get("luminosity_points", 9)))
    teffs = np.linspace(*bounds["teff"], int(s.get("teff_points", 11)))
    seed = current_seed(cfg)
    return [
        Params(float(m), float(l), float(t), seed.x, seed.z)
        for m, l, t in product(masses, lums, teffs)
    ]


def rank_lina(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda r: (
            not bool(r.get("accepted_lina", False)),
            float(r.get("objective", math.inf)),
            float(r.get("period_frac_error", math.inf)),
        ),
    )


def run_grid(cfg: dict[str, Any], workers: int | None, limit: int | None, force: bool) -> list[dict[str, Any]]:
    require_period(cfg)
    require_binary()
    pts = grid_points(cfg)
    if limit is not None:
        pts = pts[: max(0, limit)]
    n_workers = workers or int(cfg_get(cfg, "search", "workers", 1))
    n_workers = max(1, n_workers)
    print(f"LINA grid: {len(pts)} models; workers={n_workers}; OMP threads/run={cfg_get(cfg, 'search', 'threads_per_run', 1)}")

    results: list[dict[str, Any]] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_map = {ex.submit(evaluate_lina, p, cfg, force): p for p in pts}
        for fut in cf.as_completed(future_map):
            r = fut.result()
            results.append(r)
            done += 1
            if done <= 10 or done % 25 == 0 or done == len(pts):
                print(
                    f"[{done:4d}/{len(pts):4d}] M={r['params']['mass']:.3f} "
                    f"L={r['params']['luminosity']:.1f} T={r['params']['teff']:.1f} "
                    f"P={r.get('period_days', math.nan):.5g} "
                    f"g={r.get('growth', math.nan):+.4e} "
                    f"{'PASS' if r.get('accepted_lina') else 'reject'}"
                )

    ranked = rank_lina(results)
    write_csv(GRID_CSV, results)
    write_csv(GRID_RANKED_CSV, ranked)
    n_pass = sum(bool(r.get("accepted_lina")) for r in results)
    print(f"grid complete: {n_pass}/{len(results)} linearly excited")
    if ranked:
        b = ranked[0]
        print(
            "best grid point: "
            f"M={b['params']['mass']:.5g} L={b['params']['luminosity']:.5g} "
            f"Teff={b['params']['teff']:.5g} P={b.get('period_days', math.nan):.7g} "
            f"growth={b.get('growth', math.nan):+.5e}"
        )
    return ranked


def normalized_to_params(xn: np.ndarray, cfg: dict[str, Any]) -> Params:
    b = search_bounds(cfg)
    seed = current_seed(cfg)
    vals = []
    for val, key in zip(xn, ("mass", "luminosity", "teff")):
        lo, hi = b[key]
        vals.append(lo + float(val) * (hi - lo))
    return Params(vals[0], vals[1], vals[2], seed.x, seed.z)


def params_to_normalized(p: Params, cfg: dict[str, Any]) -> np.ndarray:
    b = search_bounds(cfg)
    return np.array(
        [
            (p.mass - b["mass"][0]) / (b["mass"][1] - b["mass"][0]),
            (p.luminosity - b["luminosity"][0]) / (b["luminosity"][1] - b["luminosity"][0]),
            (p.teff - b["teff"][0]) / (b["teff"][1] - b["teff"][0]),
        ],
        dtype=float,
    )


def outside_unit_penalty(x: np.ndarray) -> float:
    below = np.clip(-x, 0.0, None)
    above = np.clip(x - 1.0, 0.0, None)
    d2 = float(np.sum(below * below + above * above))
    return 1e3 + 1e4 * d2 if d2 > 0 else 0.0


def load_ranked_grid() -> list[dict[str, Any]]:
    if not GRID_RANKED_CSV.is_file():
        raise RuntimeError("grid results missing; run: python fit_gaia_rsp.py grid")
    df = pd.read_csv(GRID_RANKED_CSV)
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "params": {
                    "mass": float(row.mass),
                    "luminosity": float(row.luminosity),
                    "teff": float(row.teff),
                    "x": float(row.x),
                    "z": float(row.z),
                },
                "accepted_lina": bool(row.accepted_lina),
                "objective": float(row.objective),
                "period_days": float(row.period_days),
                "growth": float(row.growth),
                "period_frac_error": float(row.period_frac_error),
            }
        )
    return out


def run_simplex_lina(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    require_period(cfg)
    grid = load_ranked_grid()
    if not grid:
        raise RuntimeError("grid results are empty")
    starts = int(cfg_get(cfg, "search", "simplex_starts", 3))
    candidates = [r for r in grid if r.get("accepted_lina")]
    if not candidates:
        print("WARNING: coarse grid found no excited points; simplex will start from best damped points")
        candidates = grid
    candidates = candidates[: max(1, starts)]

    maxiter = int(cfg_get(cfg, "search", "simplex_maxiter", 80))
    step = float(cfg_get(cfg, "search", "simplex_step_fraction", 0.06))
    xatol = float(cfg_get(cfg, "search", "simplex_xatol", 2e-3))
    fatol = float(cfg_get(cfg, "search", "simplex_fatol", 1e-6))
    seen: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    def objective(xn: np.ndarray) -> float:
        penalty = outside_unit_penalty(xn)
        if penalty:
            return penalty
        p = normalized_to_params(xn, cfg)
        r = evaluate_lina(p, cfg, force=force)
        seen[r["run_id"]] = r
        return float(r["objective"])

    for idx, candidate in enumerate(candidates, 1):
        p0 = Params(**candidate["params"])
        x0 = np.clip(params_to_normalized(p0, cfg), 0.0, 1.0)
        simplex = np.vstack([x0, x0.copy(), x0.copy(), x0.copy()])
        for j in range(3):
            simplex[j + 1, j] = min(1.0, x0[j] + step)
            if simplex[j + 1, j] == x0[j]:
                simplex[j + 1, j] = max(0.0, x0[j] - step)
        print(f"LINA simplex {idx}/{len(candidates)} from M={p0.mass:g} L={p0.luminosity:g} T={p0.teff:g}")
        opt = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "initial_simplex": simplex,
                "maxiter": maxiter,
                "xatol": xatol,
                "fatol": fatol,
                "adaptive": True,
            },
        )
        pbest = normalized_to_params(np.clip(opt.x, 0.0, 1.0), cfg)
        rbest = evaluate_lina(pbest, cfg, force=False)
        seen[rbest["run_id"]] = rbest
        summaries.append(
            {
                "start": idx,
                "success": bool(opt.success),
                "message": str(opt.message),
                "nit": int(opt.nit),
                "nfev": int(opt.nfev),
                "best": rbest,
            }
        )

    all_results = rank_lina(list(seen.values()))
    write_csv(SIMPLEX_CSV, all_results)
    overall = all_results[0] if all_results else None
    summary = {"starts": summaries, "best": overall}
    write_json_atomic(SIMPLEX_JSON, summary)
    if overall:
        print(
            "best LINA simplex point: "
            f"M={overall['params']['mass']:.6g} L={overall['params']['luminosity']:.6g} "
            f"Teff={overall['params']['teff']:.6g} P={overall['period_days']:.8g} "
            f"growth={overall['growth']:+.6e}"
        )
    return summary


# -----------------------------------------------------------------------------
# Nonlinear RSP + Colors evaluation
# -----------------------------------------------------------------------------

def configure_root() -> None:
    global _ROOT_CONFIGURED
    with _CONFIGURE_LOCK:
        if _ROOT_CONFIGURED:
            return
        proc = subprocess.run(
            [sys.executable, str(ROOT / "configure_target.py")],
            cwd=ROOT,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError("configure_target.py failed")
        _ROOT_CONFIGURED = True


def clean_stage_files(run_dir: Path) -> None:
    for name in (
        "settled.mod",
        "colors_final.mod",
        "LINA_RESULT",
        "LINA_REJECTED",
        "SETTLING_REJECTED",
        "SETTLED_ACCEPTED",
    ):
        (run_dir / name).unlink(missing_ok=True)
    for name in ("LOGS", "LOGS_settle", "LOGS_colors", "SED"):
        shutil.rmtree(run_dir / name, ignore_errors=True)
    (run_dir / "SED").mkdir(exist_ok=True)


def read_history(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if "model_number" in line.split())
    return pd.read_csv(path, sep=r"\s+", skiprows=header_idx)


def final_model_period(run_dir: Path) -> float:
    paths = [run_dir / "LOGS_colors/history.data", run_dir / "LOGS_settle/history.data"]
    for path in paths:
        if path.is_file():
            h = read_history(path)
            if "rsp_period_in_days" in h.columns:
                vals = pd.to_numeric(h["rsp_period_in_days"], errors="coerce")
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals):
                    return float(np.nanmedian(vals.iloc[-min(200, len(vals)) :]))
    return math.nan


def observation_path(cfg: dict[str, Any]) -> Path:
    raw = str(cfg_get(cfg, "fit", "observations_file", "observations/gaia_epoch_photometry.csv"))
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    return p if p.is_absolute() else ROOT / p


def normalize_band(raw: Any) -> str:
    x = str(raw).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "g": "G",
        "gaia g": "G",
        "gaiag": "G",
        "bp": "Gbp",
        "gbp": "Gbp",
        "gaiabp": "Gbp",
        "rp": "Grp",
        "grp": "Grp",
        "gaiarp": "Grp",
    }
    return aliases.get(x, str(raw).strip())


def load_observations(cfg: dict[str, Any], required: bool = False) -> pd.DataFrame | None:
    path = observation_path(cfg)
    if not path.is_file():
        if required:
            raise RuntimeError(
                f"Gaia epoch photometry file is missing: {path}\n"
                "Use long format columns: time,band,mag,mag_err (phase may replace time)."
            )
        return None
    obs = pd.read_csv(path)
    lower = {c.lower(): c for c in obs.columns}
    if "band" not in lower or "mag" not in lower:
        raise RuntimeError(f"{path} must contain band and mag columns")
    band_col = lower["band"]
    mag_col = lower["mag"]
    err_col = lower.get("mag_err") or lower.get("error") or lower.get("err")
    phase_col = lower.get("phase")
    time_col = lower.get("time") or lower.get("jd") or lower.get("bjd") or lower.get("mjd")
    if phase_col is None and time_col is None:
        raise RuntimeError(f"{path} must contain phase or time/JD/BJD/MJD")

    out = pd.DataFrame()
    out["band"] = obs[band_col].map(normalize_band)
    out["mag"] = pd.to_numeric(obs[mag_col], errors="coerce")
    if err_col:
        out["mag_err"] = pd.to_numeric(obs[err_col], errors="coerce")
    else:
        out["mag_err"] = np.nan
    period = require_period(cfg)
    if phase_col:
        out["phase"] = pd.to_numeric(obs[phase_col], errors="coerce") % 1.0
    else:
        times = pd.to_numeric(obs[time_col], errors="coerce")
        t0 = float(np.nanmin(times))
        out["phase"] = ((times - t0) / period) % 1.0
    out = out[np.isfinite(out["phase"]) & np.isfinite(out["mag"]) & out["band"].isin(BANDS)].copy()
    if out.empty:
        raise RuntimeError(f"{path} contains no usable G/Gbp/Grp measurements")
    return out


def periodic_interp(phase: np.ndarray, mag: np.ndarray, q: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, float) % 1.0
    mag = np.asarray(mag, float)
    q = np.asarray(q, float) % 1.0
    order = np.argsort(phase)
    phase, mag = phase[order], mag[order]
    x = np.concatenate([phase - 1.0, phase, phase + 1.0])
    y = np.concatenate([mag, mag, mag])
    return np.interp(q, x, y)


def model_curves(run_dir: Path, cfg: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = run_dir / "LOGS_colors/history.data"
    if not path.is_file():
        raise RuntimeError("Colors history is missing")
    h = read_history(path)
    for band in BANDS:
        if band not in h.columns:
            raise RuntimeError(f"Colors history does not contain {band}")
    if "rsp_phase" not in h.columns or "rsp_num_periods" not in h.columns:
        raise RuntimeError("Colors history lacks RSP phase/period columns")
    ncycles = int(cfg_get(cfg, "fit", "model_cycles", 5))
    nmax = int(np.nanmax(pd.to_numeric(h["rsp_num_periods"], errors="coerce")))
    use = h[pd.to_numeric(h["rsp_num_periods"], errors="coerce") >= nmax - ncycles + 1].copy()
    phase = pd.to_numeric(use["rsp_phase"], errors="coerce").to_numpy(float) % 1.0
    nbins = int(cfg_get(cfg, "fit", "model_phase_bins", 300))
    ibin = np.floor(phase * nbins).astype(int)
    ibin = np.clip(ibin, 0, nbins - 1)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for band in BANDS:
        vals = pd.to_numeric(use[band], errors="coerce").to_numpy(float)
        tmp = pd.DataFrame({"bin": ibin, "mag": vals})
        med = tmp[np.isfinite(tmp.mag)].groupby("bin").mag.median()
        if len(med) < 10:
            raise RuntimeError(f"too few usable {band} model phase bins")
        ph = (med.index.to_numpy(float) + 0.5) / nbins
        curves[band] = (ph, med.to_numpy(float))
    return curves


def score_photometry(run_dir: Path, cfg: dict[str, Any], obs: pd.DataFrame) -> dict[str, float]:
    curves = model_curves(run_dir, cfg)
    model_distance = float(cfg_get(cfg, "photometry", "model_distance_pc", 10.0))
    fit_distance = float(cfg_get(cfg, "fit", "distance_pc", 0.0))
    if fit_distance <= 0:
        fit_distance = float(cfg_get(cfg, "observations", "distance_pc", 0.0))
    if fit_distance <= 0 or model_distance <= 0:
        raise RuntimeError("a positive distance is required for direct apparent-magnitude fitting")
    distance_term = 5.0 * math.log10(fit_distance / model_distance)
    ext = {
        "G": float(cfg_get(cfg, "fit", "extinction_G", 0.0)),
        "Gbp": float(cfg_get(cfg, "fit", "extinction_Gbp", 0.0)),
        "Grp": float(cfg_get(cfg, "fit", "extinction_Grp", 0.0)),
    }
    nshift = int(cfg_get(cfg, "fit", "phase_grid_points", 720))
    shifts = np.linspace(0.0, 1.0, nshift, endpoint=False)
    best: tuple[float, float, dict[str, float]] | None = None
    for shift in shifts:
        band_rms: dict[str, float] = {}
        for band in BANDS:
            ob = obs[obs.band == band]
            if ob.empty:
                continue
            ph, mag = curves[band]
            pred = periodic_interp(ph, mag, (ob.phase.to_numpy(float) + shift) % 1.0)
            pred = pred + distance_term + ext[band]
            resid = ob.mag.to_numpy(float) - pred
            err = ob.mag_err.to_numpy(float)
            good_err = np.isfinite(err) & (err > 0)
            if np.any(good_err):
                w = np.where(good_err, 1.0 / err**2, 0.0)
                rms = math.sqrt(float(np.sum(w * resid**2) / np.sum(w)))
            else:
                rms = math.sqrt(float(np.mean(resid**2)))
            band_rms[band] = rms
        if not band_rms:
            continue
        score = float(np.mean([v * v for v in band_rms.values()]))
        if best is None or score < best[0]:
            best = (score, float(shift), band_rms)
    if best is None:
        raise RuntimeError("could not score Gaia photometry")
    sigma_mag = float(cfg_get(cfg, "fit", "photometric_sigma_mag", 0.03))
    _, shift, band_rms = best
    photo_obj = float(np.mean([(r / sigma_mag) ** 2 for r in band_rms.values()]))
    out = {
        "phase_shift": shift,
        "photometry_objective": photo_obj,
    }
    for band in BANDS:
        out[f"rms_{band}"] = float(band_rms.get(band, math.nan))
    return out


def rescore_nonlinear_result(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(result.get("run_dir", ""))
    if not result.get("colors_complete") or not (run_dir / "LOGS_colors/history.data").is_file():
        return result
    lina_path = run_dir / "LINA_RESULT"
    if lina_path.is_file():
        lina_kv = parse_key_value(lina_path)
        result["growth"] = float(lina_kv.get("selected_growth", result.get("growth", math.nan)))
        result["lina_period_days"] = float(lina_kv.get("selected_period_days", result.get("lina_period_days", math.nan)))
    period = final_model_period(run_dir)
    obs_period = require_period(cfg)
    frac = abs(period - obs_period) / obs_period if math.isfinite(period) else math.inf
    result["period_days"] = period
    result["period_frac_error"] = frac
    obs = load_observations(cfg, required=False)
    if obs is None:
        result["objective"] = frac * frac
        for key in ("phase_shift", "photometry_objective", "full_objective", "rms_G", "rms_Gbp", "rms_Grp", "photometry_error"):
            result.pop(key, None)
        if result.get("status") == "photometry_error":
            result["status"] = "complete"
        return result
    try:
        photo = score_photometry(run_dir, cfg, obs)
    except Exception as exc:
        result["status"] = "photometry_error"
        result["photometry_error"] = str(exc)
        result["objective"] = float(cfg_get(cfg, "search", "crash_penalty", 1e6))
    else:
        result["status"] = "complete"
        result.pop("photometry_error", None)
        result.update(photo)
        psig = float(cfg_get(cfg, "fit", "period_sigma_fraction", 0.005))
        period_obj = (frac / psig) ** 2
        full = float(photo["photometry_objective"]) + period_obj
        result["full_objective"] = full
        result["objective"] = full
    return result


def evaluate_nonlinear(p: Params, cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    p = p.rounded()
    run_id = deterministic_id("nonlinear", p, cfg)
    run_dir = NONLINEAR_RUNS / run_id
    result_file = run_dir / "result.json"
    if result_file.is_file() and not force:
        result = json.loads(result_file.read_text())
        result = rescore_nonlinear_result(result, cfg)
        result["cached"] = True
        write_json_atomic(result_file, result)
        return result

    configure_root()
    prepare_run_dir(run_dir)
    clean_stage_files(run_dir)
    stage1 = patch_rsp_inlist(SETTLE_TEMPLATE.read_text(), p, mode=1)
    stage2 = patch_rsp_inlist(COLORS_TEMPLATE.read_text(), p, mode=2)
    timeout = float(cfg_get(cfg, "search", "nonlinear_timeout_seconds", 14400.0))
    total_start = time.monotonic()

    rc1, rt1 = run_binary(run_dir, stage1, "stage1.log", cfg, timeout)
    if (run_dir / "LOGS").is_dir():
        (run_dir / "LOGS").rename(run_dir / "LOGS_settle")

    result: dict[str, Any] = {
        "kind": "nonlinear",
        "run_id": run_id,
        "params": asdict(p),
        "target_mode": target_mode(cfg),
        "returncode_stage1": rc1,
        "runtime_stage1_s": rt1,
        "run_dir": str(run_dir),
        "cached": False,
        "settled": False,
        "colors_complete": False,
    }

    if rc1 != 0 or (run_dir / "LINA_REJECTED").is_file() or (run_dir / "SETTLING_REJECTED").is_file():
        result["status"] = "stage1_rejected" if rc1 == 0 else "stage1_crash"
        result["runtime_s"] = time.monotonic() - total_start
        result["objective"] = float(cfg_get(cfg, "search", "crash_penalty", 1e6))
        (run_dir / "settled.mod").unlink(missing_ok=True)
        write_json_atomic(result_file, result)
        return result

    if not (run_dir / "SETTLED_ACCEPTED").is_file() or not (run_dir / "settled.mod").is_file():
        result["status"] = "stage1_unaccepted"
        result["runtime_s"] = time.monotonic() - total_start
        result["objective"] = float(cfg_get(cfg, "search", "crash_penalty", 1e6))
        write_json_atomic(result_file, result)
        return result

    result["settled"] = True
    lina_result_path = run_dir / "LINA_RESULT"
    if lina_result_path.is_file():
        lina_kv = parse_key_value(lina_result_path)
        result["growth"] = float(lina_kv.get("selected_growth", math.nan))
        result["lina_period_days"] = float(lina_kv.get("selected_period_days", math.nan))
    shutil.rmtree(run_dir / "LOGS", ignore_errors=True)
    rc2, rt2 = run_binary(run_dir, stage2, "stage2.log", cfg, timeout)
    if (run_dir / "LOGS").is_dir():
        (run_dir / "LOGS").rename(run_dir / "LOGS_colors")
    result["returncode_stage2"] = rc2
    result["runtime_stage2_s"] = rt2

    if rc2 != 0 or not (run_dir / "colors_final.mod").is_file() or not (run_dir / "LOGS_colors/history.data").is_file():
        result["status"] = "stage2_crash"
        result["runtime_s"] = time.monotonic() - total_start
        result["objective"] = float(cfg_get(cfg, "search", "crash_penalty", 1e6))
        write_json_atomic(result_file, result)
        return result

    result["colors_complete"] = True
    period = final_model_period(run_dir)
    obs_period = require_period(cfg)
    frac = abs(period - obs_period) / obs_period if math.isfinite(period) else math.inf
    result["period_days"] = period
    result["period_frac_error"] = frac
    result["status"] = "complete"

    obs = load_observations(cfg, required=False)
    if obs is not None:
        try:
            photo = score_photometry(run_dir, cfg, obs)
        except Exception as exc:
            result["status"] = "photometry_error"
            result["photometry_error"] = str(exc)
            result["objective"] = float(cfg_get(cfg, "search", "crash_penalty", 1e6))
        else:
            result.update(photo)
            psig = float(cfg_get(cfg, "fit", "period_sigma_fraction", 0.005))
            period_obj = (frac / psig) ** 2
            full = float(photo["photometry_objective"]) + period_obj
            result["full_objective"] = full
            result["objective"] = full
    else:
        result["objective"] = frac * frac
    result["runtime_s"] = time.monotonic() - total_start
    write_json_atomic(result_file, result)
    return result


def params_from_row(row: pd.Series) -> Params:
    return Params(float(row.mass), float(row.luminosity), float(row.teff), float(row.x), float(row.z))


def best_lina_candidates(cfg: dict[str, Any], n: int) -> list[Params]:
    frames = []
    if SIMPLEX_CSV.is_file():
        frames.append(pd.read_csv(SIMPLEX_CSV))
    if GRID_RANKED_CSV.is_file():
        frames.append(pd.read_csv(GRID_RANKED_CSV))
    if not frames:
        raise RuntimeError("no LINA search results; run grid and simplex-lina first")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.accepted_lina.astype(bool)].copy()
    if df.empty:
        raise RuntimeError("no linearly excited candidates were found")
    df = df.sort_values(["objective", "period_frac_error"])
    seen: set[tuple[float, float, float]] = set()
    out: list[Params] = []
    for _, row in df.iterrows():
        p = params_from_row(row).rounded()
        key = (p.mass, p.luminosity, p.teff)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= n:
            break
    return out


def run_nonlinear_candidates(cfg: dict[str, Any], n: int | None, force: bool) -> list[dict[str, Any]]:
    keep = n or int(cfg_get(cfg, "search", "nonlinear_keep", 5))
    candidates = best_lina_candidates(cfg, keep)
    workers = max(1, int(cfg_get(cfg, "search", "nonlinear_workers", 1)))
    print(f"nonlinear stage: {len(candidates)} candidates; workers={workers}")
    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(evaluate_nonlinear, p, cfg, force): p for p in candidates}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(
                f"[{i}/{len(candidates)}] {r['status']} M={r['params']['mass']:.5g} "
                f"L={r['params']['luminosity']:.5g} T={r['params']['teff']:.5g} "
                f"P={r.get('period_days', math.nan):.7g} obj={r.get('objective', math.inf):.6g}"
            )
    results.sort(key=lambda r: float(r.get("objective", math.inf)))
    write_csv(NONLINEAR_CSV, results)
    return results


# -----------------------------------------------------------------------------
# Full nonlinear simplex against Gaia photometry
# -----------------------------------------------------------------------------

def full_seed(cfg: dict[str, Any]) -> Params:
    if NONLINEAR_CSV.is_file():
        df = pd.read_csv(NONLINEAR_CSV).sort_values("objective")
        good = df[df.status == "complete"]
        if not good.empty:
            return params_from_row(good.iloc[0])
    cands = best_lina_candidates(cfg, 1)
    return cands[0]


def run_full_simplex(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    load_observations(cfg, required=True)
    require_period(cfg)
    p0 = full_seed(cfg)
    x0 = np.clip(params_to_normalized(p0, cfg), 0.0, 1.0)
    step = float(cfg_get(cfg, "fit", "simplex_step_fraction", 0.03))
    maxiter = int(cfg_get(cfg, "fit", "simplex_maxiter", 30))
    simplex = np.vstack([x0, x0.copy(), x0.copy(), x0.copy()])
    for j in range(3):
        simplex[j + 1, j] = min(1.0, x0[j] + step)
        if simplex[j + 1, j] == x0[j]:
            simplex[j + 1, j] = max(0.0, x0[j] - step)
    seen: dict[str, dict[str, Any]] = {}

    def objective(xn: np.ndarray) -> float:
        penalty = outside_unit_penalty(xn)
        if penalty:
            return penalty
        p = normalized_to_params(xn, cfg)
        lina = evaluate_lina(p, cfg, force=False)
        if not lina.get("accepted_lina", False):
            return 100.0 + float(lina.get("objective", 1e3))
        r = evaluate_nonlinear(p, cfg, force=False)
        seen[r["run_id"]] = r
        return float(r.get("objective", 1e6))

    print(f"full nonlinear simplex from M={p0.mass:g} L={p0.luminosity:g} T={p0.teff:g}")
    opt = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "maxiter": maxiter,
            "xatol": float(cfg_get(cfg, "fit", "simplex_xatol", 1e-3)),
            "fatol": float(cfg_get(cfg, "fit", "simplex_fatol", 1e-3)),
            "adaptive": True,
        },
    )
    pbest = normalized_to_params(np.clip(opt.x, 0.0, 1.0), cfg)
    best = evaluate_nonlinear(pbest, cfg, force=False)
    seen[best["run_id"]] = best
    results = sorted(seen.values(), key=lambda r: float(r.get("objective", math.inf)))
    write_csv(FULL_SIMPLEX_CSV, results)
    summary = {
        "success": bool(opt.success),
        "message": str(opt.message),
        "nit": int(opt.nit),
        "nfev": int(opt.nfev),
        "best": best,
    }
    write_json_atomic(FULL_SIMPLEX_JSON, summary)
    return summary


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def doctor(cfg: dict[str, Any]) -> int:
    mesa = require_mesa_dir()
    binary_ok = BINARY.is_file() and os.access(BINARY, os.X_OK)
    print(f"ROOT: {ROOT}")
    print(f"MESA_DIR: {mesa}")
    print(f"binary: {BINARY} {'OK' if binary_ok else 'MISSING -- run make'}")
    print(f"LINA template: {LINA_TEMPLATE} {'OK' if LINA_TEMPLATE.is_file() else 'MISSING'}")
    print(f"target mode: {target_mode(cfg)}")
    print(f"observed period: {cfg_get(cfg, 'observations', 'period_days', 0)} d")
    b = search_bounds(cfg)
    print(
        "search bounds: "
        f"M={b['mass'][0]}..{b['mass'][1]}, "
        f"L={b['luminosity'][0]}..{b['luminosity'][1]}, "
        f"Teff={b['teff'][0]}..{b['teff'][1]}"
    )
    print(
        "grid: "
        f"{cfg_get(cfg, 'search', 'mass_points', 5)} x "
        f"{cfg_get(cfg, 'search', 'luminosity_points', 9)} x "
        f"{cfg_get(cfg, 'search', 'teff_points', 11)}"
    )
    obs = observation_path(cfg)
    print(f"Gaia photometry: {obs} {'FOUND' if obs.is_file() else 'not present yet'}")
    return 0 if binary_ok and LINA_TEMPLATE.is_file() else 1


def smoke(cfg: dict[str, Any], force: bool) -> int:
    r = evaluate_lina(current_seed(cfg), cfg, force=force)
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0 if r.get("status") not in ("crash", "timeout") else 1


def summary_command() -> int:
    for label, path in (
        ("grid", GRID_RANKED_CSV),
        ("LINA simplex", SIMPLEX_CSV),
        ("nonlinear", NONLINEAR_CSV),
        ("full simplex", FULL_SIMPLEX_CSV),
    ):
        print(f"\n{label}: {path}")
        if not path.is_file():
            print("  not run")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print("  empty")
            continue
        if "objective" in df.columns:
            df = df.sort_values("objective")
        cols = [c for c in ("status", "mass", "luminosity", "teff", "period_days", "growth", "objective") if c in df.columns]
        print(df[cols].head(10).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--force", action="store_true")
    p_grid = sub.add_parser("grid")
    p_grid.add_argument("--workers", type=int)
    p_grid.add_argument("--limit", type=int, help="evaluate only the first N grid points (smoke/debug)")
    p_grid.add_argument("--force", action="store_true")
    p_sl = sub.add_parser("simplex-lina")
    p_sl.add_argument("--force", action="store_true")
    p_nl = sub.add_parser("nonlinear")
    p_nl.add_argument("--count", type=int)
    p_nl.add_argument("--force", action="store_true")
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--force", action="store_true")
    p_all = sub.add_parser("all")
    p_all.add_argument("--workers", type=int)
    p_all.add_argument("--force", action="store_true")
    sub.add_parser("summary")
    args = parser.parse_args()

    FIT_DIR.mkdir(exist_ok=True)
    LINA_RUNS.mkdir(parents=True, exist_ok=True)
    NONLINEAR_RUNS.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()

    if args.command == "doctor":
        return doctor(cfg)
    if args.command == "smoke":
        return smoke(cfg, args.force)
    if args.command == "grid":
        run_grid(cfg, args.workers, args.limit, args.force)
        return 0
    if args.command == "simplex-lina":
        run_simplex_lina(cfg, args.force)
        return 0
    if args.command == "nonlinear":
        run_nonlinear_candidates(cfg, args.count, args.force)
        return 0
    if args.command == "fit":
        run_full_simplex(cfg, args.force)
        return 0
    if args.command == "all":
        run_grid(cfg, args.workers, None, args.force)
        run_simplex_lina(cfg, args.force)
        run_nonlinear_candidates(cfg, None, args.force)
        if bool(cfg_get(cfg, "fit", "run_full_simplex_in_all", False)):
            run_full_simplex(cfg, args.force)
        return 0
    if args.command == "summary":
        return summary_command()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
