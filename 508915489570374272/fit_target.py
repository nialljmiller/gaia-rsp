#!/usr/bin/env python3
"""Automated RSP fitting driver for a Gaia radial pulsator.

Encodes the sequence that worked by hand, with the failure modes that were
actually encountered rather than the ones that seemed likely in advance.

    ./fit_target.py all                 # everything, resumable
    ./fit_target.py period              # stage 1 only
    ./fit_target.py --status            # what has been done so far

Stages
  0 screen    analytic period-mean-density check on the adopted M, L, Teff.
              Costs nothing and catches inputs that cannot produce the target
              period before any MESA call. Brad's original L = 4100 Lsun with
              P = 14.79 d would have been rejected here.
  1 period    root-find luminosity so the LINA period matches the target, at
              fixed M and Teff. ~15 s per evaluation. Verifies the mode is
              linearly excited, and steps around luminosities where the
              envelope builder fails (there is a dead band near 6530-6540 for
              the reference target, and isolated holes elsewhere).
  2 baseline  one full settle + Colors run at the current convection settings.
              Measures cost so later stages get a timeout derived from a
              measurement rather than a guess.
  3 shape     scan rsp_alfa. Mixing length is the strongest lever on the rise
              asymmetry. Ranked on rise fraction, NOT on RMS: RMS mixes shape
              and amplitude error, and alfa moves both, so ranking on RMS picks
              the wrong value.
  4 amplitude scan rsp_alfam at the chosen alfa. Eddy viscosity scales the
              limit-cycle amplitude and barely touches shape, so it is fitted
              second and independently.
  5 final     rerun at the chosen parameters, write a summary.

Only the key being scanned is restored to target.toml afterwards; edits made
to other keys while a scan runs are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "target.toml"
STATE = ROOT / "fit_state.json"
WORK = ROOT / "fit_auto"

L_EXPONENT = 0.919          # P ~ L^0.919 at fixed M, Teff (De Somma+2022)
Q_MIN, Q_MAX = 0.030, 0.060  # plausible pulsation constant, days


# ----------------------------------------------------------------- state ---
def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def cfg() -> dict:
    with CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def set_key(key: str, value) -> None:
    text = CONFIG.read_text()
    new, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"expected one {key!r} in target.toml, found {n}")
    CONFIG.write_text(new)


def set_lina(key: str, value: str) -> None:
    """inlist_rsp_LINA is not managed by configure_target.py."""
    p = ROOT / "inlist_rsp_LINA"
    text = p.read_text()
    new, n = re.subn(rf"^(\s*){key} = .*$", rf"\g<1>{key} = {value}", text, flags=re.M)
    if n != 1:
        raise SystemExit(f"expected one {key!r} in inlist_rsp_LINA, found {n}")
    p.write_text(new)


def clear_outputs() -> None:
    for name in ("LOGS_colors", "LOGS_settle", "LOGS"):
        d = ROOT / name
        if d.is_dir():
            shutil.rmtree(d)
    for name in ("settled.mod", "colors_final.mod", "SETTLED_ACCEPTED",
                 "LINA_REJECTED", "SETTLING_REJECTED", "MODE_REJECTED",
                 "ENERGY_REJECTED", "LINA_RESULT"):
        f = ROOT / name
        if f.is_file():
            f.unlink()


# ------------------------------------------------------------- observables --
def observed():
    """G-band epochs folded on the catalogue period."""
    p = ROOT / "observations" / "gaia_epoch_photometry.csv"
    if not p.is_file():
        return None, None, None
    rows = [l.split(",") for l in p.read_text().splitlines()[1:] if l.strip()]
    rows = [r for r in rows if r[1].strip().upper() == "G"]
    if not rows:
        return None, None, None
    t = np.array([float(r[0]) for r in rows])
    m = np.array([float(r[2]) for r in rows])
    p_obs = float(cfg()["observations"]["period_days"])
    ph = ((t - t.min()) / p_obs) % 1.0
    o = np.argsort(ph)
    return ph[o], m[o], float(m.max() - m.min())


def read_history(path: Path) -> dict:
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if "model_number" in l.split())
    names = lines[i].split()
    d = np.genfromtxt(path, skip_header=i + 1)
    return {n: d[:, j] for j, n in enumerate(names)}


def model_curve(h: dict, band="G", nbins=300):
    ph = h["rsp_phase"] % 1.0
    mg = h[band]
    good = np.isfinite(ph) & np.isfinite(mg) & (h["rsp_num_periods"] >= 1)
    ph, mg = ph[good], mg[good]
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nbins - 1)
    c, m = [], []
    for b in range(nbins):
        sel = mg[idx == b]
        if sel.size >= 3:
            c.append(0.5 * (edges[b] + edges[b + 1]))
            m.append(np.median(sel))
    return np.asarray(c), np.asarray(m)


def rise_fraction(ph, mg):
    return float((ph[int(np.argmin(mg))] - ph[int(np.argmax(mg))]) % 1.0)


def fourier(ph, mg):
    cols = [np.ones_like(ph)]
    for k in range(1, 5):
        cols += [np.cos(2 * np.pi * k * ph), np.sin(2 * np.pi * k * ph)]
    coef, *_ = np.linalg.lstsq(np.vstack(cols).T, mg, rcond=None)
    a, p = [], []
    for k in range(1, 5):
        c, s = coef[2 * k - 1], coef[2 * k]
        a.append(np.hypot(c, s))
        p.append(np.arctan2(-s, c))
    return float(a[1] / a[0]) if a[0] else float("nan")


def rms_vs_obs(mp, mm, op, om):
    if op is None:
        return float("nan")
    x = np.concatenate([mp - 1, mp, mp + 1])
    y = np.concatenate([mm, mm, mm])
    best = np.inf
    for sh in np.linspace(0, 1, 500, endpoint=False):
        pred = np.interp((op - sh) % 1.0, x, y)
        pred = pred + np.median(om - pred)
        best = min(best, float(np.sqrt(np.mean((om - pred) ** 2))))
    return best


def metrics():
    """Metrics from the run currently sitting in LOGS_colors (or LOGS)."""
    for name in ("LOGS_colors", "LOGS"):
        h = ROOT / name / "history.data"
        if h.is_file():
            break
    else:
        return None
    d = read_history(h)
    if "G" not in d:
        return None
    op, om, a_obs = observed()
    mp, mm = model_curve(d)
    return dict(
        amplitude=float(mm.max() - mm.min()),
        rise=rise_fraction(mp, mm),
        r21=fourier(mp, mm),
        rms=rms_vs_obs(mp, mm, op, om),
        period=float(np.nanmedian(d["rsp_period_in_days"])),
        a_obs=a_obs,
    )


# ------------------------------------------------------------------ runs ---
def run_lina(timeout=180.0):
    """Full LINA screen; returns mode periods and growth, or None on build fail."""
    shutil.copy(ROOT / "inlist_rsp_LINA", ROOT / "inlist")
    res = ROOT / "LINA_RESULT"
    if res.exists():
        res.unlink()
    WORK.mkdir(exist_ok=True)
    try:
        with (WORK / "lina.log").open("w") as fh:
            subprocess.run(["./rn1"], cwd=ROOT, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if not (res.is_file() and res.stat().st_size):
        return None
    f = {}
    for line in res.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            f[k.strip()] = v.strip()
    try:
        return dict(p0=float(f["mode_0_period_days"]),
                    growth=float(f["mode_0_growth"]),
                    status=f.get("status", "?"))
    except (KeyError, ValueError):
        return None


def run_full(tag: str, timeout: float):
    """Full settle + Colors. Returns (ok, minutes, note)."""
    WORK.mkdir(exist_ok=True)
    clear_outputs()
    log = WORK / f"{tag}.log"
    t0 = time.time()
    try:
        with log.open("w") as fh:
            rc = subprocess.run(["./rn"], cwd=ROOT, stdout=fh,
                                stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        note = "ok" if rc == 0 else f"rc={rc}"
    except subprocess.TimeoutExpired:
        rc, note = -1, "TIMEOUT"
    except OSError as exc:
        rc, note = -1, str(exc)
    mins = (time.time() - t0) / 60
    ok = rc == 0 and (ROOT / "LOGS_colors" / "history.data").is_file()
    if ok:
        dest = WORK / f"LOGS_colors_{tag}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ROOT / "LOGS_colors", dest)
    return ok, mins, note


# ---------------------------------------------------------------- stages ---
def stage_screen(st):
    c = cfg()
    M = c["stellar"]["mass_msun"]
    L = c["stellar"]["luminosity_lsun"]
    T = c["stellar"]["teff_k"]
    P = c["observations"]["period_days"]
    R = math.sqrt(L) / (T / 5772.0) ** 2
    Q = P * math.sqrt(M) / R ** 1.5
    print(f"  M = {M} Msun, L = {L} Lsun, Teff = {T} K  ->  R = {R:.1f} Rsun")
    print(f"  target P = {P:.5f} d  implies Q = {Q:.4f} d")
    ok = Q_MIN <= Q <= Q_MAX
    if ok:
        print(f"  Q is inside the plausible range {Q_MIN}-{Q_MAX}; inputs are "
              f"self-consistent")
    else:
        # invert P = Q R^1.5 / sqrt(M) at a representative Q
        R_need = (P * math.sqrt(M) / 0.045) ** (2.0 / 3.0)
        L_need = R_need ** 2 * (T / 5772.0) ** 4
        print(f"  Q = {Q:.4f} is OUTSIDE {Q_MIN}-{Q_MAX}: these inputs cannot")
        print(f"  produce P = {P:.4f} d. At this Teff you would need")
        print(f"  R ~ {R_need:.0f} Rsun, L ~ {L_need:.0f} Lsun.")
    st["screen"] = dict(M=M, L=L, teff=T, R=R, Q=Q, ok=ok)
    return ok


def stage_period(st, args):
    c = cfg()
    target_lin = c["observations"]["period_days"] / (1.0 + args.nl_offset)
    L = c["stellar"]["luminosity_lsun"]
    print(f"  matching LINA P0 = {target_lin:.4f} d "
          f"(observed / {1+args.nl_offset:.4f})")
    tried = []
    for it in range(args.max_iter):
        set_key("luminosity_lsun", f"{L:.1f}")
        set_lina("RSP_L", f"{L:.6f}d0")
        r = run_lina()
        if r is None:
            # envelope builder failures come in bands, not isolated points
            step = 20.0 * (1 + it)
            print(f"    L = {L:7.0f}  BUILD FAILED, stepping +{step:.0f}")
            tried.append((L, None, None))
            L += step
            continue
        tried.append((L, r["p0"], r["growth"]))
        err = r["p0"] / target_lin - 1.0
        print(f"    L = {L:7.0f}  P0 = {r['p0']:8.4f}  growth = {r['growth']:+.3e}"
              f"  ({100*err:+.2f}%)  {r['status']}")
        if r["growth"] <= 0:
            print("    mode is damped here; move Teff cooler and rerun this stage")
            st["period"] = dict(ok=False, reason="damped", tried=tried)
            return False
        if abs(err) < args.period_tol:
            st["period"] = dict(ok=True, L=L, p0=r["p0"],
                                growth=r["growth"], tried=tried)
            print(f"  converged: L = {L:.0f}, P0 = {r['p0']:.4f} d")
            return True
        L *= (target_lin / r["p0"]) ** (1.0 / L_EXPONENT)
    st["period"] = dict(ok=False, reason="no convergence", tried=tried)
    return False


def stage_baseline(st, args):
    ok, mins, note = run_full("baseline", args.timeout)
    if not ok:
        print(f"  FAILED after {mins:.1f} min ({note}); see fit_auto/baseline.log")
        st["baseline"] = dict(ok=False, note=note, minutes=mins)
        return False
    m = metrics()
    print(f"  {mins:.1f} min   P = {m['period']:.5f} d   A_G = {m['amplitude']:.3f}"
          f" (obs {m['a_obs']:.3f})   rise = {m['rise']:.3f}   RMS = {m['rms']:.4f}")
    st["baseline"] = dict(ok=True, minutes=mins, **m)
    return True


def scan_stage(st, key, values, args, rank):
    """Scan one target.toml key. rank(metrics) -> lower is better."""
    original = cfg()["run"][key]
    timeout = args.timeout
    if st.get("baseline", {}).get("minutes"):
        timeout = max(600.0, st["baseline"]["minutes"] * 60 * 3)
    print(f"  scanning {key} (currently {original}), timeout {timeout/60:.0f} min")
    rows = []
    try:
        for v in values:
            set_key(key, v)
            ok, mins, note = run_full(f"{key}{v}", timeout)
            if not ok:
                print(f"    {key} = {v:<6}  FAILED ({note}, {mins:.1f} min)")
                rows.append(dict(value=v, ok=False, note=note))
                continue
            m = metrics()
            print(f"    {key} = {v:<6}  A_G = {m['amplitude']:.3f}"
                  f"  rise = {m['rise']:.3f}  R21 = {m['r21']:.3f}"
                  f"  RMS = {m['rms']:.4f}  P = {m['period']:.4f}  ({mins:.1f} min)")
            rows.append(dict(value=v, ok=True, minutes=mins, **m))
    finally:
        set_key(key, original)
    good = [r for r in rows if r.get("ok")]
    if not good:
        st[key] = dict(ok=False, rows=rows)
        return None
    best = min(good, key=rank)
    print(f"  best {key} = {best['value']}")
    st[key] = dict(ok=True, best=best["value"], rows=rows)
    return best["value"]


def stage_shape(st, args):
    # rank on rise fraction, not RMS: alfa moves amplitude too, and RMS would
    # then select on amplitude error rather than on shape
    target = st.get("baseline", {}).get("a_obs")
    op, om, _ = observed()
    obs_rise = rise_fraction(op, om) if op is not None else 0.287
    print(f"  ranking on |rise - {obs_rise:.3f}|")
    return scan_stage(st, "rsp_alfa", args.alfa, args,
                      lambda r: abs(r["rise"] - obs_rise))


def stage_amplitude(st, args):
    a_obs = st.get("baseline", {}).get("a_obs") or 0.946
    print(f"  ranking on |A_G - {a_obs:.3f}|")
    if st.get("rsp_alfa", {}).get("ok"):
        set_key("rsp_alfa", st["rsp_alfa"]["best"])
        print(f"  holding rsp_alfa = {st['rsp_alfa']['best']}")
    return scan_stage(st, "rsp_alfam", args.alfam, args,
                      lambda r: abs(r["amplitude"] - a_obs))


def stage_final(st, args):
    for key in ("rsp_alfa", "rsp_alfam"):
        if st.get(key, {}).get("ok"):
            set_key(key, st[key]["best"])
    if st.get("period", {}).get("ok"):
        set_key("luminosity_lsun", f"{st['period']['L']:.1f}")
    c = cfg()
    print(f"  M = {c['stellar']['mass_msun']}  L = {c['stellar']['luminosity_lsun']}"
          f"  Teff = {c['stellar']['teff_k']}  Z = {c['stellar']['z']}")
    print(f"  alfa = {c['run']['rsp_alfa']}  alfam = {c['run']['rsp_alfam']}")
    ok, mins, note = run_full("final", args.timeout)
    if not ok:
        st["final"] = dict(ok=False, note=note)
        return False
    m = metrics()
    st["final"] = dict(ok=True, minutes=mins, **m)
    print()
    print(f"  period     {m['period']:.5f} d  vs {c['observations']['period_days']:.5f}"
          f"  ({100*(m['period']/c['observations']['period_days']-1):+.2f}%)")
    print(f"  amplitude  {m['amplitude']:.3f} mag  vs {m['a_obs']:.3f}"
          f"  ({100*(m['amplitude']/m['a_obs']-1):+.1f}%)")
    print(f"  rise       {m['rise']:.3f}   R21 {m['r21']:.3f}   RMS {m['rms']:.4f}")
    print()
    print("  target.toml now holds the fitted parameters; LOGS_colors is the")
    print("  final model. Plot with plot_gaia_fit.py and plot_cmd_loop.py.")
    return True


STAGES = [
    ("screen", stage_screen, "analytic consistency check"),
    ("period", stage_period, "root-find L on the LINA period"),
    ("baseline", stage_baseline, "one full run, measures cost"),
    ("shape", stage_shape, "scan rsp_alfa on rise fraction"),
    ("amplitude", stage_amplitude, "scan rsp_alfam on G amplitude"),
    ("final", stage_final, "rerun at the fitted parameters"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all"] + [s[0] for s in STAGES])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--period-tol", type=float, default=0.003)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--nl-offset", type=float, default=0.0036,
                    help="nonlinear period runs this much longer than linear")
    ap.add_argument("--alfa", type=float, nargs="+",
                    default=[1.0, 1.2, 1.3, 1.5])
    ap.add_argument("--alfam", type=float, nargs="+",
                    default=[0.22, 0.26, 0.30])
    ap.add_argument("--force", action="store_true",
                    help="rerun stages already recorded as done")
    args = ap.parse_args()

    st = load_state()

    if args.status:
        for name, _, desc in STAGES:
            k = {"shape": "rsp_alfa", "amplitude": "rsp_alfam"}.get(name, name)
            r = st.get(k)
            mark = "done" if (r and r.get("ok")) else \
                   ("failed" if r else "-")
            print(f"  {name:<10} {mark:<8} {desc}")
        return 0

    wanted = [s for s in STAGES if args.stage in ("all", s[0])]
    for name, fn, desc in wanted:
        k = {"shape": "rsp_alfa", "amplitude": "rsp_alfam"}.get(name, name)
        if not args.force and st.get(k, {}).get("ok") and args.stage == "all":
            print(f"[{name}] already done, skipping (--force to redo)")
            continue
        print(f"[{name}] {desc}")
        try:
            ok = fn(st, args) if name != "screen" else fn(st)
        except KeyboardInterrupt:
            save_state(st)
            print("\ninterrupted; state saved, rerun to resume", file=sys.stderr)
            return 1
        save_state(st)
        if not ok:
            if args.stage == "all":
                print(f"[{name}] stopped. Fix and rerun to resume.")
            return 1
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
