#!/usr/bin/env python3
"""Find which RSP_alfap values the envelope builder can actually relax.

Set C died in RELAX_ENV with ALFAP diverging to -3e8 before the 8001-iteration
limit. That happens at model creation, before LINA, so it can be tested in ~15 s
per value instead of a full pipeline run.

Sweeps alfap (turbulent pressure) and optionally RSP_relax_dm_tolerance, using
the LINA-only inlist so nothing expensive runs. Reports whether the model built
and, when it did, the linear period and growth rate.

    ./scan_alfap_build.py 0.05 0.1 0.2 0.4 0.667
    ./scan_alfap_build.py --relax-tol 1e-5 0.4 0.667
    ./scan_alfap_build.py --alfat 0.0 0.667      # isolate turbulent flux

inlist_rsp_LINA is read but never modified; only the scratch inlist is written.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINA = ROOT / "inlist_rsp_LINA"
SCRATCH = ROOT / ".inlist_alfap_scan"
LOGDIR = ROOT / "scan_alfap"


def patch(text: str, key: str, value: str) -> str:
    """Replace the assignment if present, otherwise insert it in &controls."""
    new, n = re.subn(rf"^(\s*){key} = .*$", rf"\g<1>{key} = {value}", text,
                     flags=re.MULTILINE)
    if n:
        return new
    anchor = re.search(r"^(\s*)RSP_alfam = .*$", text, re.MULTILINE)
    if not anchor:
        raise SystemExit(f"cannot place {key}: no RSP_alfam line in {LINA.name}")
    indent = anchor.group(1)
    return (text[:anchor.end()] + f"\n{indent}{key} = {value}"
            + text[anchor.end():])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("alfap", nargs="+", type=float)
    ap.add_argument("--alfat", type=float, default=0.01)
    ap.add_argument("--gammar", type=float, default=0.0)
    ap.add_argument("--relax-tol", type=float, default=None,
                    help="RSP_relax_dm_tolerance; omit to leave at default")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    if not LINA.is_file():
        raise SystemExit(f"missing {LINA}")
    LOGDIR.mkdir(exist_ok=True)
    base = LINA.read_text()

    print(f"alfat = {args.alfat}, gammar = {args.gammar}, "
          f"relax_dm_tolerance = {args.relax_tol or 'default'}")
    print(f"\n{'alfap':<10} {'build':<14} {'P0(d)':<16} {'growth':<16} {'s':<6}")
    print(f"{'-----':<10} {'-----':<14} {'-----':<16} {'------':<16} {'-':<6}")

    for value in args.alfap:
        text = patch(base, "RSP_alfap", f"{value}d0")
        text = patch(text, "RSP_alfat", f"{args.alfat}d0")
        text = patch(text, "RSP_gammar", f"{args.gammar}d0")
        if args.relax_tol is not None:
            text = patch(text, "RSP_relax_dm_tolerance", f"{args.relax_tol}d0")
        SCRATCH.write_text(text)
        (ROOT / "inlist").write_text(text)

        result = ROOT / "LINA_RESULT"
        if result.exists():
            result.unlink()

        log = LOGDIR / f"alfap{value}.log"
        started = time.time()
        try:
            with log.open("w") as fh:
                subprocess.run(["./rn1"], cwd=ROOT, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=args.timeout)
            note = ""
        except subprocess.TimeoutExpired:
            note = "TIMEOUT"
        except OSError as exc:
            note = f"error: {exc}"
        secs = time.time() - started

        if result.is_file() and result.stat().st_size:
            fields = dict(
                l.split("=", 1) for l in result.read_text().splitlines()
                if "=" in l)
            period = fields.get("mode_0_period_days", "?").strip()
            growth = fields.get("mode_0_growth", "?").strip()
            status = fields.get("status", "?").strip()
            print(f"{value:<10} {'ok:'+status:<14} {period:<16} "
                  f"{growth:<16} {secs:<6.0f}")
        else:
            body = log.read_text()
            if "RELAX_ENV" in body:
                why = "RELAX_ENV"
            elif "T_anchor" in body:
                why = "T_anchor"
            else:
                why = note or "unknown"
            print(f"{value:<10} {'FAILED:'+why:<14} {'--':<16} {'--':<16} "
                  f"{secs:<6.0f}")

    SCRATCH.unlink(missing_ok=True)
    print(f"\nlogs in {LOGDIR.name}/; {LINA.name} was not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
