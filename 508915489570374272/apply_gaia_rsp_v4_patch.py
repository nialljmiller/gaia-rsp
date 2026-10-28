#!/usr/bin/env python3
"""Apply Gaia RSP work-directory v4 in place, preserving target.toml."""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / ".gaia_rsp_v4_payload"
TARGET = ROOT / "target.toml"

RUN_DEFAULTS = {
    "target_mode": "0",
    "minimum_linear_growth": "0.0",
    "minimum_delta_r_rsun": "0.05",
    "minimum_delta_mag": "0.01",
    "amplitude_relative_tolerance": "0.02",
}

FILES = [
    "mk", "rn", "rn1", "re", "clean", "Makefile", "makefile",
    "configure_target.py", "inlist_rsp_RR_Lyrae", "inlist_rsp_RR_Lyrae_colors",
    "history_columns_colors.list", "README.md", "src/run_star_extras.f90",
]


def require_mesa_dir() -> Path:
    raw = os.environ.get("MESA_DIR", "").strip()
    if not raw:
        raise SystemExit("ERROR: MESA_DIR must be set before applying this patch")
    mesa_dir = Path(raw).expanduser().resolve()
    required = mesa_dir / "star" / "work" / "Makefile"
    if not required.is_file():
        raise SystemExit(
            f"ERROR: invalid MESA_DIR={mesa_dir}\n"
            f"       missing {required}"
        )
    return mesa_dir


def add_missing_run_keys(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    section_start = None
    section_end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[run]":
            section_start = i
            continue
        if section_start is not None and i > section_start and stripped.startswith("[") and stripped.endswith("]"):
            section_end = i
            break
    if section_start is None:
        raise SystemExit(f"ERROR: {path} has no [run] section")

    existing: set[str] = set()
    for line in lines[section_start + 1:section_end]:
        stripped = line.split("#", 1)[0].strip()
        if "=" in stripped:
            existing.add(stripped.split("=", 1)[0].strip())

    missing = [key for key in RUN_DEFAULTS if key not in existing]
    if missing:
        addition = ["", "# Added by Gaia RSP v4: mode screen and nonzero-amplitude settling."]
        addition.extend(f"{key} = {RUN_DEFAULTS[key]}" for key in missing)
        lines[section_end:section_end] = addition
        path.write_text("\n".join(lines) + "\n")
    return missing


def archive_failed_run(stamp: str) -> Path | None:
    names = ["settled.mod", "colors_final.mod", "LOGS_settle", "LOGS_colors", "SED",
             "LINA_REJECTED", "SETTLING_REJECTED", "SETTLED_ACCEPTED"]
    existing = [ROOT / name for name in names if (ROOT / name).exists()]
    if not existing:
        return None
    archive = ROOT / f"pre_v4_run_{stamp}"
    archive.mkdir()
    for src in existing:
        shutil.move(str(src), str(archive / src.name))
    return archive


def copy_payload(backup: Path) -> None:
    for rel in FILES:
        src = PAYLOAD / rel
        dst = ROOT / rel
        if not src.is_file():
            raise SystemExit(f"ERROR: patch payload is missing {src}")
        if dst.exists() or dst.is_symlink():
            b = backup / rel
            b.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                b.write_text(f"symlink -> {os.readlink(dst)}\n")
            elif dst.is_file():
                shutil.copy2(dst, b)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel in ["mk", "rn", "rn1", "re", "clean", "configure_target.py"]:
        (ROOT / rel).chmod(0o755)
    (ROOT / ".gaia_rsp_workdir_version").write_text("4.0.0\n")


def validate() -> None:
    py_compile.compile(str(ROOT / "configure_target.py"), doraise=True)
    for rel in ["mk", "rn", "rn1", "re", "clean"]:
        subprocess.run(["bash", "-n", str(ROOT / rel)], check=True)

    # Colors-owned history names must NOT appear in the user history list.
    h = (ROOT / "history_columns_colors.list").read_text()
    forbidden = ["Mag_bol", "Flux_bol", "Interp_rad", "Gbp", "Grp"]
    present = [name for name in forbidden if any(line.strip() == name for line in h.splitlines())]
    # 'G' needs exact-line handling so prose/comments do not match.
    if any(line.strip() == "G" for line in h.splitlines()):
        present.append("G")
    if present:
        raise SystemExit("ERROR: Colors-owned names remain in history_columns_colors.list: " + ", ".join(present))

    subprocess.run([sys.executable, str(ROOT / "configure_target.py"), "--check-only"], check=True)


def main() -> int:
    mesa_dir = require_mesa_dir()
    if not TARGET.is_file():
        raise SystemExit(f"ERROR: run this from the Gaia RSP work directory; missing {TARGET}")
    if not PAYLOAD.is_dir():
        raise SystemExit(f"ERROR: patch payload is missing: {PAYLOAD}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / f".gaia_rsp_pre_v4_files_{stamp}"
    backup.mkdir()
    shutil.copy2(TARGET, backup / "target.toml")

    archived = archive_failed_run(stamp)
    missing = add_missing_run_keys(TARGET)
    copy_payload(backup)
    validate()

    print()
    print("Gaia RSP v4 patch applied")
    print(f"MESA_DIR={mesa_dir}")
    print(f"file backup={backup}")
    if archived is not None:
        print(f"previous run archived={archived}")
    if missing:
        print("added target.toml keys: " + ", ".join(missing))
    else:
        print("target.toml already contained all v4 run keys")
    print("Colors history list corrected: Colors columns are appended by MESA automatically")
    print("next: ./clean && ./mk && ./rn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
