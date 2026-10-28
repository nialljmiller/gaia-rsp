#!/usr/bin/env bash
# Scan luminosity values through the LINA screen to find ones where the RSP
# envelope builder succeeds, and report the resulting period and growth rate.
#
# The builder fails at scattered points in parameter space for reasons that have
# nothing to do with the physics: L = 6700 builds at Teff = 5400 and L = 6530
# does not, despite being 2.5% apart. This finds a nearby value that works.
#
# Each attempt takes roughly 6 s on failure, 15 s on success. inlist_rsp_LINA is
# read but never modified; only the scratch inlist is overwritten.
#
# Usage:
#   ./scan_lina_L.sh 6450 6480 6510 6530 6560 6590 6620 6650 6680 6700

set -u

ORIG=inlist_rsp_LINA
TMP=.inlist_lina_scan
LOGDIR=scan_logs

if [[ $# -eq 0 ]]; then
    echo "usage: $0 L1 [L2 L3 ...]   (luminosities in Lsun)" >&2
    exit 1
fi

if [[ ! -f "$ORIG" ]]; then
    echo "missing $ORIG" >&2
    exit 1
fi

mkdir -p "$LOGDIR"

printf '%-10s %-12s %-16s %-16s\n' "L(Lsun)" "build" "P0(days)" "growth"
printf '%-10s %-12s %-16s %-16s\n' "-------" "-----" "--------" "------"

for L in "$@"; do
    sed -E "s/^([[:space:]]*)RSP_L = .*/\1RSP_L = ${L}.000000d0/" "$ORIG" > "$TMP"

    if ! grep -q "RSP_L = ${L}" "$TMP"; then
        printf '%-10s %-12s\n' "$L" "SED-FAILED"
        continue
    fi

    cp "$TMP" inlist
    rm -f LINA_RESULT
    ./rn1 > "$LOGDIR/L${L}.log" 2>&1

    if [[ -s LINA_RESULT ]]; then
        period=$(awk -F'=' '/^mode_0_period_days/ {print $2}' LINA_RESULT | tr -d ' ')
        growth=$(awk -F'=' '/^mode_0_growth/ {print $2}' LINA_RESULT | tr -d ' ')
        status=$(awk -F'=' '/^status/ {print $2}' LINA_RESULT | tr -d ' ')
        printf '%-10s %-12s %-16s %-16s\n' "$L" "ok:$status" "$period" "$growth"
        cp LINA_RESULT "$LOGDIR/LINA_RESULT_L${L}"
    else
        printf '%-10s %-12s\n' "$L" "BUILD-FAILED"
    fi
done

rm -f "$TMP"
echo
echo "logs in $LOGDIR/; $ORIG was not modified"
