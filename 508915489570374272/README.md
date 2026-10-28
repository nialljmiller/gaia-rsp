# Gaia RSP fitting work directory

This work directory always uses the active `$MESA_DIR`. The local `Makefile`
includes `$MESA_DIR/star/work/Makefile`, and every entry-point validates and
prints the resolved MESA installation before running.

The two-stage calculation is:

1. Build the RSP model and inspect the LINA mode selected by
   `[run] target_mode` (`0` is the fundamental mode). Stage 1 is rejected
   immediately when that mode has growth less than or equal to
   `minimum_linear_growth`.
2. Settle the nonlinear calculation. A model is accepted only when
   `abs(rsp_GREKM)` is small and either `rsp_DeltaR` or `rsp_DeltaMag` remains
   above its configured floor and changes by less than the amplitude tolerance
   for the required number of consecutive periods.
3. Restart the accepted `settled.mod` with Colors and write Gaia `Gbp`, `G`, and
   `Grp` to `LOGS_colors/history.data`.

A damped LINA model writes `LINA_REJECTED`. A model that reaches the maximum
settling period without an accepted limit cycle writes `SETTLING_REJECTED`.
Neither case is allowed to proceed to the Colors stage.

## Run

```bash
cd <work-directory>
$EDITOR target.toml
python configure_target.py
./mk
./rn
```

With `gaia_instrument = "auto"` and `stellar_atmosphere = "auto"`, path
resolution prioritizes `$MESA_DIR/data/colors_data`. The resolved installation
and paths are recorded in `target_resolved.json`.

MESA photometry is evaluated at 10 pc. Catalogue distance and extinction are
applied later in the fitting likelihood, so they do not require another RSP run.

## v4 notes

- Every build/run helper uses the active `$MESA_DIR`; no test-suite makefile path is hard-coded.
- Stage 1 screens the selected zero-based LINA mode before nonlinear settling.
- A damped mode writes `LINA_REJECTED` and stage 2 is not run.
- Reaching the settling-period ceiling without a nonzero stable amplitude writes `SETTLING_REJECTED`; it is not accepted as a settled model.
- `history_columns_colors.list` contains only ordinary stellar/RSP columns. MESA appends `Mag_bol`, `Flux_bol`, `Interp_rad`, and the loaded Gaia filters automatically when `use_colors = .true.`.
