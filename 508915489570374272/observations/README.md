# Observational data

Place the cross-matched Gaia target metadata and phase-resolved light curves here.
The eventual fitter should consume a tidy table with at least:

- `phase` or observation time plus an ephemeris;
- `band` (`G`, `Gbp`, or `Grp`);
- `magnitude`;
- `magnitude_error`;
- an optional quality flag.

Catalogue values should be written into `../target.toml`; they are not hard-coded
into the setup utility.
