# Gaia RSP: Brad's pulsator project

MESA radial-pulsation and Gaia photometry fitting for source **508915489570374272** (observed period 14.78917 days).

The canonical work directory is `508915489570374272/`. The September 29 Windybox uppercase `GAIA_RSP` copy supplied the active project. Older laptop and Windybox variants are preserved in the initial research-snapshots release, with SHA-256 manifests in `provenance/`. This consolidation does not establish that the most recent interrupted run completed successfully.

## Work on either machine

```sh
cd ~/MESA/gaia-rsp
git pull --ff-only
cd 508915489570374272
python configure_target.py --check
```

Set `MESA_DIR` to the local compatible MESA installation and activate its SDK first. Colors paths use `auto` so they resolve from that installation. See the target's README for build and fitting commands. Generated inlists may contain old machine paths; regenerate them with `python configure_target.py` before building. Compiled binaries must be rebuilt on the machine where they will run.

Commit source/configuration changes and push before switching machines. Pull on the other machine before editing. Git tracks scripts, inputs, observations, plots, and lightweight fit summaries; simulation histories, binary models, photos, build products, and logs remain local.

## Preserved original results

All files from the three original copies are stored once per unique content hash in the `research-snapshots.tar.gz` asset of release `initial-consolidation`. Each snapshot manifest records original relative paths, modes, timestamps, sizes, and SHA-256 hashes. This includes conflicting source versions and results; nothing was silently overwritten in the archive.

```sh
gh release download initial-consolidation --repo nialljmiller/gaia-rsp --pattern research-snapshots.tar.gz --dir /tmp
python tools/restore_snapshot.py /tmp/research-snapshots.tar.gz --snapshot laptop --output /tmp/gaia-laptop
```

Snapshots: `laptop`, `windy-older`, `windy-current`. Restore to a separate directory when comparing historical work. The release archive is the historical backup, not another active project tree.

Legacy `~/MESA/Gaia_RSP` and (on Windybox) `~/MESA/GAIA_RSP` paths link to this single checkout.
