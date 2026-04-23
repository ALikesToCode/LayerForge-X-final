# Report Artifacts

This folder collects the report artifacts referenced by the README and `PROJECT_MANIFEST.json`.

- `metrics_snapshots/` contains compact JSON summaries copied from the measured local `runs/` and `results/` directories.
- `figure_sources/figure_manifest.json` records which raw runs and datasets were used to build the report figures and whether those dependencies are omitted from the submission ZIP.
- `command_log.md` lists the command families used to generate the copied artifacts and the verified package/runtime versions used for the current archive refresh.

The goal is to keep the repository reviewable even when heavyweight directories such as `data/`, `runs/`, and `results/` are excluded from the public tree and submission ZIP.

Treat `PROJECT_MANIFEST.json`, `metrics_snapshots/`, and `command_log.md` as the canonical reported artifacts for the repository and submission ZIP.
