# Report Artifacts

This folder is the submission-safe evidence pack for the claims made in the README and `PROJECT_MANIFEST.json`.

- `metrics_snapshots/` contains compact JSON summaries copied from the measured local `runs/` and `results/` directories.
- `figure_sources/figure_manifest.json` records which raw runs and datasets were used to build the report figures and whether those dependencies are omitted from the submission ZIP.
- `command_log.md` lists the command families used to generate the copied artifacts and the verified package/runtime versions used for the current archive refresh.

The goal is to keep the archive auditable even when heavyweight directories such as `data/`, `runs/`, and `results/` are excluded from a ZIP submission.

Treat `PROJECT_MANIFEST.json`, `metrics_snapshots/`, and `command_log.md` as the canonical evidence pack in the submission archive.
