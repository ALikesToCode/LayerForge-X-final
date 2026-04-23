# Documentation Map

This directory contains the public documentation surface for the repository.

## Start here

- [Submission index](SUBMISSION_INDEX.md)
- [Final report pack](final_report_pack/)
- [Final report pack guide](final_report_pack/README.md)
- [Figure index](FIGURES.md)
- [Current results summary](RESULTS_SUMMARY_CURRENT.md)
- [Qwen comparison](QWEN_IMAGE_LAYERED_COMPARISON.md)

## Reference documents

- [Method overview](METHOD.md)
- [Benchmarking summary](BENCHMARKING.md)
- [Benchmarking protocol](BENCHMARKING_PROTOCOL_FINAL.md)
- [Novelty and ablations](NOVELTY_AND_ABLATIONS_FINAL.md)
- [Report tables](REPORT_TABLES.md)
- [Public benchmark notes](PUBLIC_BENCHMARKS.md)
- [Product and launch framing](PRODUCT_ARCHITECTURE_AND_LAUNCH.md)
- [References](REFERENCES.md)

## Public web surfaces

- [Project site entry](index.html)
- [About page](about.html)
- [Local browser UI](webui.html)

## Contracts and schemas

- [API README](api/README.md)
- [OpenAPI contract](api/openapi.yaml)
- `../schemas/dalg.schema.json`

## Generated assets

- [Figures](figures/)
- [Site data payload](site-data/project_site.json)
- [Final report build manifest](final_report_pack/build_manifest.json)

## Notes on omitted heavyweight artifacts

The public repository and submission archive do not include the heavyweight local `runs/`, `results/`, or `data/` directories used during measurement. The shipped source of truth is the evidence pack rooted at `../PROJECT_MANIFEST.json` and `../report_artifacts/`.
