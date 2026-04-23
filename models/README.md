# Models

This directory stores lightweight repository-native model artifacts that are shipped with the repo.

## Included artifact

- `order_ranker_fast.json`: compact learned pairwise near/far ordering model used by the fast ordering path

## Relevance

The ranker is used by the graph-ordering stack when the repository selects the lightweight learned ordering path rather than the purely heuristic boundary-order baseline.

## Reproduction note

The shipped repository is oriented around inference and evaluation. If the ranker is retrained, update this artifact together with the corresponding measured summaries and manifest entries so the public evidence pack remains coherent.
