---
layout: default
title: TINTOlib Benchmark Results
---

# TINTOlib Benchmark Hub

TINTOlib provides a suite of vision-inspired baselines for tabular learning. This hub gathers curated benchmark results across regression, binary classification, and multiclass classification tasks using the new per-dataset data sources in `_data/datasets/`.

## Explore the Results

- [Regression Benchmarks](regression.html)
- [Binary Classification Benchmarks](binary.html)
- [Multiclass Classification Benchmarks](multiclass.html)

## Dataset Coverage

| Task | Datasets |
| --- | --- |
| Regression | {{ site.data.datasets.regression | size }} |
| Binary | {{ site.data.datasets.binary | size }} |
| Multiclass | {{ site.data.datasets.multiclass | size }} |

## How to Read the Reports

Each task page lists every dataset with:

1. **Dataset information** – core statistics plus links back to the original source.
2. **Leaderboard** – baseline tree models for quick reference.
3. **Architecture sections** – detailed metrics for TINTOlib transformations (ViT, CNN, hybrids, etc.).

Use these pages to compare architectures, check data characteristics, and identify promising starting points for new experiments.
