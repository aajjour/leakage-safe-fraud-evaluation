# Leakage-Safe Temporal Evaluation of Credit-Card Fraud Detection

This repository contains the computational and reproducibility materials accompanying the manuscript:

**A Reproducible Computational Framework for Leakage-Safe Temporal Evaluation, Calibration, and Decision Support in Credit Card Fraud Detection**

## Authors
- Ali Al Mazari — Jadara University
- Ahmed Bani-Mustafa — Australian University
- Mohammad Hayel Al-Refai — Jadara University
- Hassan Al Sukhni — Jadara University

## Repository contents

- `code/` — analysis, calibration, temporal benchmarking, random-comparator, and OPBGB implementations.
- `results/` — frozen derived outputs used to support the reported analyses.
- `tables/` — manuscript-facing summary tables and reporting materials.
- `docs/` — data-acquisition instructions, leakage-safe protocol details, method-to-output mapping, and evidence-status notes.
- `environment/` — environment manifest and dependency lock information.
- `execution_logs/` — verification and execution logs.
- `supplementary_material/statistical_uncertainty/` — fold-level uncertainty and resampling materials.
- `SHA256_MANIFEST.txt` — SHA-256 checksums for repository files.

## Data availability

The raw Sparkov and IEEE-CIS datasets are third-party resources and are **not redistributed** in this repository. See `docs/DATA_DOWNLOAD_INSTRUCTIONS.md` for the original data sources, expected filenames, and integrity-verification instructions.

## Reproduction workflow

1. Obtain the third-party datasets using `docs/DATA_DOWNLOAD_INSTRUCTIONS.md`.
2. Verify the expected raw-data checksums where supplied.
3. Install the documented dependencies in `environment/`.
4. Follow the dataset-specific `README` and `setup_and_run.sh` files under `code/`.
5. Compare reproduced outputs with the frozen files under `results/` and `tables/`.
6. Use `docs/METHOD_TO_OUTPUT_MAP.md` to map manuscript analyses to scripts and outputs.
7. Verify repository integrity against `SHA256_MANIFEST.txt`.

## Scope of the archive

The repository is intended to make the evaluation workflow auditable and reproducible. It contains code, configurations, derived outputs, manifests, and verification materials, but it does not redistribute restricted or third-party raw transaction data.

## Citation

Cite the archived release using Zenodo DOI: https://doi.org/10.5281/zenodo.22311831.

## Repository version

Release v1.0.1, permanently archived on Zenodo under DOI 10.5281/zenodo.22311831; prepared for the revised Array submission, September 2026.
