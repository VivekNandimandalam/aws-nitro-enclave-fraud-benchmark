# AWS Nitro Enclave Fraud Detection Benchmark

This repository contains the implementation and reproducible experiment for an MSc Cloud Computing research project:

> Evaluating the computational and communication overhead of AWS Nitro Enclaves for XGBoost-based financial fraud-detection inference.

## Research scope

The experiment compares a resource-controlled native XGBoost inference service with the equivalent inference service running inside an AWS Nitro Enclave. The primary outcomes are steady-state end-to-end latency, throughput, process resource use, and parent-to-enclave vsock overhead.

The repository does not yet contain an AWS deployment, an enclave image, a vsock service, a benchmark harness, or experimental results.

## Current local state

The following local artefacts already exist but are deliberately ignored by Git:

- `creditcard.csv` — source dataset.
- `models/xgb_fraud.json` — current trained XGBoost artefact.
- `models/test_payload.csv` — preprocessed test records for future benchmark input.
- `models/lr_fraud.joblib` — exploratory logistic-regression artefact.

`train_models.py` is the original preliminary training script and is retained only for historical reference. Do not run it. The supported, leakage-free training workflow is `model/train.py`.

## Repository layout

```text
model/       Training, preprocessing, validation, and model metadata
baseline/    Native inference implementation
enclave/     Enclave inference implementation and container definition
parent/      Parent-instance relay and vsock protocol
client/      Benchmark/load-generation client
config/      Versioned non-secret experiment configuration
scripts/     Reproducible local and AWS setup/build commands
analysis/    Data processing, statistical analysis, and figure generation
results/     Generated raw data, processed data, and figures (not committed)
models/      Generated local model artefacts (not committed)
```

## Local environment

Create a Python 3.12 virtual environment and install the direct dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Development rules

- Never commit AWS credentials, private keys, `.env` files, the dataset, or generated benchmark results.
- Record each final model's SHA-256 hash, preprocessing version, dataset source, train/test split, seed, and package versions.
- Use the same frozen model artefact and test inputs in the baseline and enclave conditions.
- Do not fabricate benchmark data or conclusions.

## Train and freeze the benchmark model

With `creditcard.csv` in the repository root, run:

```powershell
python model/train.py
```

The script performs a stratified 80/20 split before fitting `RobustScaler`, trains the frozen XGBoost model, and writes these ignored artefacts to `models/`:

- `xgb_fraud.json` — the model for both benchmark conditions.
- `preprocessor.joblib` — fitted preprocessing specification.
- `benchmark_payload.csv` and `benchmark_labels.csv` — fixed, aligned test inputs and labels.
- `validation_metrics.json` and `model_metadata.json` — validation and reproducibility metadata.
- `SHA256SUMS.txt` — SHA-256 integrity hashes.

Use the exact frozen artefacts in both the baseline and enclave environments. `models/test_payload.csv` is a legacy preliminary payload and must not be used.

## Next milestone

Build and test a local native inference service that consumes `benchmark_payload.csv`. Do not provision AWS infrastructure until that service is functionally verified.