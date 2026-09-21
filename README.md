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

`train_models.py` is the existing preliminary training script. It will be corrected, version-pinned, and documented before the model is frozen for the experiment. Do not treat the current artefacts as the final experimental model.

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

## Next milestone

Correct and document the deterministic training pipeline, then freeze a single model artefact and its metadata before implementing either inference service.
