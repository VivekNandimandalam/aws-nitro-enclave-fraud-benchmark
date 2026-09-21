# Native Baseline Service

Run the native XGBoost service from the repository root:

```powershell
python baseline/server.py
```

The service verifies the SHA-256 of `models/xgb_fraud.json` against `models/model_metadata.json` before it accepts requests. It listens on `127.0.0.1:8000` by default.

- `GET /health` returns the model hash and expected feature count.
- `POST /predict` accepts `{"instances": [[feature_1, ..., feature_30], ...]}` and returns a probability, a thresholded class, `inference_ms`, and `request_total_ms`.

The feature order is exactly the order recorded in `model_metadata.json`; use rows from `models/benchmark_payload.csv` unchanged.

In a second PowerShell window, run the smoke test:

```powershell
python baseline/smoke_test.py
```

This is a functional check, not an experiment. The benchmark client and measurement protocol will be implemented separately.