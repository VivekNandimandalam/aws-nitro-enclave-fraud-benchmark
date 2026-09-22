# Benchmark Client

`benchmark.py` produces reproducible raw HTTP measurements for the native baseline and, later, the enclave parent relay.

Start the service in one terminal:

```powershell
python baseline/server.py
```

Then execute a short functional benchmark in a second activated virtual environment:

```powershell
python client/benchmark.py --batch-sizes 10 100 --warmup-runs 2 --repetitions 3
```

For the planned full local baseline experiment, use the default batch sizes (`10`, `100`, `1000`, `10000`) and increase `--repetitions` only after confirming that the 10,000-record request remains under the service request-size limit.

For every batch size, the client sends the same deterministic prefix of `models/benchmark_payload.csv` for every warm-up and measured repetition. It records warm-up rows separately and does not include them in later analysis.

Generated files are stored in `results/raw/` and deliberately ignored by Git:

- `baseline_runs_<timestamp>.csv` contains one row per request.
- `baseline_metadata_<timestamp>.json` records the model hash reported by `/health`, payload SHA-256, protocol, and runtime information.

The client’s `client_http_round_trip_ms` starts after JSON serialization and ends after the HTTP response body is received. It therefore measures HTTP transport plus server processing, while the server additionally reports inference and total request timing.