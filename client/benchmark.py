"""Run repeatable HTTP benchmark measurements against an inference service."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = ROOT / "models" / "benchmark_payload.csv"
DEFAULT_OUTPUT = ROOT / "results" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, body: bytes, timeout_seconds: float) -> tuple[dict[str, Any], float]:
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter_ns()
    with urlopen(request, timeout=timeout_seconds) as response:
        response_bytes = response.read()
    client_round_trip_ms = (time.perf_counter_ns() - started) / 1_000_000
    return json.loads(response_bytes.decode("utf-8")), client_round_trip_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[10, 100, 1000, 10000])
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--environment", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0 or args.repetitions < 1 or any(size < 1 for size in args.batch_sizes):
        raise ValueError("Warmups must be non-negative; repetitions and batch sizes must be positive.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Batch sizes must be unique.")

    payload_path = args.payload.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = pd.read_csv(payload_path)
    if max(args.batch_sizes) > len(payload):
        raise ValueError(f"Largest batch size exceeds payload rows ({len(payload)}).")

    base_url = args.base_url.rstrip("/")
    health = get_json(f"{base_url}/health")
    if health.get("status") != "ok" or health.get("feature_count") != payload.shape[1]:
        raise RuntimeError(f"Service health/schema check failed: {health}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"{args.environment}_runs_{run_id}.csv"
    metadata_path = output_dir / f"{args.environment}_metadata_{run_id}.json"
    fields = [
        "timestamp_utc", "environment", "run_id", "batch_size", "run", "phase",
        "client_http_round_trip_ms", "server_inference_ms", "server_request_total_ms",
        "throughput_predictions_per_second", "prediction_count", "status", "error",
    ]
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "environment": args.environment,
        "started_at_utc": utc_now(),
        "service_base_url": base_url,
        "service_health": health,
        "payload": {"file": str(payload_path), "sha256": sha256(payload_path), "rows": int(len(payload)), "feature_count": int(payload.shape[1])},
        "protocol": {"warmup_runs_per_batch": args.warmup_runs, "measured_repetitions_per_batch": args.repetitions, "batch_sizes": args.batch_sizes, "client_timing_scope": "HTTP request send through response-body receipt; JSON request serialization occurs before timing."},
        "runtime": {"python": sys.version},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failures = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for batch_size in args.batch_sizes:
            # The same deterministic prefix of the frozen payload is used for
            # every repetition and must also be used in the enclave condition.
            body = json.dumps({"instances": payload.iloc[:batch_size].to_numpy().tolist()}, separators=(",", ":")).encode("utf-8")
            for phase, count in (("warmup", args.warmup_runs), ("measurement", args.repetitions)):
                for run in range(1, count + 1):
                    row: dict[str, Any] = {
                        "timestamp_utc": utc_now(), "environment": args.environment, "run_id": run_id,
                        "batch_size": batch_size, "run": run, "phase": phase,
                        "client_http_round_trip_ms": "", "server_inference_ms": "", "server_request_total_ms": "",
                        "throughput_predictions_per_second": "", "prediction_count": "", "status": "ok", "error": "",
                    }
                    try:
                        response, client_ms = post_json(f"{base_url}/predict", body, args.timeout_seconds)
                        prediction_count = len(response.get("predictions", []))
                        if prediction_count != batch_size:
                            raise RuntimeError(f"Expected {batch_size} predictions; received {prediction_count}.")
                        timing = response.get("timing", {})
                        server_inference = float(timing["inference_ms"])
                        server_total = float(timing["request_total_ms"])
                        row.update({
                            "client_http_round_trip_ms": client_ms,
                            "server_inference_ms": server_inference,
                            "server_request_total_ms": server_total,
                            "throughput_predictions_per_second": batch_size / (client_ms / 1000),
                            "prediction_count": prediction_count,
                        })
                    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, RuntimeError) as error:
                        failures += 1
                        row.update({"status": "error", "error": str(error)})
                    writer.writerow(row)
                    csv_file.flush()

    print(f"Raw results: {csv_path}")
    print(f"Run metadata: {metadata_path}")
    print(f"Measured observations: {len(args.batch_sizes) * args.repetitions}")
    print(f"Failures across warm-up and measurement: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()