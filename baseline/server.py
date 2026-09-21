"""Local native XGBoost inference service used as the benchmark baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "xgb_fraud.json"
DEFAULT_METADATA = ROOT / "models" / "model_metadata.json"
MAX_REQUEST_BYTES = 10 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceService:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_order = metadata["preprocessing"]["output_feature_order"]
        self.model_hash = sha256(model_path)
        expected_hash = metadata["artifact_sha256"][model_path.name]
        if self.model_hash != expected_hash:
            raise RuntimeError(
                f"Model hash mismatch: expected {expected_hash}, got {self.model_hash}. "
                "Regenerate or restore the frozen artefact before serving."
            )
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        if self.model.n_features_in_ != len(self.feature_order):
            raise RuntimeError("Model and metadata disagree on feature count.")

    def predict(self, instances: Any) -> dict[str, Any]:
        if not isinstance(instances, list) or not instances:
            raise ValueError("'instances' must be a non-empty JSON array.")
        try:
            values = np.asarray(instances, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("All instance values must be numeric.") from error
        expected_shape = (len(instances), len(self.feature_order))
        if values.ndim != 2 or values.shape != expected_shape:
            raise ValueError(
                f"'instances' must have shape [batch_size, {len(self.feature_order)}]; "
                f"received {list(values.shape)}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Instances must not contain NaN or infinite values.")

        started = time.perf_counter_ns()
        probabilities = self.model.predict_proba(pd.DataFrame(values, columns=self.feature_order))[:, 1]
        inference_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "predictions": [
                {"fraud_probability": float(probability), "predicted_class": int(probability >= 0.5)}
                for probability in probabilities
            ],
            "timing": {"inference_ms": inference_ms},
        }


class RequestHandler(BaseHTTPRequestHandler):
    service: InferenceService

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self.write_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "model_sha256": self.service.model_hash,
                "feature_count": len(self.service.feature_order),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self.write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            received = time.perf_counter_ns()
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError(f"Content-Length must be between 1 and {MAX_REQUEST_BYTES} bytes.")
            request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object.")
            response = self.service.predict(request.get("instances"))
            response["timing"]["request_total_ms"] = (time.perf_counter_ns() - received) / 1_000_000
            self.write_json(HTTPStatus.OK, response)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "detail": str(error)})
        except Exception as error:  # pragma: no cover - retained for operational diagnostics
            self.write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference_failed", "detail": str(error)})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--metadata", default=DEFAULT_METADATA, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    RequestHandler.service = InferenceService(args.model.resolve(), args.metadata.resolve())
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Baseline server listening on http://{args.host}:{args.port}")
    print(f"Model SHA-256: {RequestHandler.service.model_hash}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down baseline server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()