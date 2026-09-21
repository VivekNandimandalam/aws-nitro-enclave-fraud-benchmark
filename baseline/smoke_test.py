"""Send a small fixed batch to the local native baseline service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--rows", default=4, type=int)
    args = parser.parse_args()
    payload = pd.read_csv(ROOT / "models" / "benchmark_payload.csv", nrows=args.rows)
    body = json.dumps({"instances": payload.to_numpy().tolist()}).encode("utf-8")
    request = Request(args.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))
    if len(result.get("predictions", [])) != args.rows:
        raise RuntimeError(f"Expected {args.rows} predictions; received {result}.")
    timing = result.get("timing", {})
    if not isinstance(timing.get("inference_ms"), (int, float)):
        raise RuntimeError(f"Inference timing was missing: {result}.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()