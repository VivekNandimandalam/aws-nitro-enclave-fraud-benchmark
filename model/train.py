"""Train a deterministic, leakage-free XGBoost benchmark model."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "creditcard.csv"
ARTIFACTS = ROOT / "models"
SEED = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    if not DATASET.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    dataset = pd.read_csv(DATASET)
    expected = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}
    missing = sorted(expected - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    raw_features = dataset.drop(columns="Class")
    labels = dataset["Class"].astype("int64")
    raw_train, raw_test, y_train, y_test = train_test_split(
        raw_features, labels, test_size=TEST_SIZE, stratify=labels,
        random_state=SEED, shuffle=True,
    )

    # The scaler is fitted on training data only, preventing test-data leakage.
    scaled_columns = ["Time", "Amount"]
    scaler = RobustScaler().fit(raw_train[scaled_columns])

    def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        scaled = scaler.transform(transformed[scaled_columns])
        transformed["scaled_time"] = scaled[:, 0]
        transformed["scaled_amount"] = scaled[:, 1]
        return transformed.drop(columns=scaled_columns)

    x_train = preprocess(raw_train)
    x_test = preprocess(raw_test).loc[:, x_train.columns]
    feature_order = list(x_train.columns)
    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    hyperparameters = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "scale_pos_weight": scale_pos_weight, "random_state": SEED,
        "n_jobs": 1, "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**hyperparameters).fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype("int64")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "evaluation_partition": "held-out stratified test partition",
        "classification_threshold": THRESHOLD,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]), "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]), "true_positive": int(matrix[1, 1]),
        },
    }

    model_path = ARTIFACTS / "xgb_fraud.json"
    preprocessor_path = ARTIFACTS / "preprocessor.joblib"
    payload_path = ARTIFACTS / "benchmark_payload.csv"
    labels_path = ARTIFACTS / "benchmark_labels.csv"
    metrics_path = ARTIFACTS / "validation_metrics.json"
    metadata_path = ARTIFACTS / "model_metadata.json"
    manifest_path = ARTIFACTS / "SHA256SUMS.txt"
    model.save_model(model_path)
    joblib.dump({"schema_version": 1, "feature_order": feature_order, "scaled_columns": scaled_columns, "scaler": scaler}, preprocessor_path)
    x_test.to_csv(payload_path, index=False)
    pd.DataFrame({"source_row_index": y_test.index, "Class": y_test.to_numpy()}).to_csv(labels_path, index=False)
    write_json(metrics_path, metrics)
    hashes = {path.name: file_hash(path) for path in (model_path, preprocessor_path, payload_path, labels_path, metrics_path)}
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen artefacts for native-versus-Nitro-Enclave XGBoost inference benchmarking.",
        "dataset": {"local_file": DATASET.name, "sha256": file_hash(DATASET), "rows": int(len(dataset)), "columns": list(dataset.columns), "class_distribution": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}},
        "split": {"method": "train_test_split", "train_fraction": 1 - TEST_SIZE, "test_fraction": TEST_SIZE, "stratified_by": "Class", "random_seed": SEED, "train_rows": int(len(x_train)), "test_rows": int(len(x_test))},
        "preprocessing": {"fit_on_training_partition_only": True, "transformer": "sklearn.preprocessing.RobustScaler", "scaled_source_columns": scaled_columns, "output_feature_order": feature_order},
        "model": {"framework": "xgboost.XGBClassifier", "hyperparameters": hyperparameters},
        "evaluation": metrics,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__},
        "artifact_sha256": hashes,
        "benchmark_payload": {"file": payload_path.name, "rows": int(len(x_test)), "columns": feature_order, "labels_file": labels_path.name, "row_alignment": "Rows in benchmark_labels.csv align positionally with benchmark_payload.csv."},
    }
    write_json(metadata_path, metadata)
    manifest = [(path.name, file_hash(path)) for path in (model_path, preprocessor_path, payload_path, labels_path, metrics_path, metadata_path)]
    manifest_path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(manifest)), encoding="utf-8")
    print(f"Model SHA-256: {hashes[model_path.name]}")
    print(f"Benchmark rows: {len(x_test)}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Validation average precision: {metrics['average_precision']:.6f}")


if __name__ == "__main__":
    main()