from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["obi", "spread", "depth_skew", "momentum_10s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BTC direction classifier.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Transformed CSV with target column.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("model_training/artifacts"),
        help="Directory to save model artifact and metrics.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    numeric_cols = [c for c in df.columns if c != "datetime"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    model_df = df[FEATURE_COLS + ["target"]].dropna().copy()
    model_df["target"] = model_df["target"].astype(int)

    x = model_df[FEATURE_COLS]
    y = model_df["target"]
    split_idx = int(len(model_df) * args.train_ratio)
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "log_reg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        rows.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1": float(f1_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            }
        )
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = str(results.iloc[0]["model"])
    best_model = fitted[best_name]

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = args.artifact_dir / f"btc_direction_model_{run_id}.joblib"
    latest_path = args.artifact_dir / "btc_direction_model.joblib"
    metrics_path = args.artifact_dir / f"metrics_{run_id}.json"

    artifact = {
        "model_name": best_name,
        "model": best_model,
        "feature_cols": FEATURE_COLS,
        "decision_threshold": 0.5,
        "prediction_window_sec": 30,
        "created_at_utc": run_id,
    }
    joblib.dump(artifact, artifact_path)
    joblib.dump(artifact, latest_path)
    metrics_path.write_text(results.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"Best model: {best_name}")
    print(results.to_string(index=False))
    print(f"Saved artifact: {artifact_path}")
    print(f"Updated latest artifact: {latest_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
