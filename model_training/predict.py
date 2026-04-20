from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BTC direction inference from saved model artifact."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("model_training/artifacts/btc_direction_model.joblib"),
        help="Path to saved joblib artifact.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("transform_data/l2_data_btcusdt_transformed.csv"),
        help="CSV containing feature columns.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=-1,
        help="Row index to score. Default -1 means last row.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.artifact.is_file():
        raise SystemExit(f"Artifact not found: {args.artifact}")
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    artifact = joblib.load(args.artifact)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = float(artifact["decision_threshold"])

    df = pd.read_csv(args.input_csv)
    for col in feature_cols:
        if col not in df.columns:
            raise SystemExit(f"Missing feature column in CSV: {col}")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.dropna()
    if features.empty:
        raise SystemExit("No valid rows available after numeric conversion/dropna.")

    if args.row == -1:
        row = features.iloc[[-1]]
        row_idx = int(features.index[-1])
    else:
        if args.row not in features.index:
            raise SystemExit(
                f"Row {args.row} is unavailable after cleaning. "
                f"Try one of: {list(features.index[:5])} ... {list(features.index[-5:])}"
            )
        row = features.loc[[args.row]]
        row_idx = int(args.row)

    proba_up = float(model.predict_proba(row)[0, 1])
    pred_up = int(proba_up >= threshold)

    result = {
        "row_index": row_idx,
        "decision_threshold": threshold,
        "proba_up": proba_up,
        "pred_up": pred_up,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
