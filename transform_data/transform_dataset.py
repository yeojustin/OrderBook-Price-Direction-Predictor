from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform raw order book CSV into model-ready dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Raw input CSV from collector.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path for transformed output CSV.",
    )
    parser.add_argument(
        "--prediction-window-sec",
        type=int,
        default=30,
        help="Forward horizon for target label.",
    )
    parser.add_argument(
        "--momentum-periods",
        type=int,
        default=10,
        help="Periods for momentum feature (pct_change).",
    )
    return parser.parse_args()


def transform_dataset(df: pd.DataFrame, prediction_window_sec: int, momentum_periods: int) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "datetime"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["timestamp", "mid_price"]).copy()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    obi_denom = (df["b_q_0"] + df["a_q_0"]).replace(0, pd.NA)
    df["obi"] = (df["b_q_0"] - df["a_q_0"]) / obi_denom
    df["spread"] = df["a_p_0"] - df["b_p_0"]

    bid_depth = df[[f"b_q_{i}" for i in range(5)]].sum(axis=1)
    ask_depth = df[[f"a_q_{i}" for i in range(5)]].sum(axis=1)
    depth_denom = (bid_depth + ask_depth).replace(0, pd.NA)
    df["depth_skew"] = (bid_depth - ask_depth) / depth_denom
    df["momentum_10s"] = df["mid_price"].pct_change(periods=momentum_periods)
    df.dropna(subset=["momentum_10s"], inplace=True)

    base = df[["mid_price"]].copy().reset_index().rename(columns={"datetime": "t"})
    base["t_plus_window"] = base["t"] + pd.Timedelta(seconds=prediction_window_sec)

    future = base[["t", "mid_price"]].rename(
        columns={"t": "future_t", "mid_price": "future_mid_price"}
    )
    merged = pd.merge_asof(
        base.sort_values("t_plus_window"),
        future.sort_values("future_t"),
        left_on="t_plus_window",
        right_on="future_t",
        direction="forward",
    )
    merged.sort_index(inplace=True)

    df["future_mid_price"] = merged["future_mid_price"].values
    df["target"] = (df["future_mid_price"] > df["mid_price"]).astype("Int64")
    df.dropna(subset=["future_mid_price"], inplace=True)
    df["target"] = df["target"].astype(int)
    return df.reset_index()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    transformed = transform_dataset(
        df=df,
        prediction_window_sec=args.prediction_window_sec,
        momentum_periods=args.momentum_periods,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(args.output_csv, index=False)
    print(f"Saved transformed dataset: {args.output_csv} ({len(transformed)} rows)")


if __name__ == "__main__":
    main()
