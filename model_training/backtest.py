from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple directional backtest with trading costs.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("model_training/artifacts/btc_direction_model.joblib"),
        help="Model artifact path.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Transformed dataset CSV with mid_price.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override decision threshold.",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=1.0,
        help="Round-turn fee+slippage cost in bps per position change.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.artifact.is_file():
        raise SystemExit(f"Artifact not found: {args.artifact}")
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    artifact = joblib.load(args.artifact)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = float(args.threshold) if args.threshold is not None else float(artifact["decision_threshold"])

    df = pd.read_csv(args.input_csv)
    need = feature_cols + ["mid_price"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    work = df[need].copy()
    work[need] = work[need].apply(pd.to_numeric, errors="coerce")
    work.dropna(inplace=True)
    if len(work) < 3:
        raise SystemExit("Not enough rows for backtest.")

    probs = model.predict_proba(work[feature_cols])[:, 1]
    signal = (probs >= threshold).astype(int)

    # position at t predicts t+1 direction; shift by one row for realistic execution.
    position = pd.Series(signal, index=work.index).shift(1).fillna(0.0)
    returns = work["mid_price"].pct_change().fillna(0.0)

    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (args.fee_bps / 10_000.0)

    strategy_ret = position * returns - cost
    equity = (1.0 + strategy_ret).cumprod()
    buy_hold = (1.0 + returns).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    bh_return = float(buy_hold.iloc[-1] - 1.0)
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    sharpe = float(np.sqrt(365 * 24 * 60 * 60) * strategy_ret.mean() / (strategy_ret.std() + 1e-12))

    summary = {
        "rows": int(len(work)),
        "threshold": threshold,
        "fee_bps": args.fee_bps,
        "strategy_total_return": total_return,
        "buy_hold_return": bh_return,
        "max_drawdown": max_drawdown,
        "annualized_sharpe_approx": sharpe,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
