from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque
from pathlib import Path

import joblib
import pandas as pd
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live BTC/ETH direction inference from Binance stream.")
    parser.add_argument("--symbol", default="btcusdt", help="Binance symbol, e.g. btcusdt.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("model_training/artifacts/btc_direction_model.joblib"),
        help="Path to trained model artifact.",
    )
    parser.add_argument("--depth-levels", type=int, default=5, help="Order book depth levels.")
    parser.add_argument("--interval-ms", type=int, default=1000, help="Binance stream interval ms.")
    parser.add_argument(
        "--momentum-periods",
        type=int,
        default=10,
        help="Periods used for momentum feature.",
    )
    return parser.parse_args()


def make_feature_row(bids: list[list[str]], asks: list[list[str]], mid_history: deque[float], momentum_periods: int) -> dict[str, float] | None:
    if len(bids) < 5 or len(asks) < 5:
        return None

    b_q0 = float(bids[0][1])
    a_q0 = float(asks[0][1])
    obi_denom = b_q0 + a_q0
    if obi_denom == 0:
        return None

    bid_depth = sum(float(q) for _, q in bids[:5])
    ask_depth = sum(float(q) for _, q in asks[:5])
    depth_denom = bid_depth + ask_depth
    if depth_denom == 0:
        return None

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    mid_history.append(mid)
    if len(mid_history) <= momentum_periods:
        return None

    old_mid = mid_history[-(momentum_periods + 1)]
    momentum = (mid / old_mid) - 1.0 if old_mid != 0 else 0.0
    return {
        "obi": (b_q0 - a_q0) / obi_denom,
        "spread": best_ask - best_bid,
        "depth_skew": (bid_depth - ask_depth) / depth_denom,
        "momentum_10s": momentum,
        "mid_price": mid,
    }


async def main_async() -> None:
    args = parse_args()
    if not args.artifact.is_file():
        raise SystemExit(f"Artifact not found: {args.artifact}")

    artifact = joblib.load(args.artifact)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = float(artifact["decision_threshold"])

    symbol = args.symbol.lower()
    stream_url = (
        f"wss://stream.binance.com:9443/ws/{symbol}@depth{args.depth_levels}@{args.interval_ms}ms"
    )
    mid_history: deque[float] = deque(maxlen=max(120, args.momentum_periods + 2))

    while True:
        try:
            print(f"connecting -> {stream_url}")
            async with websockets.connect(stream_url) as ws:
                while True:
                    payload = json.loads(await ws.recv())
                    bids = payload.get("bids", [])[: args.depth_levels]
                    asks = payload.get("asks", [])[: args.depth_levels]
                    row = make_feature_row(bids, asks, mid_history, args.momentum_periods)
                    if row is None:
                        continue

                    x = pd.DataFrame([row])[feature_cols]
                    proba_up = float(model.predict_proba(x)[0, 1])
                    pred_up = int(proba_up >= threshold)
                    out = {
                        "symbol": symbol,
                        "mid_price": row["mid_price"],
                        "proba_up": proba_up,
                        "pred_up": pred_up,
                        "threshold": threshold,
                    }
                    print(json.dumps(out))
        except Exception as exc:  # noqa: BLE001
            print(f"stream error: {exc}. reconnecting in 1s")
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Stopped.")
