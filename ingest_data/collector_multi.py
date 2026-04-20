from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Binance depth snapshots for one or more symbols."
    )
    parser.add_argument(
        "--symbols",
        default="btcusdt,ethusdt",
        help="Comma-separated symbols (example: btcusdt,ethusdt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ingest_data/data"),
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--depth-levels",
        type=int,
        default=5,
        help="Depth levels to store per side.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1000,
        help="Binance depth stream interval in milliseconds.",
    )
    return parser.parse_args()


def csv_headers(depth_levels: int) -> list[str]:
    cols = ["timestamp", "mid_price"]
    cols += [f"b_p_{i}" for i in range(depth_levels)]
    cols += [f"b_q_{i}" for i in range(depth_levels)]
    cols += [f"a_p_{i}" for i in range(depth_levels)]
    cols += [f"a_q_{i}" for i in range(depth_levels)]
    return cols


def ensure_csv(path: Path, depth_levels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers(depth_levels))


async def stream_symbol(symbol: str, out_dir: Path, depth_levels: int, interval_ms: int) -> None:
    symbol = symbol.lower()
    out_path = out_dir / f"l2_data_{symbol.upper()}.csv"
    ensure_csv(out_path, depth_levels)
    stream_url = f"wss://stream.binance.com:9443/ws/{symbol}@depth{depth_levels}@{interval_ms}ms"

    while True:
        try:
            print(f"{symbol}: connecting -> {stream_url}")
            async with websockets.connect(stream_url) as ws:
                with out_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    while True:
                        payload = json.loads(await ws.recv())
                        bids = payload.get("bids", [])[:depth_levels]
                        asks = payload.get("asks", [])[:depth_levels]
                        if len(bids) < depth_levels or len(asks) < depth_levels:
                            continue

                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        mid = (best_bid + best_ask) / 2.0

                        # Keep raw strings for book levels to preserve exchange precision.
                        row: list[object] = [time.time(), f"{mid:.2f}"]
                        row += [p for p, q in bids] + [q for p, q in bids]
                        row += [p for p, q in asks] + [q for p, q in asks]
                        writer.writerow(row)
                        f.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: stream error -> {exc}. reconnecting in 1s")
            await asyncio.sleep(1)


async def main_async() -> None:
    args = parse_args()
    symbols = [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")

    tasks = [
        stream_symbol(
            symbol=s,
            out_dir=args.out_dir,
            depth_levels=args.depth_levels,
            interval_ms=args.interval_ms,
        )
        for s in symbols
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Stopped.")
