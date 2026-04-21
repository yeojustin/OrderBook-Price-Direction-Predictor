from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections import deque

import pandas as pd
import websockets
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tui_app.cli import load_artifact
from tui_app.features import make_feature_row
from tui_app.ui import (
    features_panel,
    footer_text,
    history_table,
    signal_panel,
    title_panel,
)

DRIFT_ZSCORE_MEAN_THRESHOLD = 0.8


def run_auto_training(args, c: Console) -> None:
    artifact_dir = args.artifact.parent
    cmd = [
        sys.executable,
        "model_training/train_model.py",
        "--input-csv",
        str(args.auto_train_csv),
        "--artifact-dir",
        str(artifact_dir),
    ]
    c.print(
        f"[dim]Auto-train: {args.auto_train_csv} -> {artifact_dir}[/dim]"
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        c.print("[red]Auto-train failed[/red]")
        if result.stderr:
            c.print(result.stderr.strip(), style="dim")
        return
    c.print("[green]Auto-train complete. Latest artifact updated.[/green]")


def load_baseline_feature_stats(csv_path, feature_cols: list[str]) -> tuple[pd.Series, pd.Series] | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if x.empty:
        return None
    means = x.mean()
    stds = x.std(ddof=0).replace(0, 1e-9)
    return means, stds


def compute_feature_drift_score(x: pd.DataFrame, baseline_means: pd.Series, baseline_stds: pd.Series) -> float:
    curr_means = x.mean()
    z = ((curr_means - baseline_means).abs() / baseline_stds).fillna(0.0)
    return float(z.mean())


async def run_live(args, artifact: dict) -> None:
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = float(artifact["decision_threshold"])
    model_name = artifact.get("model_name", "unknown")
    window_sec = int(artifact.get("prediction_window_sec", 30))

    symbol = args.symbol.lower().strip()
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth{args.depth_levels}@{args.interval_ms}ms"
    history: deque[float] = deque(maxlen=max(240, args.momentum_periods + 5))

    rows: list[dict] = []
    latest_feat: dict | None = None
    latest_mid = latest_bid = latest_ask = 0.0
    latest_p = 0.0
    latest_pred = 0
    status = "connecting"
    ticks = 0
    last_reload = time.time()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=8),
        Layout(name="signal", size=7),
        Layout(name="features", size=10),
        Layout(name="table", minimum_size=4),
        Layout(name="footer", size=3),
    )

    def render() -> Layout:
        mids = [float(r["mid_price"]) for r in rows[-48:]]
        probs = [float(r["proba_up"]) for r in rows[-48:]]
        layout["header"].update(title_panel())
        layout["signal"].update(
            signal_panel(
                symbol,
                latest_mid,
                latest_bid,
                latest_ask,
                latest_p,
                latest_pred,
                threshold,
                status,
                ticks,
                mids,
                probs,
            )
        )
        layout["features"].update(features_panel(latest_feat, model_name, window_sec))
        layout["table"].update(
            Panel(
                history_table(rows, args.max_rows),
                title="[dim]history (newest first)[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
        layout["footer"].update(
            Panel(
                footer_text(symbol, "live", ticks, status, model_name),
                border_style="dim",
                padding=(0, 0),
            )
        )
        return layout

    with Live(render(), refresh_per_second=4, screen=True) as live:
        while True:
            try:
                if args.model_reload_sec > 0 and (time.time() - last_reload) >= args.model_reload_sec:
                    artifact = load_artifact(args.artifact)
                    model = artifact["model"]
                    feature_cols = artifact["feature_cols"]
                    threshold = float(artifact["decision_threshold"])
                    model_name = artifact.get("model_name", "unknown")
                    last_reload = time.time()

                status = "connected"
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    while True:
                        payload = json.loads(await ws.recv())
                        bids = payload.get("bids", [])[: args.depth_levels]
                        asks = payload.get("asks", [])[: args.depth_levels]
                        feat = make_feature_row(bids, asks, history, args.momentum_periods)
                        if feat is None:
                            live.update(render())
                            continue

                        x = pd.DataFrame([feat])[feature_cols]
                        p_up = float(model.predict_proba(x)[0, 1])
                        pred = int(p_up >= threshold)
                        ticks += 1

                        latest_feat = feat
                        latest_mid = feat["mid_price"]
                        latest_bid = feat["best_bid"]
                        latest_ask = feat["best_ask"]
                        latest_p = p_up
                        latest_pred = pred

                        rows.append(
                            {
                                "ts": time.time(),
                                "mid_price": latest_mid,
                                "proba_up": p_up,
                                "pred_up": pred,
                                "obi": feat["obi"],
                                "spread": feat["spread"],
                                "depth_skew": feat["depth_skew"],
                                "momentum_10s": feat["momentum_10s"],
                            }
                        )
                        rows[:] = rows[-500:]
                        live.update(render())
            except KeyboardInterrupt:
                break
            except Exception:
                status = "reconnecting"
                live.update(render())
                await asyncio.sleep(1)


async def collect_batch_from_socket(args) -> pd.DataFrame:
    symbol = args.symbol.lower().strip()
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth{args.depth_levels}@{args.interval_ms}ms"
    history: deque[float] = deque(maxlen=max(240, args.momentum_periods + 5))
    rows: list[dict[str, float]] = []
    end_at = time.time() + max(args.batch_collect_sec, 1.0)

    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        while time.time() < end_at:
            payload = json.loads(await ws.recv())
            bids = payload.get("bids", [])[: args.depth_levels]
            asks = payload.get("asks", [])[: args.depth_levels]
            feat = make_feature_row(bids, asks, history, args.momentum_periods)
            if feat is None:
                continue
            rows.append(
                {
                    "mid_price": feat["mid_price"],
                    "obi": feat["obi"],
                    "spread": feat["spread"],
                    "depth_skew": feat["depth_skew"],
                    "momentum_10s": feat["momentum_10s"],
                }
            )
    return pd.DataFrame(rows)


def run_batch(args, artifact: dict) -> None:
    c = Console()
    c.print(title_panel())
    c.print()
    cycles = max(1, int(args.batch_cycles)) if args.batch_source == "socket" else 1
    baseline_stats: tuple[pd.Series, pd.Series] | None = None

    for cycle_idx in range(cycles):
        # Reload artifact each cycle so newly trained models can be picked up.
        current_artifact = artifact if cycle_idx == 0 else load_artifact(args.artifact)
        model = current_artifact["model"]
        feature_cols = current_artifact["feature_cols"]
        threshold = float(current_artifact["decision_threshold"])
        model_name = current_artifact.get("model_name", "unknown")

        if args.batch_source == "socket" and args.auto_train and baseline_stats is None:
            baseline_stats = load_baseline_feature_stats(args.auto_train_csv, feature_cols)
            if baseline_stats is None:
                c.print(
                    f"[yellow]Drift baseline unavailable from {args.auto_train_csv}; auto-train disabled.[/yellow]"
                )
                args.auto_train = False

        if args.batch_source == "socket":
            if cycles > 1:
                c.print(f"[dim]Cycle {cycle_idx + 1}/{cycles}[/dim]")
            c.print(
                f"[dim]Collecting live depth data for {args.batch_collect_sec:.1f}s from {args.symbol.upper()}...[/dim]"
            )
            try:
                df = asyncio.run(collect_batch_from_socket(args))
            except Exception as exc:
                raise SystemExit(f"Socket collection failed: {exc}") from exc
        else:
            if not args.input_csv.is_file():
                raise SystemExit(f"Input CSV not found: {args.input_csv}")
            df = pd.read_csv(args.input_csv)

        x = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if x.empty:
            raise SystemExit("No valid rows after conversion/collection")

        probs = model.predict_proba(x)[:, 1]
        preds = (probs >= threshold).astype(int)
        out = pd.DataFrame({"row_index": x.index, "proba_up": probs, "pred_up": preds})
        for col in feature_cols:
            out[col] = df[col].iloc[x.index].values
        out = out[out["row_index"] == args.row] if args.row >= 0 else out.tail(args.max_rows)

        t = Table(show_header=True, header_style="dim", border_style="dim", box=None, padding=(0, 1))
        t.add_column("ROW", justify="right", width=7)
        t.add_column("P(UP)", justify="right", width=8)
        t.add_column("SIGNAL", justify="center", width=8)
        t.add_column("OBI", justify="right", width=9)
        t.add_column("SPREAD", justify="right", width=9)
        t.add_column("DPTH SKW", justify="right", width=10)
        t.add_column("MOMENTUM", justify="right", width=11)

        for _, r in out.iterrows():
            sig = "▲ UP" if int(r["pred_up"]) == 1 else "▼ DN"
            sty = "green" if sig.startswith("▲") else "red"
            t.add_row(
                str(int(r["row_index"])),
                f"{float(r['proba_up']):.4f}",
                f"[{sty}]{sig}[/{sty}]",
                f"{float(r.get('obi', 0)):+.4f}",
                f"${float(r.get('spread', 0)):.2f}",
                f"{float(r.get('depth_skew', 0)):+.4f}",
                f"{float(r.get('momentum_10s', 0)):+.6f}",
            )

        c.print(Panel(t, title="[dim]batch inference[/dim]", border_style="dim"))
        c.print()
        c.print(
            Text.assemble(
                ("  source:", "dim"),
                (args.batch_source, "white"),
                ("  model:", "dim"),
                (model_name, "white"),
                ("  threshold:", "dim"),
                (f"{threshold:.4f}", "white"),
                ("  rows:", "dim"),
                (str(len(out)), "white"),
            )
        )

        if args.batch_source == "socket" and args.auto_train:
            assert baseline_stats is not None
            baseline_means, baseline_stds = baseline_stats
            drift_score = compute_feature_drift_score(x, baseline_means, baseline_stds)
            drift_flag = drift_score >= DRIFT_ZSCORE_MEAN_THRESHOLD
            c.print(
                f"[dim]drift_score(z-mean): {drift_score:.3f} "
                f"(threshold {DRIFT_ZSCORE_MEAN_THRESHOLD:.2f}) -> "
                f"{'DRIFT' if drift_flag else 'stable'}[/dim]"
            )
            if drift_flag:
                run_auto_training(args, c)

        if args.batch_source == "socket" and cycle_idx < cycles - 1:
            time.sleep(max(args.batch_pause_sec, 0.0))

