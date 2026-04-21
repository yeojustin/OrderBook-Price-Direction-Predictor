from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from rich.console import Console

from tui_app.ui import title_panel

DEFAULT_ARTIFACT = Path("model_training/artifacts/btc_direction_model.joblib")
DEFAULT_CSV = Path("transform_data/l2_data_btcusdt_transformed.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Order Book Price-Movement Predictor TUI")
    p.add_argument("--mode", choices=["live", "batch"], default=None)
    p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--depth-levels", type=int, default=5)
    p.add_argument("--interval-ms", type=int, default=1000)
    p.add_argument("--momentum-periods", type=int, default=10)
    p.add_argument("--max-rows", type=int, default=18)
    p.add_argument("--row", type=int, default=-1)
    p.add_argument("--batch-source", choices=["csv", "socket"], default="csv")
    p.add_argument("--batch-collect-sec", type=float, default=20.0)
    p.add_argument("--batch-cycles", type=int, default=1)
    p.add_argument("--batch-pause-sec", type=float, default=1.0)
    p.add_argument("--auto-train", action="store_true")
    p.add_argument("--auto-train-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--model-reload-sec", type=float, default=0.0)
    p.add_argument("--no-interactive", action="store_true")
    return p.parse_args()


def load_artifact(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Artifact not found: {path}")
    return joblib.load(path)


def choose_mode(default: str = "live") -> str:
    c = Console()
    c.print("[bold]Select mode[/bold]")
    c.print("  [green]1[/green]  Live inference   [dim](real-time Binance WebSocket)[/dim]")
    c.print("  [red]2[/red]  Batch inference  [dim](csv or socket collect mode)[/dim]")
    c.print()
    while True:
        choice = c.input("Choose mode [1/2]: ").strip()
        if choice in {"1", "2", ""}:
            break
        c.print("Please type [green]1[/green] or [red]2[/red].", style="dim")
    if choice == "":
        return default
    return "live" if choice == "1" else "batch"


def configure_batch_interactive(args: argparse.Namespace, c: Console) -> argparse.Namespace:
    c.print("[bold]Batch source[/bold]")
    c.print("  [green]1[/green]  CSV input", style="dim")
    c.print("  [red]2[/red]  Socket collect then score", style="dim")
    while True:
        src_choice = c.input("Choose batch source [1/2]: ").strip()
        if src_choice in {"1", "2", ""}:
            break
        c.print("Please type [green]1[/green] or [red]2[/red].", style="dim")

    args.batch_source = "socket" if src_choice == "2" else "csv"

    if args.batch_source == "socket":
        collect = c.input(f"Collect seconds [{args.batch_collect_sec:g}]: ").strip()
        if collect:
            args.batch_collect_sec = float(collect)

        cycles = c.input(f"Batch cycles [{args.batch_cycles}]: ").strip()
        if cycles:
            args.batch_cycles = int(cycles)

        pause = c.input(f"Pause between cycles sec [{args.batch_pause_sec:g}]: ").strip()
        if pause:
            args.batch_pause_sec = float(pause)

        auto_train = c.input("Enable drift-triggered auto-train? [y/N]: ").strip().lower()
        args.auto_train = auto_train in {"y", "yes"}
        if args.auto_train:
            csv_path = c.input(f"Auto-train CSV [{args.auto_train_csv}]: ").strip()
            if csv_path:
                args.auto_train_csv = Path(csv_path)
    else:
        csv_path = c.input(f"CSV path [{args.input_csv}]: ").strip()
        if csv_path:
            args.input_csv = Path(csv_path)

    c.print()
    return args


def configure_interactive(args: argparse.Namespace) -> argparse.Namespace:
    if args.no_interactive and args.mode is not None:
        return args
    c = Console()
    c.clear()
    c.print(title_panel())
    c.print()
    if args.mode is None and not args.no_interactive:
        args.mode = choose_mode()
    elif args.mode is None:
        args.mode = "live"
    if not args.no_interactive and args.mode == "batch":
        args = configure_batch_interactive(args, c)
    c.print()
    return args

