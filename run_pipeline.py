from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transform + training pipeline end-to-end."
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        required=True,
        help="Raw collector CSV input.",
    )
    parser.add_argument(
        "--transformed-csv",
        type=Path,
        default=Path("transform_data/l2_data_btcusdt_transformed.csv"),
        help="Output path for transformed dataset.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("model_training/artifacts"),
        help="Directory for model artifacts.",
    )
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable

    run_step(
        [
            python,
            "transform_data/transform_dataset.py",
            "--input-csv",
            str(args.raw_csv),
            "--output-csv",
            str(args.transformed_csv),
        ]
    )
    run_step(
        [
            python,
            "model_training/train_model.py",
            "--input-csv",
            str(args.transformed_csv),
            "--artifact-dir",
            str(args.artifact_dir),
        ]
    )
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
