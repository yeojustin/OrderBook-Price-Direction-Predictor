from __future__ import annotations

import asyncio

from tui_app.cli import configure_interactive, load_artifact, parse_args
from tui_app.modes import run_batch, run_live


def main() -> None:
    args = configure_interactive(parse_args())
    artifact = load_artifact(args.artifact)
    if args.mode == "batch":
        run_batch(args, artifact)
    else:
        asyncio.run(run_live(args, artifact))


if __name__ == "__main__":
    main()
