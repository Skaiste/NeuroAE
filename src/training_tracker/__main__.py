"""CLI entrypoint for training_tracker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch training tracker Streamlit app")
    parser.add_argument("--results-dir", default="results", help="Results directory containing index.jsonl")
    parser.add_argument(
        "--index-file",
        default=None,
        help="Index JSONL file to use (default: <results-dir>/index.jsonl)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for Streamlit server")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit server")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode")
    args, extra = parser.parse_known_args()

    os.environ["TRAINING_TRACKER_RESULTS_DIR"] = str(Path(args.results_dir))
    if args.index_file:
        os.environ["TRAINING_TRACKER_INDEX_FILE"] = str(Path(args.index_file))

    app_path = Path(__file__).parent / "ui" / "app.py"

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.headless else "false",
        *extra,
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
