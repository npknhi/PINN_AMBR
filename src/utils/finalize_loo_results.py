from __future__ import annotations

"""Finalize leave-one-bioreactor-out results from saved per-fold files."""

import argparse
from pathlib import Path

from src.utils.evaluation import (
    plot_saved_history,
    plot_saved_prediction_splits,
    write_leave_one_out_table,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create summary tables and figures from saved LOO outputs.")
    parser.add_argument("--results-dir", type=Path, required=True, help="LOO result directory, e.g. results/LOO_6_10000.")
    parser.add_argument(
        "--plots",
        nargs="*",
        default=("test", "val"),
        choices=("train", "val", "test"),
        help="Prediction splits to plot. Default: test val.",
    )
    parser.add_argument(
        "--history-plots",
        nargs="*",
        default=("loss", "r2", "r2_by_target"),
        choices=("loss", "r2", "r2_by_target"),
        help="History plots to create from checkpoints/history.csv. Default: loss r2 r2_by_target.",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    table_path = write_leave_one_out_table(results_dir)
    print(f"Wrote {table_path}")

    plot_paths = []
    history_plot_paths = []
    for fold_dir in sorted(results_dir.glob("fold_*_seed_*"), key=_fold_sort_key):
        history_plot_paths.extend(plot_saved_history(fold_dir, plots=tuple(args.history_plots)))
        plot_paths.extend(plot_saved_prediction_splits(fold_dir, splits=tuple(args.plots)))

    print(f"Wrote {len(plot_paths)} prediction figure(s)")
    print(f"Wrote {len(history_plot_paths)} history figure(s)")
    return 0


def _fold_sort_key(path: Path) -> tuple[int, str]:
    parts = path.name.split("_")
    try:
        return int(parts[1]), path.name
    except (IndexError, ValueError):
        return 10**9, path.name


if __name__ == "__main__":
    raise SystemExit(main())
