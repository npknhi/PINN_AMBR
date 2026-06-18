from __future__ import annotations

"""Finalize leave-one-bioreactor-out results from saved per-fold files."""

import argparse
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.evaluation import (
    OBSERVABLE_TABLE_LABELS,
    plot_loss,
    plot_r2,
    plot_r2_by_target,
    plot_saved_observable_predictions,
)
from src.utils.training import _v1_metric_table


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
    metrics_path = results_dir / "leave_one_out_metrics_long.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    table = _v1_metric_table(
        metrics,
        benchmark="Leave-One-Bioreactor-Out",
        observable_labels=OBSERVABLE_TABLE_LABELS,
    )
    table_path = results_dir / "leave_one_out_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")

    plot_paths = []
    history_plot_paths = []
    for fold_dir in sorted(results_dir.glob("fold_*_seed_*"), key=_fold_sort_key):
        history_result = _history_result(fold_dir)
        if history_result is not None:
            if "loss" in args.history_plots:
                output_path = fold_dir / "loss.png"
                plot_loss(history_result, output_path=output_path)
                plt.close("all")
                history_plot_paths.append(output_path)
            if "r2" in args.history_plots:
                output_path = fold_dir / "r2_mean.png"
                plot_r2(history_result, output_path=output_path)
                plt.close("all")
                history_plot_paths.append(output_path)
            if "r2_by_target" in args.history_plots:
                output_path = fold_dir / "r2_by_target.png"
                plot_r2_by_target(history_result, output_path=output_path)
                plt.close("all")
                history_plot_paths.append(output_path)

        for split in args.plots:
            predictions_csv = fold_dir / f"predictions_{split}.csv"
            if not predictions_csv.exists():
                continue
            frame = pd.read_csv(predictions_csv, usecols=["Experiment_id"])
            experiment_ids = frame["Experiment_id"].dropna().astype(str).drop_duplicates().tolist()
            if split == "train" and len(experiment_ids) > 1:
                continue
            experiment_id = experiment_ids[0] if len(experiment_ids) == 1 else None
            label = experiment_id or split
            output_path = fold_dir / f"observable_predictions_{split}_{label}.png"
            plot_path = plot_saved_observable_predictions(
                predictions_csv,
                experiment_id=experiment_id,
                output_path=output_path,
            )
            plt.close("all")
            plot_paths.append(plot_path)

    print(f"Wrote {len(plot_paths)} prediction figure(s)")
    print(f"Wrote {len(history_plot_paths)} history figure(s)")
    return 0


def _fold_sort_key(path: Path) -> tuple[int, str]:
    parts = path.name.split("_")
    try:
        return int(parts[1]), path.name
    except (IndexError, ValueError):
        return 10**9, path.name


def _history_result(fold_dir: Path) -> dict | None:
    history_path = fold_dir / "checkpoints" / "history.csv"
    if not history_path.exists():
        return None
    history_frame = pd.read_csv(history_path)
    history = {
        column: history_frame[column].dropna().tolist()
        for column in history_frame.columns
    }
    config = SimpleNamespace(
        experiment_name=fold_dir.name,
        num_epochs=len(history_frame),
    )
    return {
        "history": history,
        "config": config,
        "output_dir": fold_dir,
    }


if __name__ == "__main__":
    raise SystemExit(main())
