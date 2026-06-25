from __future__ import annotations

"""Leave-one-bioreactor-out experiment orchestration."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from src.utils.evaluation import (
    evaluate_observables,
    plot_saved_history,
    plot_saved_prediction_splits,
    save_reports,
    write_hierarchical_leave_one_out_table,
    write_leave_one_out_table,
)
from src.utils.splits import (
    load_pseudomonas_leave_one_bioreactor_out_split,
    make_leave_one_bioreactor_out_folds,
)
from src.utils.training import TrainingConfig, predict_dataset, train_pinn


def run_leave_one_bioreactor_out(
    config: TrainingConfig | None = None,
    seeds: Sequence[int] | None = None,
    fold_indices: Sequence[int] | None = None,
    keep_results: bool = False,
) -> dict[str, Any]:
    """Train selected LOO folds and append their per-seed results."""

    base_config = config or TrainingConfig()
    if base_config.train_experiment_ids is not None or base_config.test_experiment_ids is not None:
        raise ValueError("LOO expects experiment_ids, not explicit train/test ids.")
    seed_values = tuple(int(seed) for seed in (seeds or (base_config.seed,)))
    base_name = (base_config.experiment_name or "leave_one_bioreactor_out").strip()
    output_dir = Path(base_config.results_dir) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = make_leave_one_bioreactor_out_folds(
        _selected_experiment_frame(base_config.processed_csv, base_config.experiment_ids)
    )
    all_ids = tuple(value for fold in folds for value in fold)
    selected = None if fold_indices is None else {int(index) for index in fold_indices}
    if selected is not None:
        invalid = sorted(index for index in selected if index < 0 or index >= len(folds))
        if invalid:
            raise ValueError(f"Invalid fold indices {invalid}; valid range is 0 to {len(folds) - 1}.")

    kept_results = []
    for fold_index, test_ids in enumerate(folds):
        if selected is not None and fold_index not in selected:
            continue
        for seed in seed_values:
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = seed_dir / "leave_one_out_metrics_long.csv"
            runtime_path = seed_dir / "leave_one_out_runtime.csv"
            run_config = replace(
                base_config,
                seed=seed,
                results_dir=str(seed_dir),
                experiment_ids=all_ids,
                loo_fold_index=fold_index,
                experiment_name=f"fold_{fold_index}_{'_'.join(test_ids)}",
            )
            train_dataset, val_dataset, test_dataset, split_metadata = (
                load_pseudomonas_leave_one_bioreactor_out_split(
                    run_config.processed_csv,
                    experiment_ids=run_config.experiment_ids,
                    fold_index=fold_index,
                    validation_strategy=run_config.validation_strategy,
                    validation_seed=seed if run_config.validation_seed is None else run_config.validation_seed,
                )
            )
            started = perf_counter()
            result = train_pinn(
                run_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                split_metadata=split_metadata,
            )
            training_time_s = perf_counter() - started
            started = perf_counter()
            predict_dataset(result["model"], result["test_dataset"])
            inference_time_s = perf_counter() - started
            save_reports(result)

            _remove_fold_seed(metrics_path, fold_index, seed)
            _remove_fold_seed(runtime_path, fold_index, seed)
            runtime = pd.DataFrame(
                [{
                    "fold_index": fold_index,
                    "seed": seed,
                    "training_time_s": float(training_time_s),
                    "inference_time_s": float(inference_time_s),
                }]
            )
            _append_frame(runtime_path, runtime)
            metrics = evaluate_observables(result["model"], result["test_dataset"])
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "fold_index", fold_index)
            _append_frame(metrics_path, metrics)
            if keep_results:
                kept_results.append(result)

    seed_dirs = [output_dir / f"seed_{seed}" for seed in seed_values]
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "leave_one_out_metrics_long.csv"
        if metrics_path.exists():
            write_leave_one_out_table(seed_dir)
    metric_frames = [
        pd.read_csv(seed_dir / "leave_one_out_metrics_long.csv")
        for seed_dir in seed_dirs
        if (seed_dir / "leave_one_out_metrics_long.csv").exists()
    ]
    runtime_frames = [
        pd.read_csv(seed_dir / "leave_one_out_runtime.csv")
        for seed_dir in seed_dirs
        if (seed_dir / "leave_one_out_runtime.csv").exists()
    ]
    metrics_long = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    runtime = pd.concat(runtime_frames, ignore_index=True) if runtime_frames else pd.DataFrame()
    return {
        "output_dir": str(output_dir),
        "metrics_long": metrics_long,
        "runtime": runtime,
        "seed_dirs": [str(path) for path in seed_dirs],
        "results": kept_results,
    }


def summarize_leave_one_out(
    results_dir: str | Path,
    *,
    prediction_splits: tuple[str, ...] = ("test", "val"),
    history_plots: tuple[str, ...] = ("loss", "r2", "r2_by_target"),
) -> dict[str, Any]:
    """Create the hierarchical root table and regenerate requested LOO figures."""

    results_path = Path(results_dir)
    root_table = write_hierarchical_leave_one_out_table(results_path)
    prediction_figures = []
    history_figures = []
    for seed_dir in sorted(results_path.glob("seed_*")):
        for fold_dir in sorted(seed_dir.glob("fold_*"), key=_fold_sort_key):
            history_figures.extend(plot_saved_history(fold_dir, plots=history_plots))
            prediction_figures.extend(plot_saved_prediction_splits(fold_dir, splits=prediction_splits))
    return {
        "table": root_table,
        "prediction_figures": prediction_figures,
        "history_figures": history_figures,
    }


def _selected_experiment_frame(processed_csv: str | Path, experiment_ids: Sequence[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(processed_csv, usecols=["Experiment_id"])
    frame["Experiment_id"] = frame["Experiment_id"].astype(str)
    if experiment_ids is None:
        return frame.drop_duplicates().reset_index(drop=True)
    selected = tuple(str(value) for value in experiment_ids)
    available = set(frame["Experiment_id"].unique())
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Unknown Experiment_id values: {missing}")
    return frame[frame["Experiment_id"].isin(selected)].drop_duplicates().reset_index(drop=True)


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _remove_fold_seed(path: Path, fold_index: int, seed: int) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty or not {"fold_index", "seed"}.issubset(frame.columns):
        return
    keep = ~(
        pd.to_numeric(frame["fold_index"], errors="coerce").eq(fold_index)
        & pd.to_numeric(frame["seed"], errors="coerce").eq(seed)
    )
    frame.loc[keep].to_csv(path, index=False)


def _fold_sort_key(path: Path) -> tuple[int, str]:
    parts = path.name.split("_")
    try:
        return int(parts[1]), path.name
    except (IndexError, ValueError):
        return 10**9, path.name


__all__ = ["run_leave_one_bioreactor_out", "summarize_leave_one_out"]
